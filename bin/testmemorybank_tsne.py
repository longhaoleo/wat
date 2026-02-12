"""
Extract PatchCore memorybank features (before sampling) and visualize with t-SNE.

This is a t-SNE counterpart of `bin/testmemorybank.py` (UMAP).

Pipeline:
1) For each bank (e.g. ai/nature), iterate TRAIN split and extract patch embeddings via `PatchCore._embed`.
   This matches the "memorybank features BEFORE `featuresampler.run(...)`" stage.
2) Randomly subsample points (t-SNE is expensive on huge patch sets).
3) Standardize features and run t-SNE -> 2D.
4) Save `tsne_memorybank.png` and `.npz` artifacts.
"""

import argparse
import logging
import os
from typing import List, Tuple

import numpy as np
import torch

import wat.backbones
import wat.common
import wat.wat
import wat.sampler
from wat.datasets.tiny_genimage import Dataset, DatasetSplit


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("testmemorybank_tsne")


def infer_input_shape(dl) -> Tuple[int, int, int]:
    x = next(iter(dl))
    if isinstance(x, dict):
        x = x.get("image", next(iter(x.values())))
    if isinstance(x, (tuple, list)):
        x = x[0]
    if x.ndim == 4:
        return tuple(x.shape[1:4])  # (C,H,W)
    if x.ndim == 3:
        return tuple(x.shape)
    raise RuntimeError(f"Unexpected batch shape: {getattr(x,'shape',None)}")


@torch.no_grad()
def extract_memorybank_features_before_sampling(
    pc: wat.wat.WAT, train_loader: torch.utils.data.DataLoader
) -> np.ndarray:
    """
    Iterate the train loader and collect patch embeddings:
    - This is exactly the "memorybank features BEFORE sampling".
    - Output shape is roughly [N_patches_total, D] (very large).
    """
    _ = pc.forward_modules.eval()
    chunks: List[torch.Tensor] = []
    for batch in train_loader:
        if isinstance(batch, dict):
            images = batch["image"]
        elif isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch
        images = images.to(torch.float).to(pc.device)
        batch_feat = pc._embed(images, detach=False)  # Tensor [N_patches, D]
        chunks.append(batch_feat.detach().cpu())

    if not chunks:
        raise RuntimeError("No features extracted (empty train loader).")
    return torch.cat(chunks, dim=0).numpy()


def run_tsne(
    X: np.ndarray,
    *,
    seed: int | None,
    perplexity: float,
    learning_rate: str | float,
    init: str,
    metric: str,
    max_iter: int,
) -> np.ndarray:
    from sklearn.manifold import TSNE

    X = np.asarray(X, dtype=np.float32)

    # t-SNE key parameters:
    # - perplexity: neighborhood scale (Typical: 5~50; must be < N).
    # - learning_rate: optimization step (use "auto" unless you know what you're doing).
    # - init: "pca" is usually more stable than "random".
    # - metric: distance in original space (euclidean/cosine/...) for feature vectors.
    # - random_state: set for deterministic results.
    tsne = TSNE(
        n_components=2,
        perplexity=float(perplexity),
        learning_rate=learning_rate,
        init=init,
        metric=metric,
        random_state=None if seed is None else int(seed),
        max_iter=int(max_iter),
    )
    return tsne.fit_transform(X)


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    LOGGER.info("Using device: %s", device)

    data_path = os.path.expanduser(args.data_path)
    dataset_names = list(args.dataset_names)
    banknames = list(args.banknames)

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    all_points: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    legend: List[str] = []

    for bank_idx, bank in enumerate(banknames):
        # Build TRAIN loader for a specific bank: <dataset>/train/<bank>/
        datasets = []
        for ds_name in dataset_names:
            root = os.path.join(data_path, ds_name)
            ds = Dataset(
                source=root,
                split=DatasetSplit.TRAIN,
                bankname=bank,
                resize=args.resize,
                imagesize=args.imagesize,
                seed=args.seed,
            )
            if len(ds) == 0:
                LOGGER.warning("Empty train split: %s/train/%s", ds_name, bank)
                continue
            datasets.append(ds)
        if not datasets:
            raise RuntimeError(f"No train samples found under {data_path} for bank={bank} datasets={dataset_names}")

        train_ds = datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )

        input_shape = infer_input_shape(train_loader)
        backbone = wat.backbones.load(args.backbone_name)
        backbone.name = args.backbone_name
        pc = wat.wat.WAT(device)
        pc.load(
            backbone=backbone,
            layers_to_extract_from=args.layers_to_extract_from,
            device=device,
            input_shape=input_shape,
            pretrain_embed_dimension=args.pretrain_embed_dimension,
            target_embed_dimension=args.target_embed_dimension,
            patchsize=args.patchsize,
            anomaly_score_num_nn=args.anomaly_scorer_k,
            featuresampler=wat.sampler.IdentitySampler(),
            nn_method=wat.common.FaissNN(on_gpu=False, num_workers=4),
        )

        feats = extract_memorybank_features_before_sampling(pc, train_loader)
        LOGGER.info("[%s] raw memory features (before sampling): %s", bank, feats.shape)

        # Subsample points for t-SNE.
        rng = np.random.default_rng(args.seed + bank_idx)
        n = feats.shape[0]
        take = min(int(args.max_points_per_bank), n)
        idx = rng.choice(n, size=take, replace=False) if take < n else np.arange(n)
        all_points.append(feats[idx])
        all_labels.append(np.full((take,), bank_idx, dtype=int))
        legend.append(bank)

        # np.savez_compressed(os.path.join(out_dir, f"memorybank_raw_{bank}.npz"), features=feats)

    X = np.concatenate(all_points, axis=0)
    y = np.concatenate(all_labels, axis=0)

    if X.shape[0] < 10:
        raise RuntimeError(f"Too few points for t-SNE: {X.shape}")
    if args.tsne_perplexity >= X.shape[0]:
        raise ValueError(f"--tsne_perplexity must be < N (N={X.shape[0]}), got {args.tsne_perplexity}")

    # Standardize features so distance scale doesn't dominate t-SNE.
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    Z = run_tsne(
        X,
        seed=args.tsne_seed,
        perplexity=float(args.tsne_perplexity),
        learning_rate=args.tsne_learning_rate,
        init=str(args.tsne_init),
        metric=str(args.tsne_metric),
        max_iter=int(args.tsne_max_iter),
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    for bank_idx, bank in enumerate(legend):
        m = (y == bank_idx)
        ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.6, label=bank)
    ax.set_title("t-SNE of memorybank features (before sampling)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    out_png = os.path.join(out_dir, "tsne_memorybank.png")
    fig.savefig(out_png, dpi=250)
    plt.close(fig)

    # np.savez_compressed(os.path.join(out_dir, "tsne_memorybank.npz"), Z=Z, labels=y, legend=np.array(legend))
    LOGGER.info("Saved: %s", out_png)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    HOME = os.path.expanduser("~/dreamycore")

    parser.add_argument("--data_path", type=str, default=os.path.expanduser("~/datasets/tiny_genimage 1"))
    parser.add_argument("--dataset_names", nargs="+", default=["sdv5 three class"])
    parser.add_argument("--banknames", nargs="+", default=["ai", "nature"])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=224)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--backbone_name", default="clip_vit_b16")
    parser.add_argument("--layers_to_extract_from", nargs="+", default=["transformer.resblocks.3"])
    parser.add_argument("--pretrain_embed_dimension", type=int, default=0)
    parser.add_argument("--target_embed_dimension", type=int, default=512)
    parser.add_argument("--patchsize", type=int, default=5)
    parser.add_argument("--anomaly_scorer_k", type=int, default=3)

    parser.add_argument("--max_points_per_bank", type=int, default=4000)
    # t-SNE important args:
    parser.add_argument(
        "--tsne_seed",
        type=int,
        default=0,
        help="t-SNE: random_state；设为整数可复现。",
    )
    parser.add_argument(
        "--tsne_perplexity",
        type=float,
        default=20,
        help="t-SNE: perplexity（邻域规模）。Typical: 5~50；必须 < 样本数 N。",
    )
    parser.add_argument(
        "--tsne_learning_rate",
        default="auto",
        help="t-SNE: learning_rate（优化步长）。推荐用 auto。",
    )
    parser.add_argument(
        "--tsne_init",
        type=str,
        default="pca",
        help="t-SNE: init（初始化）。pca 通常更稳定；也可 random。",
    )
    parser.add_argument(
        "--tsne_metric",
        type=str,
        default="euclidean",
        help="t-SNE: metric（原空间距离）。特征向量有时用 cosine。",
    )
    parser.add_argument(
        "--tsne_max_iter",
        type=int,
        default=1000,
        help="t-SNE: max_iter（迭代次数）。常用 1000~5000。",
    )
    parser.add_argument("--out_dir", type=str, default=os.path.join(HOME, "runs", "tsne_memorybank"))

    main(parser.parse_args())
