"""
Extract PatchCore memorybank features (before sampling) and visualize with UMAP.

What it does:
1) For each bank (e.g. ai/nature), iterate TRAIN split and extract features:
   - `--viz_feature_source=patchcore`: patch embeddings via `PatchCore._embed` (memorybank features BEFORE sampling).
   - `--viz_feature_source=backbone`: raw backbone features from `feature_aggregator` (no PatchCore preprocessing/aggregation).
2) (Optional) Apply PatchCore `RandomSampler` (same coreset sampler used in training).
3) Randomly subsample points (UMAP is expensive on huge patch sets).
4) Standardize features and run UMAP -> 2D.
5) Save `umap_memorybank.png` (and optionally `.npz` artifacts).
"""

import argparse
import logging
import os
from typing import List, Tuple

import numpy as np
import torch

import patchcore.backbones
import patchcore.common
import patchcore.patchcore
import patchcore.sampler
from patchcore.datasets.tiny_genimage import Dataset, DatasetSplit


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("testmemorybank")


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
    pc: patchcore.patchcore.PatchCore, train_loader: torch.utils.data.DataLoader
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


@torch.no_grad()
def extract_backbone_features_raw(
    pc: patchcore.patchcore.PatchCore,
    train_loader: torch.utils.data.DataLoader,
    *,
    layer: str,
    pool: str,
    drop_cls: bool,
) -> np.ndarray:
    """
    Extract raw backbone features from PatchCore's `feature_aggregator` WITHOUT PatchCore preprocessing.

    - CNN layer output: [B, C, H, W]
        - pool="mean": returns [B, C] (mean over H/W)
        - pool="none": returns [B*H*W, C] (flatten spatial locations)
    - ViT/CLIP layer output: [B, N, C]
        - drop_cls=True: uses tokens[:, 1:, :] (drop CLS)
        - pool="mean": returns [B, C] (mean over tokens)
        - pool="none": returns [B*N, C] (flatten tokens)
    """
    agg = pc.forward_modules["feature_aggregator"].eval()
    chunks: List[torch.Tensor] = []

    for batch in train_loader:
        if isinstance(batch, dict):
            images = batch["image"]
        elif isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch

        images = images.to(torch.float).to(pc.device)
        feats_dict = agg(images)
        if layer not in feats_dict:
            raise KeyError(f"Layer {layer!r} not found in feature_aggregator outputs: {list(feats_dict.keys())}")
        f = feats_dict[layer]

        if f.ndim == 4:
            # [B, C, H, W]
            if pool == "mean":
                vec = f.mean(dim=(2, 3))  # [B, C]
            elif pool == "none":
                vec = f.permute(0, 2, 3, 1).reshape(-1, f.shape[1])  # [B*H*W, C]
            else:
                raise ValueError(f"Unknown pool={pool!r}, expected mean|none.")
        elif f.ndim == 3:
            # [B, N, C]
            if drop_cls and f.shape[1] > 1:
                f = f[:, 1:, :]
            if pool == "mean":
                vec = f.mean(dim=1)  # [B, C]
            elif pool == "none":
                vec = f.reshape(-1, f.shape[-1])  # [B*N, C]
            else:
                raise ValueError(f"Unknown pool={pool!r}, expected mean|none.")
        else:
            raise ValueError(f"Unexpected backbone feature shape for layer={layer!r}: {tuple(f.shape)}")

        chunks.append(vec.detach().cpu())

    if not chunks:
        raise RuntimeError("No features extracted (empty train loader).")
    return torch.cat(chunks, dim=0).numpy()


def run_umap(
    X: np.ndarray,
    *,
    seed: int | None,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    n_jobs: int,
) -> np.ndarray:
    import umap

    X = np.asarray(X, dtype=np.float32)
    # UMAP key parameters:
    # - n_neighbors: local neighborhood size (larger -> more global structure, slower).
    # - min_dist: how tightly points pack together (smaller -> tighter clusters).
    # - metric: distance metric in original space (euclidean/cosine/...).
    # - random_state: set for deterministic results; NOTE: setting it disables parallelism (n_jobs forced to 1).
    # - n_jobs: parallelism; use `-1` for all cores (only works when random_state is None).
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=str(metric),
        random_state=None if seed is None else int(seed),
        n_jobs=int(n_jobs),
    )
    return reducer.fit_transform(X)


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
            shuffle=False,  # keep deterministic order
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )

        input_shape = infer_input_shape(train_loader)
        backbone = patchcore.backbones.load(args.backbone_name)
        backbone.name = args.backbone_name
        pc = patchcore.patchcore.PatchCore(device)
        if args.viz_feature_source == "backbone":
            # IMPORTANT:
            # `feature_aggregator` only returns features for layers registered in `layers_to_extract_from`.
            # In backbone mode, we reuse `--layers_to_extract_from` to specify which raw layer to visualize.
            layers_to_extract_from = [args.layers_to_extract_from[0]]
        else:
            layers_to_extract_from = args.layers_to_extract_from
        pc.load(
            backbone=backbone,
            layers_to_extract_from=layers_to_extract_from,
            device=device,
            input_shape=input_shape,
            pretrain_embed_dimension=args.pretrain_embed_dimension,
            target_embed_dimension=args.target_embed_dimension,
            patchsize=args.patchsize,
            anomaly_score_num_nn=args.anomaly_scorer_k,
            # IMPORTANT: we want the "before sampling" features, so sampler doesn't matter for extraction.
            featuresampler=patchcore.sampler.IdentitySampler(),
            nn_method=patchcore.common.FaissNN(on_gpu=False, num_workers=4),
        )

        if args.viz_feature_source == "patchcore":
            feats = extract_memorybank_features_before_sampling(pc, train_loader)
            LOGGER.info("[%s] patchcore memory features (before sampling): %s", bank, feats.shape)
        elif args.viz_feature_source == "backbone":
            layer = layers_to_extract_from[0]
            feats = extract_backbone_features_raw(
                pc,
                train_loader,
                layer=layer,
                pool=str(args.backbone_pool),
                drop_cls=bool(args.backbone_drop_cls),
            )
            LOGGER.info("[%s] backbone raw features (layer=%s, pool=%s): %s", bank, layer, args.backbone_pool, feats.shape)
        else:
            raise ValueError(f"Unknown --viz_feature_source={args.viz_feature_source!r}")

        # Optional: apply PatchCore's RandomSampler BEFORE visualization.
        # This is the same RandomSampler used in `bin/run_patchcore.py` memorybank building.
        feats_for_viz = feats
        if args.bank_sampler == "random":
            if not (0.0 < float(args.bank_sampler_percentage) < 1.0):
                raise ValueError("--bank_sampler_percentage must be in (0,1) for RandomSampler.")
            np.random.seed(int(args.seed + bank_idx))  # RandomSampler uses np.random.choice internally
            sampler = patchcore.sampler.RandomSampler(float(args.bank_sampler_percentage))
            feats_for_viz = np.asarray(sampler.run(feats_for_viz))
            LOGGER.info("[%s] after RandomSampler(p=%.3f): %s", bank, args.bank_sampler_percentage, feats_for_viz.shape)

        # Subsample points for UMAP (raw patch count can be millions).
        rng = np.random.default_rng(args.seed + bank_idx)
        n = feats_for_viz.shape[0]
        take = min(int(args.max_points_per_bank), n)
        idx = rng.choice(n, size=take, replace=False) if take < n else np.arange(n)
        all_points.append(feats_for_viz[idx])
        all_labels.append(np.full((take,), bank_idx, dtype=int))
        legend.append(bank)

        # np.savez_compressed(os.path.join(out_dir, f"memorybank_raw_{bank}.npz"), features=feats)

    X = np.concatenate(all_points, axis=0)
    y = np.concatenate(all_labels, axis=0)

    if X.shape[0] < 10:
        raise RuntimeError(f"Too few points for UMAP: {X.shape}")

    # Standardize features so distance scale doesn't dominate UMAP.
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    Z = run_umap(
        X,
        seed=args.umap_seed,
        n_neighbors=int(args.umap_n_neighbors),
        min_dist=float(args.umap_min_dist),
        metric=str(args.umap_metric),
        n_jobs=int(args.umap_n_jobs),
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    for bank_idx, bank in enumerate(legend):
        m = (y == bank_idx)
        ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.6, label=bank)
    ax.set_title("UMAP of memorybank features (before sampling)")
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    out_png = os.path.join(out_dir, "umap_memorybank.png")
    fig.savefig(out_png, dpi=250)
    plt.close(fig)

    # np.savez_compressed(os.path.join(out_dir, "umap_memorybank.npz"), Z=Z, labels=y, legend=np.array(legend))
    LOGGER.info("Saved: %s", out_png)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    HOME = os.path.expanduser("~/dreamycore")

    parser.add_argument("--data_path", type=str, default=os.path.expanduser("~/datasets/tiny_genimage"))
    parser.add_argument("--dataset_names", nargs="+", default=[
        # 'adm',
        'biggan', 
        'glide', 
        # 'midjourney', 
        # 'sdv5', 
        # 'vqdm', 
        # 'wukong',
        # 'Chameleon',
        # "sdv5 two class"
                                                            ])  # "sdv5 two class"
    parser.add_argument("--banknames", nargs="+", default=["ai", "nature"])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=224)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--backbone_name", default="clip_vit_b16")  
    parser.add_argument(
        "--layers_to_extract_from",
        nargs="+",
        default=["transformer.resblocks.3"],
        help="PatchCore layers / (backbone mode) the raw backbone layer to visualize (use the first item).",
    )
    parser.add_argument("--pretrain_embed_dimension", type=int, default=0)
    parser.add_argument("--target_embed_dimension", type=int, default=512)
    parser.add_argument("--patchsize", type=int, default=14)
    parser.add_argument("--anomaly_scorer_k", type=int, default=3)

    parser.add_argument("--max_points_per_bank", type=int, default=4000)
    parser.add_argument(
        "--viz_feature_source",
        type=str,
        default="patchcore",
        choices=["patchcore", "backbone"],
        help="Which features to visualize: PatchCore patch embeddings, or raw backbone features.",
    )
    parser.add_argument(
        "--backbone_pool",
        type=str,
        default="none",
        choices=["mean", "none"],
        help="If --viz_feature_source=backbone: mean pool to [B,C] or keep all tokens/locations.",
    )
    parser.add_argument(
        "--backbone_drop_cls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If --viz_feature_source=backbone and features are [B,N,C]: drop CLS token (default: True).",
    )
    parser.add_argument(
        "--bank_sampler",
        type=str,
        default="identity",
        choices=["identity", "random"],
        help="Apply a sampler to raw memorybank features before UMAP (optional).",
    )
    parser.add_argument(
        "--bank_sampler_percentage",
        type=float,
        default=0.1,
        help="If --bank_sampler=random: keep this fraction of patch features (0,1).",
    )
    # UMAP important args:
    parser.add_argument(
        "--umap_seed",
        type=int,
        default=None,
        help="UMAP: random_state；默认 None（可并行、结果不完全可复现）。设为整数可复现，但会强制 n_jobs=1。",
    )
    parser.add_argument(
        "--umap_n_neighbors",
        type=int,
        default=20,
        help="UMAP: neighborhood size (bigger -> more global, slower). Typical: 5~50.",
    )
    parser.add_argument(
        "--umap_min_dist",
        type=float,
        default=1,
        help="UMAP: minimum distance in embedding (smaller -> tighter clusters). Typical: 0.0~0.5.",
    )
    parser.add_argument(
        "--umap_metric",
        type=str,
        default="euclidean",
        help="UMAP: distance metric in original space (euclidean/cosine/...).",
    )
    parser.add_argument(
        "--umap_n_jobs",
        type=int,
        default=-1,
        help="UMAP: 并行线程数（-1 表示用全部核心）；仅在 --umap_seed=None 时真正生效。",
    )
    parser.add_argument("--out_dir", type=str, default=os.path.join(HOME, "runs", "umap_backbone"))

    main(parser.parse_args())
