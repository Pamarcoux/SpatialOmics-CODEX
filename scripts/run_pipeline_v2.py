"""
Pipeline v2 — CLR standard + MECR filter
"""
import scanpy as sc
import spatialdata as sd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings('ignore')
sc.settings.verbosity = 0

ZARR = '/home/paul.marcoux/Postdoc/Python/CODEX_TMA/data/FFPE-IO60-Eq09-25TMA-22052026.zarr'
OUT = '/home/paul.marcoux/Postdoc/R/SpatialOmics-CODEX'
CELL_INFO = os.path.join(OUT, 'cell_info.npz')

MECR_PAIRS = [
    ("CD3e", "CD20"), ("CD3e", "CD79a"), ("CD3e", "CD68"),
    ("CD3e", "CD163"), ("CD3e", "Pan-Cytokeratin"), ("CD3e", "EpCAM"),
    ("CD20", "CD68"), ("CD20", "Pan-Cytokeratin"), ("CD79a", "CD68"),
    ("CD68", "Pan-Cytokeratin"), ("CD163", "EpCAM"), ("CD56", "CD20"),
    ("CD3e", "SMA"), ("CD20", "Vimentin"), ("CD68", "EpCAM"),
]
MECR_THRESHOLD = 1.0
MECR_MAX_PAIRS = 3


def clr(X):
    X_log = np.log1p(X)
    return X_log - X_log.mean(axis=1, keepdims=True)


def filter_qc(a):
    area_low = a.obs['area'].quantile(0.01)
    area_high = a.obs['area'].quantile(0.99)
    a = a[(a.obs['area'] >= area_low) & (a.obs['area'] <= area_high)]
    a = a[((a.X > 0).sum(1) >= 3).flatten()]
    return a


def mecr(adata):
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    n_violations = np.zeros(adata.n_obs, dtype=int)
    for m1, m2 in MECR_PAIRS:
        i1 = np.where(adata.var_names == m1)[0]
        i2 = np.where(adata.var_names == m2)[0]
        if len(i1) == 0 or len(i2) == 0:
            print(f'  MECR: marker not found: {m1} or {m2}', flush=True)
            continue
        n_violations += ((X[:, i1[0]] > MECR_THRESHOLD) & (X[:, i2[0]] > MECR_THRESHOLD))
    n_violations_over = (n_violations > MECR_MAX_PAIRS).sum()
    print(f'  MECR: {n_violations_over} cells with >{MECR_MAX_PAIRS} violations', flush=True)
    return n_violations <= MECR_MAX_PAIRS


def process_adata(a, label):
    print(f'\n=== {label} ===', flush=True)
    t0 = time.time()

    t1 = time.time()
    bg = np.percentile(a.X, 10, 0)
    a.X = (a.X - bg).clip(0)
    print(f'  BG sub: {time.time()-t1:.0f}s', flush=True)

    t1 = time.time()
    a.X = clr(a.X)
    print(f'  CLR std: {time.time()-t1:.1f}s', flush=True)

    t1 = time.time()
    keep = mecr(a)
    a = a[keep]
    print(f'  MECR: {a.n_obs} cells kept ({time.time()-t1:.1f}s)', flush=True)

    t1 = time.time()
    sc.pp.scale(a, max_value=10)
    print(f'  Scale: {time.time()-t1:.0f}s', flush=True)

    t1 = time.time()
    sc.pp.pca(a, n_comps=30)
    print(f'  PCA: {time.time()-t1:.0f}s', flush=True)

    t1 = time.time()
    sc.external.pp.bbknn(a, batch_key='tma_label', neighbors_within_batch=3, computation='pynndescent')
    print(f'  BBKNN: {time.time()-t1:.0f}s', flush=True)

    t1 = time.time()
    sc.tl.umap(a)
    print(f'  UMAP: {time.time()-t1:.0f}s', flush=True)

    t1 = time.time()
    sc.tl.leiden(a, flavor='igraph', n_iterations=-1, resolution=1.0, key_added='leiden')
    print(f'  Leiden: {a.obs["leiden"].nunique()} clusters, {time.time()-t1:.0f}s', flush=True)

    print(f'  Total: {time.time()-t0:.0f}s', flush=True)
    return a


def main():
    print('Loading zarr...', flush=True)
    t0 = time.time()
    sdata = sd.read_zarr(ZARR)
    adata_all = sdata.tables['FFPE_IO60_Eq09_25TMA_22052026_adata'].copy()
    print(f'Loaded: {adata_all.n_obs} cells in {time.time()-t0:.0f}s', flush=True)

    ci = np.load(CELL_INFO)
    cell_ids = ci['cell_ids']; y16 = ci['y16']; x16 = ci['x16']
    print(f'Cell info: {len(cell_ids)} cells in tissue', flush=True)

    # === ANALYSIS 0: Reference ===
    a0 = adata_all.copy()
    a0 = filter_qc(a0)
    print(f'QC (reference): {a0.n_obs} cells', flush=True)
    a0 = process_adata(a0, 'Analysis 0: Reference')
    a0.write_h5ad(os.path.join(OUT, 'data', 'processed', 'adata_processed_v2.h5ad'))
    print('Saved: adata_processed_v2.h5ad', flush=True)

    # === ANALYSIS 1: Tissue mask ===
    t1 = time.time()
    a1 = adata_all.copy()
    keep = np.isin(a1.obs['cell_ID'].values.astype(np.int64), cell_ids)
    a1 = a1[keep]
    print(f'Tissue filter: {a1.n_obs} cells ({time.time()-t1:.0f}s)', flush=True)
    a1 = filter_qc(a1)
    print(f'QC (tissue): {a1.n_obs} cells', flush=True)
    a1 = process_adata(a1, 'Analysis 1: Tissue mask')
    a1.write_h5ad(os.path.join(OUT, 'data', 'processed', 'adata_tissue_v2.h5ad'))
    print('Saved: adata_tissue_v2.h5ad', flush=True)

    # === ANALYSIS 2: Core circle 80% ===
    t1 = time.time()
    circles = sdata.shapes['tma_core']
    radius_80 = circles.radius.values[0] * 0.8
    in_any_core = np.zeros(len(cell_ids), dtype=bool)
    for _, row in circles.iterrows():
        cx, cy = row['x'], row['y']
        dist2 = (y16 - cy)**2 + (x16 - cx)**2
        in_any_core = in_any_core | (dist2 <= radius_80**2)
    core_cell_ids = cell_ids[in_any_core]
    print(f'Core circle 80%: {len(core_cell_ids)} cells ({time.time()-t1:.0f}s)', flush=True)

    a2 = adata_all.copy()
    keep2 = np.isin(a2.obs['cell_ID'].values.astype(np.int64), core_cell_ids)
    a2 = a2[keep2]
    print(f'Core filter: {a2.n_obs} cells', flush=True)
    a2 = filter_qc(a2)
    print(f'QC (core): {a2.n_obs} cells', flush=True)
    a2 = process_adata(a2, 'Analysis 2: Core circle 80%')
    a2.write_h5ad(os.path.join(OUT, 'data', 'processed', 'adata_core_v2.h5ad'))
    print('Saved: adata_core_v2.h5ad', flush=True)

    print('\nAll done!', flush=True)


if __name__ == '__main__':
    main()
