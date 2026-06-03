"""
Pipeline v3 — Merge TMA (tissue mask) + Tonsil, corrected CLR + MECR
"""
import scanpy as sc
import spatialdata as sd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings('ignore')
sc.settings.verbosity = 0

ZARR_TMA = '/home/paul.marcoux/Postdoc/Python/CODEX_TMA/data/FFPE-IO60-Eq09-25TMA-22052026.zarr'
CELL_INFO = '/home/paul.marcoux/Postdoc/R/SpatialOmics-CODEX/cell_info.npz'
TONSIL = '/home/paul.marcoux/Postdoc/R/SpatialOmics-CODEX/data/raw/adata_merged_Test2_leiden_0_5.h5ad'
OUT = '/home/paul.marcoux/Postdoc/R/SpatialOmics-CODEX'

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
    sc.external.pp.bbknn(a, batch_key='batch', neighbors_within_batch=3, computation='pynndescent')
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
    # 1. Load TMA zarr → tissue mask filter
    print('Loading TMA zarr...', flush=True)
    t0 = time.time()
    sdata = sd.read_zarr(ZARR_TMA)
    adata_tma = sdata.tables['FFPE_IO60_Eq09_25TMA_22052026_adata'].copy()
    print(f'Loaded TMA: {adata_tma.n_obs} cells ({time.time()-t0:.0f}s)', flush=True)

    ci = np.load(CELL_INFO)
    cell_ids = ci['cell_ids']

    t1 = time.time()
    keep = np.isin(adata_tma.obs['cell_ID'].values.astype(np.int64), cell_ids)
    adata_tma = adata_tma[keep]
    print(f'Tissue mask: {adata_tma.n_obs} cells ({time.time()-t1:.0f}s)', flush=True)

    adata_tma.obs['dataset'] = 'TMA'

    # 2. Load tonsil → extract raw counts
    print('\nLoading tonsil...', flush=True)
    t1 = time.time()
    adata_tonsil = sc.read_h5ad(TONSIL)
    print(f'Loaded tonsil: {adata_tonsil.n_obs} cells ({time.time()-t1:.0f}s)', flush=True)

    # Raw counts from layer, discard existing uns/obsm (will be recomputed)
    adata_tonsil.X = adata_tonsil.layers['counts'].copy()
    for key in list(adata_tonsil.uns.keys()):
        del adata_tonsil.uns[key]
    for key in list(adata_tonsil.obsm.keys()):
        del adata_tonsil.obsm[key]
    if 'leiden' in adata_tonsil.obs.columns:
        del adata_tonsil.obs['leiden']

    adata_tonsil.obs['dataset'] = 'tonsil'

    # 3. Fix marker name (tonsil uses underscore, TMA uses slash)
    adata_tonsil.var.rename(index={'Keratin 8_18': 'Keratin 8/18'}, inplace=True)

    # 4. Unified batch column for BBKNN
    adata_tma.obs['batch'] = 'TMA_' + adata_tma.obs['tma_label'].astype(str)
    adata_tonsil.obs['batch'] = 'TONSIL_' + adata_tonsil.obs['batch'].astype(str)

    # 5. Concatenate
    print('\nConcatenating...', flush=True)
    t1 = time.time()
    adata = sc.concat([adata_tma, adata_tonsil], join='outer', index_unique='-')
    print(f'Concatenated: {adata.n_obs} cells ({time.time()-t1:.0f}s)', flush=True)
    print(f'  TMA:   {(adata.obs["dataset"] == "TMA").sum()} cells')
    print(f'  Tonsil: {(adata.obs["dataset"] == "tonsil").sum()} cells')
    print(f'  Batches: {adata.obs["batch"].nunique()}')

    # 6. QC (area percentile + min 3 markers)
    t1 = time.time()
    adata = filter_qc(adata)
    print(f'QC: {adata.n_obs} cells ({time.time()-t1:.0f}s)', flush=True)

    # 7. Run full pipeline
    adata = process_adata(adata, 'Merged v3')

    # 8. Save
    out_path = os.path.join(OUT, 'data', 'processed', 'adata_merged_v3.h5ad')
    adata.write_h5ad(out_path)
    print(f'\nSaved: {out_path}', flush=True)
    print('All done!', flush=True)


if __name__ == '__main__':
    main()
