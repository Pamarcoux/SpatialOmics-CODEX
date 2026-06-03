import scanpy as sc, spatialdata as sd, numpy as np, os, time, warnings
warnings.filterwarnings('ignore'); sc.settings.verbosity = 0
ZARR = '/home/paul.marcoux/Postdoc/Python/CODEX_TMA/data/FFPE-IO60-Eq09-25TMA-22052026.zarr'
OUT = '/home/paul.marcoux/Postdoc/R/SpatialOmics-CODEX'

print('Loading zarr...', flush=True)
t0 = time.time()
sdata = sd.read_zarr(ZARR)
adata_all = sdata.tables['FFPE_IO60_Eq09_25TMA_22052026_adata'].copy()
print(f'Loaded: {adata_all.n_obs} cells in {time.time()-t0:.0f}s', flush=True)

ci = np.load(os.path.join(OUT, 'cell_info.npz'))
cell_ids = ci['cell_ids']
y16 = ci['y16']
x16 = ci['x16']

def clr(X):
    m = X > 0
    log1p_x = np.log1p(X)
    row_sums = np.where(m, log1p_x, 0).sum(axis=1)
    gm = np.exp(row_sums / X.shape[1])
    return np.log1p(X / gm[:, None])

def filter_qc(a):
    area_low = a.obs['area'].quantile(0.01)
    area_high = a.obs['area'].quantile(0.99)
    a = a[(a.obs['area'] >= area_low) & (a.obs['area'] <= area_high)]
    a = a[((a.X > 0).sum(1) >= 3)]
    return a

def process_adata(a, label):
    print(f'\n=== {label} ===', flush=True)
    t0 = time.time()
    
    t1 = time.time()
    bg = np.percentile(a.X, 10, 0)
    a.X = (a.X - bg).clip(0)
    print(f'  BG sub: {time.time()-t1:.0f}s', flush=True)
    
    t1 = time.time()
    a.X = clr(a.X)
    print(f'  CLR: {time.time()-t1:.1f}s', flush=True)
    
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

# === ANALYSIS 1: Tissue mask ===
t1 = time.time()
a = adata_all.copy()
keep = np.isin(a.obs['cell_ID'].values.astype(np.int64), cell_ids)
a = a[keep]
print(f'Tissue filter: {a.n_obs} cells ({time.time()-t1:.0f}s)', flush=True)

a = filter_qc(a)
print(f'QC (tissue): {a.n_obs} cells', flush=True)

a = process_adata(a, 'Analysis 1: Tissue mask')
a.write_h5ad(os.path.join(OUT, 'data', 'processed', 'adata_tissue.h5ad'))
print('Saved: adata_tissue.h5ad', flush=True)

# === ANALYSIS 2: Core circle 80% ===
t1 = time.time()
circles = sdata.shapes['tma_core']
radius_80 = circles.radius.values[0] * 0.8  # ~51.32 mask-px

# Build mask: for each tissue cell, is it within 80% radius of any core?
in_any_core = np.zeros(len(cell_ids), dtype=bool)
for _, row in circles.iterrows():
    cx, cy = row['x'], row['y']
    # Distance from this cell centroid (16x-scale) to core center
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
a2.write_h5ad(os.path.join(OUT, 'data', 'processed', 'adata_core.h5ad'))
print('Saved: adata_core.h5ad', flush=True)
