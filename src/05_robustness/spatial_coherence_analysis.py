"""
================================================================================
MÓDULO: SPATIAL COHERENCE ANALYSIS — Moran's I y Nichos 
================================================================================
  Spots se clasifican independientemente. No hay análisis de coherencia
  espacial de dominios fenotípicos. ¿Los spots Desert forman clusters
  compactos o están dispersos?

SOLUCIÓN:
  1. Moran's I para autocorrelación espacial de labels de fenotipo
  2. Join Count statistics para clustering de labels
  3. Distribución de tamaño de nichos contiguos

LEE:   adata_with_phenotypes.h5ad
       Requiere: .obsp['spatial_connectivities'], .obs['Phenotype']
ESCRIBE: results/spatial_coherence/
         morans_i_results.csv
         niche_sizes.csv
         Fig_S_spatial_coherence.pdf
================================================================================
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import Dict, List
from scipy.sparse import csr_matrix
from collections import Counter
import json
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from config import PATHS, PHENOTYPE_COLORS
except ImportError:
    PATHS = type('P', (), {'RESULTS_DIR': Path('results'), 'PROCESSED_DIR': Path('data/processed')})()
    PHENOTYPE_COLORS = {}

OUTPUT_DIR = PATHS.RESULTS_DIR / "spatial_coherence"
RANDOM_SEED = 42


def compute_morans_i_per_phenotype(
    adata: ad.AnnData,
    phenotypes: List[str] = None,
    n_permutations: int = 999,
) -> pd.DataFrame:
    """
    Calcula Moran's I para cada fenotipo como variable binaria.
    
    Moran's I > 0: clustering espacial (spots del mismo fenotipo agrupados)
    Moran's I ≈ 0: distribución aleatoria
    Moran's I < 0: dispersión (anti-clustering)
    
    Moran's I > 0.3 con p < 0.001 es evidencia fuerte
    de que los fenotipos forman dominios espaciales coherentes.
    """
    print("\n" + "=" * 70)
    print("MORAN'S I — Autocorrelación Espacial de Fenotipos")
    print("=" * 70)
    
    if 'spatial_connectivities' not in adata.obsp:
        try:
            import squidpy as sq
            if 'sample_id' in adata.obs.columns:
                sq.gr.spatial_neighbors(adata, n_neighs=6, library_key='sample_id')
            else:
                sq.gr.spatial_neighbors(adata, n_neighs=6)
        except Exception as e:
            print(f"  [ERROR] Cannot build graph: {e}")
            return pd.DataFrame()
    
    if 'Phenotype' not in adata.obs.columns:
        return pd.DataFrame()
    
    if phenotypes is None:
        phenotypes = ['Immune_Desert', 'Immune_Excluded', 'Inflamed']
    
    W = adata.obsp['spatial_connectivities']
    n = W.shape[0]
    
    # Row-standardize weights
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1  # avoid div by 0
    
    results = []
    
    for pheno in phenotypes:
        binary = (adata.obs['Phenotype'].values == pheno).astype(float)
        n_positive = binary.sum()
        
        if n_positive < 10:
            print(f"  {pheno}: n={n_positive} — skip (insuficiente)")
            continue
        
        # Moran's I = (n/S0) * (Σ_ij w_ij (x_i - x̄)(x_j - x̄)) / (Σ_i (x_i - x̄)²)
        x_bar = binary.mean()
        z = binary - x_bar
        
        # Numerador: Σ w_ij * z_i * z_j
        # Eficiente con sparse: z' * W * z
        numerator = float(z @ W @ z)
        
        # S0 = Σ w_ij
        S0 = float(W.sum())
        
        # Denominador
        denominator = float(np.sum(z ** 2))
        
        if denominator < 1e-10 or S0 < 1:
            continue
        
        I = (n / S0) * (numerator / denominator)
        
        # Permutation test
        np.random.seed(RANDOM_SEED)
        I_perm = np.zeros(n_permutations)
        for p in range(n_permutations):
            z_perm = np.random.permutation(z)
            num_perm = float(z_perm @ W @ z_perm)
            I_perm[p] = (n / S0) * (num_perm / denominator)
        
        p_value = (np.sum(np.abs(I_perm) >= abs(I)) + 1) / (n_permutations + 1)
        
        results.append({
            'phenotype': pheno,
            'n_spots': int(n_positive),
            'morans_I': float(I),
            'p_value': float(p_value),
            'expected_I': -1.0 / (n - 1),  # Esperado bajo H0
            'significant': p_value < 0.005,  # FIX BUG-01: min p con 999 perms = 0.001, umbral 0.005
            'interpretation': 'Clustered' if I > 0.1 else ('Random' if I > -0.1 else 'Dispersed'),
        })
        
        sig = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else "ns")
        print(f"  {pheno}: I={I:.4f}, p={p_value:.4f} [{sig}] "
              f"(n={n_positive:,}, {'clustered' if I > 0.1 else 'random'})")
    
    return pd.DataFrame(results)


def detect_niches(
    adata: ad.AnnData,
    phenotype: str = 'Immune_Desert',
    min_niche_size: int = 3,
) -> List[int]:
    """
    Detecta nichos contiguos de un fenotipo dado usando BFS en el grafo espacial.
    
    Retorna lista de tamaños de nichos (clusters de spots contiguos del mismo fenotipo).
    """
    if 'spatial_connectivities' not in adata.obsp:
        return []
    if 'Phenotype' not in adata.obs.columns:
        return []
    
    conn = adata.obsp['spatial_connectivities']
    is_target = adata.obs['Phenotype'].values == phenotype
    
    visited = np.zeros(adata.n_obs, dtype=bool)
    niche_sizes = []
    
    for start in np.where(is_target)[0]:
        if visited[start]:
            continue
        
        # BFS
        queue = [start]
        visited[start] = True
        size = 0
        
        while queue:
            node = queue.pop(0)
            size += 1
            
            neighbors = conn[node].nonzero()[1]
            for nb in neighbors:
                if not visited[nb] and is_target[nb]:
                    visited[nb] = True
                    queue.append(nb)
        
        if size >= min_niche_size:
            niche_sizes.append(size)
    
    return niche_sizes


def analyze_niche_sizes(adata: ad.AnnData) -> pd.DataFrame:
    """Analiza tamaños de nichos para cada fenotipo."""
    print("\n" + "=" * 70)
    print("NICHE SIZE ANALYSIS — Tamaños de Dominios Contiguos")
    print("=" * 70)
    
    results = []
    
    for pheno in ['Immune_Desert', 'Immune_Excluded', 'Inflamed']:
        n_total = (adata.obs['Phenotype'] == pheno).sum()
        if n_total < 10:
            print(f"  {pheno}: n={n_total} — skip")
            continue
        
        sizes = detect_niches(adata, pheno)
        
        if sizes:
            results.append({
                'phenotype': pheno,
                'n_niches': len(sizes),
                'total_spots': int(n_total),
                'median_size': float(np.median(sizes)),
                'mean_size': float(np.mean(sizes)),
                'max_size': int(np.max(sizes)),
                'min_size': int(np.min(sizes)),
                'pct_in_niches': float(100 * sum(sizes) / n_total),
            })
            print(f"  {pheno}: {len(sizes)} nichos, median={np.median(sizes):.0f}, "
                  f"max={np.max(sizes)}, {100*sum(sizes)/n_total:.1f}% en nichos")
        else:
            print(f"  {pheno}: 0 nichos detectados (spots dispersos)")
    
    return pd.DataFrame(results)


def run_spatial_coherence(adata: ad.AnnData, save_dir: Path = None) -> Dict:
    """Pipeline completo."""
    if save_dir is None:
        save_dir = OUTPUT_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("SPATIAL COHERENCE ANALYSIS")
    print("=" * 80)
    
    # 1. Moran's I por muestra (evitar artefactos cross-sample)
    all_morans = []
    if 'sample_id' in adata.obs.columns:
        samples = adata.obs['sample_id'].unique()
        print(f"\n  Calculando Moran's I por muestra ({len(samples)} muestras)...")
        for sid in samples:
            mask = adata.obs['sample_id'] == sid
            adata_s = adata[mask].copy()
            if adata_s.n_obs < 100:
                continue
            try:
                import squidpy as sq
                sq.gr.spatial_neighbors(adata_s, n_neighs=6)
                df_i = compute_morans_i_per_phenotype(adata_s, n_permutations=999)
                if not df_i.empty:
                    df_i['sample_id'] = sid
                    all_morans.append(df_i)
            except Exception:
                continue
        
        if all_morans:
            morans_df = pd.concat(all_morans, ignore_index=True)
        else:
            morans_df = pd.DataFrame()
    else:
        morans_df = compute_morans_i_per_phenotype(adata)
    
    # 2. Niche sizes (global)
    # construir grafo global si no existe (el loop per-sample no lo persiste en adata)
    if 'spatial_connectivities' not in adata.obsp:
        try:
            import squidpy as sq
            if 'sample_id' in adata.obs.columns:
                sq.gr.spatial_neighbors(adata, n_neighs=6, library_key='sample_id')
            else:
                sq.gr.spatial_neighbors(adata, n_neighs=6)
            print('  Grafo espacial global construido para niche detection')
        except Exception as _e:
            print(f'  [WARN] No se pudo construir grafo global: {_e}')
    niche_df = analyze_niche_sizes(adata)
    
    # 3. Guardar
    if not morans_df.empty:
        morans_df.to_csv(save_dir / 'morans_i_results.csv', index=False)
    if not niche_df.empty:
        niche_df.to_csv(save_dir / 'niche_sizes.csv', index=False)
    
    # 4. Summary
    summary = {}
    if not morans_df.empty:
        for pheno in morans_df['phenotype'].unique():
            sub = morans_df[morans_df['phenotype'] == pheno]
            summary[pheno] = {
                'mean_morans_I': float(sub['morans_I'].mean()),
                'pct_significant': float(100 * sub['significant'].mean()),
                'n_samples': len(sub),
            }
            print(f"\n  {pheno}: mean I={sub['morans_I'].mean():.4f}, "
                  f"{100*sub['significant'].mean():.0f}% significant across samples")
    
    with open(save_dir / 'spatial_coherence_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n  Resultados guardados: {save_dir}")
    return summary


if __name__ == '__main__':
    adata = sc.read_h5ad(PATHS.PROCESSED_DIR / 'adata_with_phenotypes.h5ad')
    run_spatial_coherence(adata)
