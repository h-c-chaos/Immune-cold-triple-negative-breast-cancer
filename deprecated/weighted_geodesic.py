"""
================================================================================
MÓDULO: WEIGHTED GEODESIC — Geodésica Ponderada por Barrera (P3)
================================================================================
PROBLEMA (Crítica 3):
  Dijkstra en grafo k=6 no ponderado = conteo de hops.
  En grilla hexagonal regular, hop ∝ Euclidiana.

SOLUCIÓN:
  Pesos en edges = 1 + Barrier_Score (o CAF abundance) del spot destino.
  Paths por regiones CAF-dense "cuestan más".
  Comparar: Euclidiana vs unweighted vs weighted.

  Si weighted ≈ unweighted → "topological separation, not barrier permeability"
  Si weighted >> unweighted → "barrier-weighted distance adds information"

LEE:   adata_with_phenotypes.h5ad (o adata_with_mechanism.h5ad)
       Requiere: .obsp['spatial_connectivities']
       Requiere: Barrier_Score o CAF abundance en .obs
ESCRIBE: results/weighted_geodesic/
         weighted_vs_unweighted.csv
         Fig_S_weighted_geodesic.pdf

Tiempo estimado: ~10 min (CPU, depende de n_spots)
Autores: Hugo Chancay, Daniel Lituma
================================================================================
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import Dict, Optional
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist
import json
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from config import PATHS
    BASE_DIR = PATHS.BASE_DIR
except ImportError:
    BASE_DIR = Path("/home/external/frjimenez/fabian/genoma")

from utils_stats import cohens_d_pooled, safe_mannwhitney

try:
    from mechanism_validation import find_cell_abundance_column
except ImportError:
    find_cell_abundance_column = None

OUTPUT_DIR = (PATHS.RESULTS_DIR if 'PATHS' in dir() else BASE_DIR / "results") / "weighted_geodesic"
RANDOM_SEED = 42


def _get_barrier_weights(adata: ad.AnnData) -> Optional[np.ndarray]:
    """
    Obtiene pesos de barrera por spot. Prioriza Barrier_Score_norm,
    luego Barrier_Score, luego CAF abundance.
    """
    for col in ['Barrier_Score_norm', 'Barrier_Score']:
        if col in adata.obs.columns:
            vals = adata.obs[col].values.astype(float)
            print(f"  Usando {col} como peso de barrera")
            return vals
    
    # Fallback: CAF abundance
    if find_cell_abundance_column is not None:
        col = find_cell_abundance_column(adata, 'CAF', 'q05')
        if col and not col.startswith('obsm:'):
            vals = adata.obs[col].values.astype(float)
            print(f"  Usando CAF abundance ({col}) como peso de barrera")
            return vals
    
    # Fallback manual
    for col in adata.obs.columns:
        if 'caf' in col.lower() and 'abundance' in col.lower():
            return adata.obs[col].values.astype(float)
    
    return None


def build_weighted_graph(adata: ad.AnnData, barrier_weights: np.ndarray) -> csr_matrix:
    """
    Construye grafo con pesos = 1 + normalizado(Barrier_Score del nodo destino).
    
    Edge (i→j) tiene peso = 1 + w_j donde w_j = Barrier_Score_norm del spot j.
    Spots con barrera alta son "caros" de atravesar.
    """
    conn = adata.obsp['spatial_connectivities']
    n = conn.shape[0]
    
    # Normalizar barrier weights a [0, 1]
    w = barrier_weights.copy()
    w = np.nan_to_num(w, nan=0.0)
    w_min, w_max = w.min(), w.max()
    if w_max > w_min:
        w_norm = (w - w_min) / (w_max - w_min)
    else:
        w_norm = np.zeros_like(w)
    
    # Construir grafo ponderado
    weighted = lil_matrix((n, n), dtype=float)
    rows, cols = conn.nonzero()
    
    for i, j in zip(rows, cols):
        # Peso = 1 (distancia base) + barrier_weight del nodo destino
        weighted[i, j] = 1.0 + w_norm[j]
    
    return weighted.tocsr()


def compute_distances_for_phenotype(
    adata: ad.AnnData,
    graph_unweighted: csr_matrix,
    graph_weighted: csr_matrix,
    target_cell: str = 'cDC1',
    source_cell: str = 'Tumor',
    phenotype: str = 'Immune_Excluded',
    max_sources: int = 500,
) -> Dict:
    """
    Calcula distancias Euclidiana, geodésica unweighted, y geodésica weighted
    desde spots Tumor-high a spots cDC1-high DENTRO de un fenotipo.
    """
    if 'Phenotype' not in adata.obs.columns:
        return {}
    
    pheno_mask = adata.obs['Phenotype'].values == phenotype
    if pheno_mask.sum() < 20:
        return {'n': 0, 'reason': f'{phenotype} has <20 spots'}
    
    # Obtener abundancias
    ab_key = 'means_cell_abundance_w_sf'
    if ab_key not in adata.obsm:
        return {'error': 'no_obsm'}
    
    ab = adata.obsm[ab_key]
    if not isinstance(ab, pd.DataFrame):
        return {'error': 'obsm_not_dataframe'}
    
    cdc1_col = [c for c in ab.columns if target_cell in c]
    tumor_col = [c for c in ab.columns if source_cell in c and 'Normal' not in c]
    
    if not cdc1_col or not tumor_col:
        return {'error': 'column_not_found'}
    
    pheno_idx = np.where(pheno_mask)[0]
    cdc1_vals = ab.iloc[pheno_idx][cdc1_col[0]].values
    tumor_vals = ab.iloc[pheno_idx][tumor_col[0]].values
    
    cdc1_thresh = max(np.percentile(cdc1_vals, 75), 0.1)
    tumor_thresh = max(np.percentile(tumor_vals, 75), 0.1)
    
    cdc1_high = pheno_idx[cdc1_vals >= cdc1_thresh]
    tumor_high = pheno_idx[tumor_vals >= tumor_thresh]
    
    if len(cdc1_high) < 3 or len(tumor_high) < 3:
        return {'n': 0, 'reason': 'insufficient high-abundance spots'}
    
    # Limitar sources para eficiencia
    np.random.seed(RANDOM_SEED)
    if len(tumor_high) > max_sources:
        tumor_high = np.random.choice(tumor_high, max_sources, replace=False)
    
    # 1. Euclidiana
    coords = adata.obsm['spatial']
    euc_dists = cdist(coords[tumor_high], coords[cdc1_high], 'euclidean')
    euc_min = euc_dists.min(axis=1)
    
    # 2. Geodésica unweighted
    try:
        sp_uw = dijkstra(graph_unweighted, indices=tumor_high, directed=False)
        geo_uw_min = np.array([sp_uw[i, cdc1_high].min() for i in range(len(tumor_high))])
        geo_uw_min = geo_uw_min[np.isfinite(geo_uw_min)]
    except Exception:
        geo_uw_min = np.array([])
    
    # 3. Geodésica weighted
    try:
        sp_w = dijkstra(graph_weighted, indices=tumor_high, directed=False)
        geo_w_min = np.array([sp_w[i, cdc1_high].min() for i in range(len(tumor_high))])
        geo_w_min = geo_w_min[np.isfinite(geo_w_min)]
    except Exception:
        geo_w_min = np.array([])
    
    return {
        'phenotype': phenotype,
        'n_sources': len(tumor_high),
        'n_targets': len(cdc1_high),
        'euclidean_median': float(np.median(euc_min)) if len(euc_min) > 0 else np.nan,
        'euclidean_mean': float(np.mean(euc_min)) if len(euc_min) > 0 else np.nan,
        'geodesic_uw_median': float(np.median(geo_uw_min)) if len(geo_uw_min) > 0 else np.nan,
        'geodesic_uw_mean': float(np.mean(geo_uw_min)) if len(geo_uw_min) > 0 else np.nan,
        'geodesic_w_median': float(np.median(geo_w_min)) if len(geo_w_min) > 0 else np.nan,
        'geodesic_w_mean': float(np.mean(geo_w_min)) if len(geo_w_min) > 0 else np.nan,
        'n_euc': len(euc_min),
        'n_geo_uw': len(geo_uw_min),
        'n_geo_w': len(geo_w_min),
    }


def run_weighted_geodesic(adata: ad.AnnData, save_dir: Path = None) -> Dict:
    """Pipeline completo: compara 3 métricas de distancia."""
    if save_dir is None:
        save_dir = OUTPUT_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("WEIGHTED GEODESIC ANALYSIS")
    print("=" * 80)
    
    # 1. Construir grafo
    if 'spatial_connectivities' not in adata.obsp:
        try:
            import squidpy as sq
            if 'sample_id' in adata.obs.columns:
                sq.gr.spatial_neighbors(adata, n_neighs=6, library_key='sample_id')
            else:
                sq.gr.spatial_neighbors(adata, n_neighs=6)
        except Exception as e:
            print(f"  [ERROR] Cannot build spatial graph: {e}")
            return {}
    
    conn = adata.obsp['spatial_connectivities']
    
    # Grafo unweighted (= hop count)
    graph_uw = conn.astype(float).copy()
    
    # 2. Obtener pesos de barrera
    barrier_weights = _get_barrier_weights(adata)
    
    if barrier_weights is None:
        print("  [WARN] No barrier weights found. Only unweighted analysis.")
        graph_w = graph_uw.copy()
        has_weighted = False
    else:
        graph_w = build_weighted_graph(adata, barrier_weights)
        has_weighted = True
    
    # 3. Calcular distancias por fenotipo
    results_rows = []
    for pheno in ['Immune_Desert', 'Immune_Excluded', 'Inflamed']:
        print(f"\n  Processing {pheno}...")
        res = compute_distances_for_phenotype(
            adata, graph_uw, graph_w, phenotype=pheno)
        if res and res.get('n_sources', 0) > 0:
            results_rows.append(res)
            print(f"    Euc={res.get('euclidean_median', 'N/A'):.1f}, "
                  f"Geo_UW={res.get('geodesic_uw_median', 'N/A'):.1f}, "
                  f"Geo_W={res.get('geodesic_w_median', 'N/A'):.1f}")
    
    df = pd.DataFrame(results_rows)
    
    # 4. Comparar weighted vs unweighted
    summary = {'has_weighted': has_weighted}
    
    if len(df) >= 2 and has_weighted:
        # ¿La geodésica ponderada agrega información?
        for i, row in df.iterrows():
            if np.isfinite(row.get('geodesic_uw_median', np.nan)) and \
               np.isfinite(row.get('geodesic_w_median', np.nan)):
                uw_med = row['geodesic_uw_median']
                w_med  = row['geodesic_w_median']
                uw_mean = row.get('geodesic_uw_mean', np.nan)
                w_mean  = row.get('geodesic_w_mean', np.nan)
                # FIX BUG-03: Inflamed mediana=0; usar mean como fallback
                if uw_med > 0:
                    ratio_w_uw = w_med / uw_med
                elif np.isfinite(uw_mean) and uw_mean > 0:
                    ratio_w_uw = w_mean / uw_mean
                else:
                    ratio_w_uw = np.nan
                ratio_str = f'{ratio_w_uw:.3f}' if np.isfinite(ratio_w_uw) else 'nan (median=0, mean=0)'
                print(chr(10) + "  " + str(row["phenotype"]) + ": weighted/unweighted ratio = " + str(ratio_str))
    
    # 5. Guardar
    if not df.empty:
        df.to_csv(save_dir / 'weighted_vs_unweighted.csv', index=False)
    
    with open(save_dir / 'weighted_geodesic_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n  Resultados guardados: {save_dir}")
    return summary


if __name__ == '__main__':
    adata = sc.read_h5ad(PATHS.PROCESSED_DIR / 'adata_with_phenotypes.h5ad')
    run_weighted_geodesic(adata)
