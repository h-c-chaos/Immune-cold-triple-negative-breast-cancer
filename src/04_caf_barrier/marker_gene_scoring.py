"""
================================================================================
MARKER GENE SCORING — Cross-Validation de Deconvolución
================================================================================
 Scoring SIMPLE de marcadores por spot (media de expresión) SIN
  deconvolución. Comparar con Cell2Location abundance por fenotipo.
  Si ambos muestran el mismo patrón → robustez confirmada.
================================================================================
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import Dict, Optional
from scipy.stats import spearmanr
from scipy.sparse import issparse
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from config import PATHS, SIGNATURES
    BASE_DIR = PATHS.BASE_DIR
except ImportError:
    BASE_DIR = Path("/home/external/frjimenez/fabian/genoma")

from utils_stats import cohens_d_pooled, safe_mannwhitney

OUTPUT_DIR = (PATHS.RESULTS_DIR if 'PATHS' in dir() else BASE_DIR / "results") / "marker_gene_scoring"

# Marcadores por tipo celular (literatura establecida, sin depender de Cell2Location)
MARKER_SETS = {
    'CAF': ['COL1A1', 'FAP', 'ACTA2', 'FN1', 'POSTN'],
    'CD8_T': ['CD8A', 'CD8B', 'GZMA', 'GZMB', 'PRF1'],
    'Macrophage': ['CD68', 'CD163', 'MRC1', 'APOE'],
    'cDC1': ['CLEC9A', 'XCR1', 'BATF3', 'IRF8'],
    'Tumor': ['EPCAM', 'KRT8', 'KRT18', 'KRT19'],
    'NK': ['NKG7', 'GNLY', 'KLRD1', 'KLRB1'],
    'B_Cell': ['CD79A', 'CD79B', 'MS4A1', 'CD19'],
    'Endothelial': ['PECAM1', 'VWF', 'CDH5'],
}


def _extract_expression(adata, gene: str) -> Optional[np.ndarray]:
    """Extrae expresión de .raw (prioridad) o .X."""
    source = adata.raw if adata.raw is not None else adata
    if gene in source.var_names:
        X = source[:, gene].X
        if issparse(X):
            return np.asarray(X.todense()).ravel()
        return np.asarray(X).ravel()
    return None


def calculate_marker_scores(adata: ad.AnnData) -> pd.DataFrame:
    """
    Calcula score de marcadores por spot como media de expresión.
    
    Score = mean(expresión de marcadores disponibles).
    NO usa deconvolución. Completamente ortogonal a Cell2Location.
    """
    print("\n" + "=" * 70)
    print("MARKER GENE SCORING (sin deconvolución)")
    print("=" * 70)
    
    scores = {}
    
    for ct, markers in MARKER_SETS.items():
        available = []
        for g in markers:
            expr = _extract_expression(adata, g)
            if expr is not None:
                available.append(expr)
        
        n_found = len(available)
        n_total = len(markers)
        
        if n_found >= 2:
            score = np.mean(available, axis=0)
            scores[f'Marker_{ct}'] = score
            print(f"  {ct}: {n_found}/{n_total} markers → score calculado")
        else:
            print(f"  {ct}: {n_found}/{n_total} markers → INSUFICIENTE (skip)")
    
    return pd.DataFrame(scores, index=adata.obs_names)


def compare_marker_vs_deconvolution(
    adata: ad.AnnData,
    marker_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Correlaciona marker scores con Cell2Location abundances.
    
    Spearman rho > 0.5: buena concordancia → deconvolución confiable.
    """
    print("\n" + "-" * 50)
    print("CORRELACIÓN: Marker Score vs Cell2Location")
    print("-" * 50)
    
    results = []
    
    # Extraer abundancias de obsm
    for obsm_key in ['means_cell_abundance_w_sf', 'q05_cell_abundance_w_sf']:
        if obsm_key in adata.obsm:
            ab = adata.obsm[obsm_key]
            if isinstance(ab, pd.DataFrame):
                break
    else:
        print("  [WARN] No abundance data in obsm")
        return pd.DataFrame()
    
    for ct in MARKER_SETS.keys():
        marker_col = f'Marker_{ct}'
        if marker_col not in marker_scores.columns:
            continue
        
        # Buscar columna de abundancia Cell2Location
        c2l_col = None
        for c in ab.columns:
            if ct in c:
                c2l_col = c
                break
        if c2l_col is None:
            continue
        
        marker_vals = marker_scores[marker_col].values
        c2l_vals = ab[c2l_col].values
        
        valid = np.isfinite(marker_vals) & np.isfinite(c2l_vals)
        if valid.sum() < 50:
            continue
        
        rho, pval = spearmanr(marker_vals[valid], c2l_vals[valid])
        
        results.append({
            'cell_type': ct,
            'spearman_rho': float(rho),
            'p_value': float(pval),
            'n_valid': int(valid.sum()),
            'concordance': 'HIGH' if rho > 0.5 else ('MODERATE' if rho > 0.3 else 'LOW'),
        })
        print(f"  {ct}: ρ={rho:.3f}, p={pval:.2e} → {results[-1]['concordance']}")
    
    return pd.DataFrame(results)


def test_caf_marker_by_phenotype(
    adata: ad.AnnData,
    marker_scores: pd.DataFrame,
) -> Dict:
    """
    Test CLAVE: ¿El marker score de CAF (sin deconvolución) muestra
    la misma diferencia Desert vs Excluded que Cell2Location?
    """
    print("\n" + "-" * 50)
    print("TEST CAF MARKER: Desert vs Excluded (sin deconvolución)")
    print("-" * 50)
    
    if 'Marker_CAF' not in marker_scores.columns:
        return {'error': 'No Marker_CAF score'}
    if 'Phenotype' not in adata.obs.columns:
        return {'error': 'No Phenotype column'}
    
    desert_mask = adata.obs['Phenotype'] == 'Immune_Desert'
    excluded_mask = adata.obs['Phenotype'] == 'Immune_Excluded'
    
    caf_desert = marker_scores.loc[desert_mask, 'Marker_CAF'].values
    caf_excluded = marker_scores.loc[excluded_mask, 'Marker_CAF'].values
    
    caf_desert = caf_desert[np.isfinite(caf_desert)]
    caf_excluded = caf_excluded[np.isfinite(caf_excluded)]
    
    if len(caf_desert) < 10 or len(caf_excluded) < 10:
        print(f"  n insuficiente: Desert={len(caf_desert)}, Excluded={len(caf_excluded)}")
        return {'error': 'insufficient_n'}
    
    d = cohens_d_pooled(caf_desert, caf_excluded)
    _, pval = safe_mannwhitney(caf_desert, caf_excluded)
    
    print(f"  Desert CAF marker: median={np.median(caf_desert):.3f} (n={len(caf_desert):,})")
    print(f"  Excluded CAF marker: median={np.median(caf_excluded):.3f} (n={len(caf_excluded):,})")
    print(f"  Cohen's d = {d:.3f}, p = {pval:.2e}")
    
    # Hipótesis: d < 0 (CAF menor en Desert)
    if d < -0.3 and pval < 0.05:
        verdict = "CONFIRMED — marker-based CAF difference replicates Cell2Location finding"
    elif d < 0:
        verdict = "WEAK — trend present but small effect or not significant"
    else:
        verdict = "NOT CONFIRMED — marker score does not show expected CAF pattern"
    
    print(f"  VERDICT: {verdict}")
    
    return {
        'cohens_d': float(d),
        'p_value': float(pval) if np.isfinite(pval) else None,
        'desert_median': float(np.median(caf_desert)),
        'excluded_median': float(np.median(caf_excluded)),
        'verdict': verdict,
    }


def run_marker_gene_scoring(adata: ad.AnnData, save_dir: Path = None) -> Dict:
    """Pipeline completo."""
    if save_dir is None:
        save_dir = OUTPUT_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("MARKER GENE SCORING — Cross-Validation Deconvolución")
    print("=" * 80)
    
    results = {}
    
    marker_scores = calculate_marker_scores(adata)
    correlation_df = compare_marker_vs_deconvolution(adata, marker_scores)
    caf_test = test_caf_marker_by_phenotype(adata, marker_scores)
    
    results['correlations'] = correlation_df.to_dict('records') if not correlation_df.empty else []
    results['caf_test'] = caf_test
    
    # Guardar
    if not correlation_df.empty:
        correlation_df.to_csv(save_dir / 'marker_vs_deconvolution.csv', index=False)
    
    with open(save_dir / 'marker_scoring_summary.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Resultados guardados: {save_dir}")
    return results


if __name__ == '__main__':
    adata = sc.read_h5ad(PATHS.PROCESSED_DIR / 'adata_with_phenotypes.h5ad')
    run_marker_gene_scoring(adata)
