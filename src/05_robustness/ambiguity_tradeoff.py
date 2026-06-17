"""
================================================================================
MÓDULO: AMBIGUITY TRADEOFF — Curva Umbral vs Effect Size
================================================================================
  Sweep de umbrales de ambigüedad [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0].
  Para cada umbral reportar:
  - % spots clasificados como Ambiguous_Cold
  - n Desert, n Excluded
  - CAF Cohen's d entre Desert y Excluded
  - p-value del test

LEE:   adata_with_phenotypes.h5ad (necesita scores normalizados)
ESCRIBE: results/ambiguity_tradeoff/
         ambiguity_sweep.csv
         Fig_S_ambiguity_tradeoff.pdf
================================================================================
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import Dict
import json
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from config import PATHS, PHENOTYPE_PARAMS
except ImportError:
    PATHS = type('P', (), {'RESULTS_DIR': Path('results'), 'PROCESSED_DIR': Path('data/processed')})()
    PHENOTYPE_PARAMS = type('PP', (), {'TUMOR_PERCENTILE': 60, 'CD8_PERCENTILE': 75})()

from utils_stats import cohens_d_pooled, safe_mannwhitney

try:
    from mechanism_validation import find_cell_abundance_column
except ImportError:
    find_cell_abundance_column = None

OUTPUT_DIR = PATHS.RESULTS_DIR / "ambiguity_tradeoff"


def _get_caf_values(adata: ad.AnnData) -> np.ndarray:
    """Obtiene CAF abundance."""
    if find_cell_abundance_column is not None:
        col = find_cell_abundance_column(adata, 'CAF', 'q05')
        if col and not col.startswith('obsm:') and col in adata.obs.columns:
            return adata.obs[col].values.astype(float)
    
    # Fallback: buscar en obsm
    for key in ['q05_cell_abundance_w_sf', 'means_cell_abundance_w_sf']:
        if key in adata.obsm:
            data = adata.obsm[key]
            if isinstance(data, pd.DataFrame):
                for c in data.columns:
                    if 'CAF' in c:
                        return data[c].values.astype(float)
    return None


def sweep_ambiguity_thresholds(adata: ad.AnnData) -> pd.DataFrame:
    """
    Reclasifica con diferentes umbrales de ambigüedad y mide efecto.
    """
    print("\n" + "=" * 70)
    print("AMBIGUITY THRESHOLD SWEEP")
    print("=" * 70)
    
    # Verificar columnas necesarias
    sil_col = 'Silencing_Score_norm' if 'Silencing_Score_norm' in adata.obs.columns else 'Silencing_Score'
    bar_col = 'Barrier_Score_norm' if 'Barrier_Score_norm' in adata.obs.columns else 'Barrier_Score'
    tumor_col = 'Tumor_Score_norm' if 'Tumor_Score_norm' in adata.obs.columns else 'Tumor_Score'
    cd8_col = 'CD8_Score_norm' if 'CD8_Score_norm' in adata.obs.columns else 'CD8_Score'
    
    for col in [sil_col, bar_col, tumor_col, cd8_col]:
        if col not in adata.obs.columns:
            print(f"  [ERROR] Columna {col} no encontrada")
            return pd.DataFrame()
    
    # CAF values
    caf_vals = _get_caf_values(adata)
    if caf_vals is None:
        print("  [WARN] No CAF abundance — solo conteo de fenotipos")
    
    # Thresholds del pipeline
    t_thresh = np.percentile(adata.obs[tumor_col], PHENOTYPE_PARAMS.TUMOR_PERCENTILE)
    c_thresh = np.percentile(adata.obs[cd8_col], PHENOTYPE_PARAMS.CD8_PERCENTILE)
    
    # Masks fijas (no dependen de ambiguity)
    tumor_mask = adata.obs[tumor_col].values >= t_thresh
    cd8_high = adata.obs[cd8_col].values > c_thresh
    cold_mask = tumor_mask & ~cd8_high
    
    # Diferencia Silencing - Barrier en spots fríos
    diff = adata.obs[sil_col].values - adata.obs[bar_col].values
    
    n_cold = cold_mask.sum()
    print(f"  Cold tumor spots: {n_cold:,}")
    print(f"  Diff (Sil-Bar) in cold: mean={diff[cold_mask].mean():.3f}, std={diff[cold_mask].std():.3f}")
    
    # Sweep
    thresholds = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]
    results = []
    
    for thresh in thresholds:
        # Clasificar
        desert_mask = cold_mask & (diff > thresh)
        excluded_mask = cold_mask & (diff < -thresh)
        ambiguous_mask = cold_mask & (np.abs(diff) <= thresh)
        
        n_desert = desert_mask.sum()
        n_excluded = excluded_mask.sum()
        n_ambiguous = ambiguous_mask.sum()
        
        row = {
            'threshold': thresh,
            'n_desert': int(n_desert),
            'n_excluded': int(n_excluded),
            'n_ambiguous': int(n_ambiguous),
            'pct_ambiguous': float(100 * n_ambiguous / n_cold) if n_cold > 0 else 0,
            'pct_classified': float(100 * (n_desert + n_excluded) / n_cold) if n_cold > 0 else 0,
        }
        
        # CAF Cohen's d si hay datos suficientes
        if caf_vals is not None and n_desert >= 10 and n_excluded >= 10:
            caf_d = caf_vals[desert_mask]
            caf_e = caf_vals[excluded_mask]
            caf_d = caf_d[np.isfinite(caf_d)]
            caf_e = caf_e[np.isfinite(caf_e)]
            
            if len(caf_d) >= 10 and len(caf_e) >= 10:
                d = cohens_d_pooled(caf_d, caf_e)
                _, p = safe_mannwhitney(caf_d, caf_e)
                row['CAF_cohens_d'] = float(d)
                row['CAF_p_value'] = float(p) if np.isfinite(p) else np.nan
            else:
                row['CAF_cohens_d'] = np.nan
                row['CAF_p_value'] = np.nan
        else:
            row['CAF_cohens_d'] = np.nan
            row['CAF_p_value'] = np.nan
        
        results.append(row)
        
        # Marcar el threshold del pipeline
        marker = " ← PIPELINE" if abs(thresh - 0.1) < 0.001 else ""
        d_str = f"d={row['CAF_cohens_d']:.3f}" if np.isfinite(row.get('CAF_cohens_d', np.nan)) else "d=N/A"
        print(f"  τ={thresh:.3f}: Desert={n_desert:>6,} Excluded={n_excluded:>6,} "
              f"Ambig={n_ambiguous:>6,} ({row['pct_ambiguous']:.1f}%) {d_str}{marker}")
    
    return pd.DataFrame(results)


def plot_tradeoff(df: pd.DataFrame, save_dir: Path):
    """Genera figura de tradeoff."""
    if df.empty:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel A: Conteo de fenotipos vs threshold
    ax = axes[0]
    ax.plot(df['threshold'], df['n_desert'], 'o-', color='#d62728', label='Desert')
    ax.plot(df['threshold'], df['n_excluded'], 's-', color='#2C7BB6', label='Excluded')
    ax.plot(df['threshold'], df['n_ambiguous'], '^-', color='#9467bd', label='Ambiguous')
    ax.axvline(x=0.1, color='black', linestyle='--', alpha=0.3, label='Pipeline (τ=0.1)')
    ax.set_xlabel('Ambiguity Threshold (τ)')
    ax.set_ylabel('Number of Spots')
    ax.set_title('A. Phenotype Counts vs Threshold')
    ax.legend(fontsize=8)
    
    # Panel B: % Ambiguous vs threshold
    ax = axes[1]
    ax.plot(df['threshold'], df['pct_ambiguous'], 'ko-')
    ax.axvline(x=0.1, color='red', linestyle='--', alpha=0.3)
    ax.set_xlabel('Ambiguity Threshold (τ)')
    ax.set_ylabel('% Cold Spots Ambiguous')
    ax.set_title('B. Classification Confidence')
    
    # Panel C: CAF Cohen's d vs threshold
    ax = axes[2]
    valid = df['CAF_cohens_d'].notna()
    if valid.any():
        ax.plot(df.loc[valid, 'threshold'], df.loc[valid, 'CAF_cohens_d'], 'ro-')
        ax.axhline(y=-0.5, color='blue', linestyle='--', alpha=0.3, label='Medium effect')
        ax.axvline(x=0.1, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Ambiguity Threshold (τ)')
        ax.set_ylabel("CAF Cohen's d (Desert vs Excluded)")
        ax.set_title('C. Effect Size Stability')
        ax.legend(fontsize=8)
    
    plt.suptitle('Ambiguity Threshold Tradeoff Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_dir / 'Fig_S_ambiguity_tradeoff.png', dpi=300, bbox_inches='tight')
    fig.savefig(save_dir / 'Fig_S_ambiguity_tradeoff.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: Fig_S_ambiguity_tradeoff.pdf")


def run_ambiguity_tradeoff(adata: ad.AnnData, save_dir: Path = None) -> Dict:
    """Pipeline completo."""
    if save_dir is None:
        save_dir = OUTPUT_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    df = sweep_ambiguity_thresholds(adata)
    
    if not df.empty:
        df.to_csv(save_dir / 'ambiguity_sweep.csv', index=False)
        plot_tradeoff(df, save_dir)
    
    # Summary
    pipeline_row = df[abs(df['threshold'] - 0.1) < 0.001]
    summary = {
        'n_thresholds_tested': len(df),
        'pipeline_threshold': 0.1,
    }
    if not pipeline_row.empty:
        summary['pipeline_pct_ambiguous'] = float(pipeline_row['pct_ambiguous'].iloc[0])
    
    # ¿El efecto es estable?
    valid_d = df['CAF_cohens_d'].dropna()
    if len(valid_d) >= 3:
        summary['d_range'] = [float(valid_d.min()), float(valid_d.max())]
        summary['d_stable'] = bool(valid_d.std() < 0.2)
    
    with open(save_dir / 'ambiguity_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n  Resultados guardados: {save_dir}")
    return summary


if __name__ == '__main__':
    adata = sc.read_h5ad(PATHS.PROCESSED_DIR / 'adata_with_phenotypes.h5ad')
    run_ambiguity_tradeoff(adata)
