"""
================================================================================
MÓDULO: COMPREHENSIVE CELL TYPE ANALYSIS — 15 tipos × 5 fenotipos (P5)
================================================================================
  Heatmap de TODOS los 15 tipos × 5 fenotipos (Cohen's d o median fold-change).
  Flagear patrones inesperados. Supplementary figure.

LEE:   adata_with_phenotypes.h5ad
ESCRIBE: results/comprehensive_celltype/
         all_celltypes_by_phenotype.csv
         Fig_S_all_celltypes_heatmap.pdf

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
import seaborn as sns

try:
    from config import PATHS, PHENOTYPE_COLORS
except ImportError:
    from pathlib import Path as _P
    PATHS = type('P', (), {'RESULTS_DIR': _P('results'), 'PROCESSED_DIR': _P('data/processed')})()
    PHENOTYPE_COLORS = {}

from utils_stats import cohens_d_pooled, apply_fdr, safe_mannwhitney

CELL_TYPES_15 = [
    'B_Cell', 'CAF', 'CD4_T', 'CD8_T', 'cDC1', 'Endothelial',
    'Macrophage', 'Monocyte', 'Myeloid_Cycling', 'NK', 'NKT',
    'Normal_Epithelial', 'PVL', 'T_Cell_Cycling', 'Tumor',
]

PHENOTYPES = ['Immune_Desert', 'Immune_Excluded', 'Inflamed', 'Normal_Stroma']

OUTPUT_DIR = PATHS.RESULTS_DIR / "comprehensive_celltype"


def _get_abundance_df(adata: ad.AnnData) -> pd.DataFrame:
    """Extrae todas las abundancias celulares como DataFrame limpio."""
    for key in ['means_cell_abundance_w_sf', 'q05_cell_abundance_w_sf']:
        if key in adata.obsm:
            data = adata.obsm[key]
            if isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                df = pd.DataFrame(data, index=adata.obs_names)
            # Limpiar nombres
            clean = {}
            for c in df.columns:
                name = c
                for pfx in ['meanscell_abundance_w_sf_', 'means_cell_abundance_w_sf_',
                            'q05cell_abundance_w_sf_', 'q05_cell_abundance_w_sf_']:
                    name = name.replace(pfx, '')
                clean[c] = name
            df = df.rename(columns=clean)
            return df
    return pd.DataFrame()


def compute_all_effects(adata: ad.AnnData) -> pd.DataFrame:
    """
    Calcula Cohen's d para CADA tipo celular entre CADA par de fenotipos.
    Genera heatmap material.
    """
    print("\n" + "=" * 70)
    print("COMPREHENSIVE CELL TYPE ANALYSIS — 15 tipos × fenotipos")
    print("=" * 70)
    
    ab_df = _get_abundance_df(adata)
    if ab_df.empty:
        print("  [ERROR] No abundance data")
        return pd.DataFrame()
    
    if 'Phenotype' not in adata.obs.columns:
        print("  [ERROR] No Phenotype column")
        return pd.DataFrame()
    
    phenotypes_present = [p for p in PHENOTYPES if (adata.obs['Phenotype'] == p).sum() >= 10]
    cell_types_present = [ct for ct in CELL_TYPES_15 if ct in ab_df.columns]
    
    print(f"  Cell types encontrados: {len(cell_types_present)}/{len(CELL_TYPES_15)}")
    print(f"  Fenotipos con n≥10: {phenotypes_present}")
    
    results = []
    
    # Comparaciones: cada fenotipo vs todos los demás (pooled)
    # Y específicamente Desert vs Excluded (hallazgo principal)
    comparisons = []
    
    # 1. Desert vs Excluded (principal)
    if 'Immune_Desert' in phenotypes_present and 'Immune_Excluded' in phenotypes_present:
        comparisons.append(('Immune_Desert', 'Immune_Excluded', 'Desert_vs_Excluded'))
    
    # 2. Desert vs Inflamed
    if 'Immune_Desert' in phenotypes_present and 'Inflamed' in phenotypes_present:
        comparisons.append(('Immune_Desert', 'Inflamed', 'Desert_vs_Inflamed'))
    
    # 3. Excluded vs Inflamed
    if 'Immune_Excluded' in phenotypes_present and 'Inflamed' in phenotypes_present:
        comparisons.append(('Immune_Excluded', 'Inflamed', 'Excluded_vs_Inflamed'))
    
    all_pvals = []
    
    for ct in cell_types_present:
        vals = ab_df[ct].values.astype(float)
        
        for pheno1, pheno2, comp_name in comparisons:
            mask1 = adata.obs['Phenotype'].values == pheno1
            mask2 = adata.obs['Phenotype'].values == pheno2
            
            g1 = vals[mask1]
            g2 = vals[mask2]
            g1 = g1[np.isfinite(g1)]
            g2 = g2[np.isfinite(g2)]
            
            if len(g1) < 10 or len(g2) < 10:
                continue
            
            d = cohens_d_pooled(g1, g2)
            _, pval = safe_mannwhitney(g1, g2)
            
            row = {
                'cell_type': ct,
                'comparison': comp_name,
                'group1': pheno1,
                'group2': pheno2,
                'n1': len(g1),
                'n2': len(g2),
                'median1': float(np.median(g1)),
                'median2': float(np.median(g2)),
                'cohens_d': float(d),
                'p_value': float(pval) if np.isfinite(pval) else 1.0,
            }
            results.append(row)
            all_pvals.append(pval if np.isfinite(pval) else 1.0)
    
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        return df
    
    # FDR correction across ALL tests
    reject, qvals = apply_fdr(np.array(all_pvals))
    df['q_value'] = qvals
    df['fdr_significant'] = reject
    
    # Flag unexpected
    df['effect_size'] = df['cohens_d'].apply(
        lambda d: 'Large' if abs(d) > 0.8 else ('Medium' if abs(d) > 0.5 else 'Small'))
    
    print(f"\n  Total tests: {len(df)}")
    print(f"  Significant (FDR): {reject.sum()}")
    
    # Print Desert vs Excluded results
    dve = df[df['comparison'] == 'Desert_vs_Excluded'].sort_values('cohens_d')
    if len(dve) > 0:
        print(f"\n  Desert vs Excluded (sorted by d):")
        for _, r in dve.iterrows():
            sig = "***" if r['fdr_significant'] else ""
            print(f"    {r['cell_type']:<20} d={r['cohens_d']:>+7.3f} q={r['q_value']:.2e} {sig}")
    
    return df


def plot_heatmap(df: pd.DataFrame, save_dir: Path):
    """Genera heatmap supplementary figure."""
    if df.empty:
        return
    
    print("\n  Generando heatmap...")
    
    # Pivot para heatmap: cell_type × comparison
    for comp in df['comparison'].unique():
        subset = df[df['comparison'] == comp]
        pivot = subset.pivot(index='cell_type', columns='comparison', values='cohens_d')
        
        # Si solo una comparación, usar Cohen's d directamente
        if pivot.shape[1] == 1:
            pivot = pivot.rename(columns={comp: "Cohen's d"})
    
    # Mejor: hacer heatmap multi-comparison
    pivot_full = df.pivot_table(index='cell_type', columns='comparison', values='cohens_d')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot_full, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                linewidths=0.5, ax=ax, vmin=-1.5, vmax=1.5,
                cbar_kws={'label': "Cohen's d (pooled, ddof=1)"})
    ax.set_title('Cell Type Abundance Differences Between Phenotypes\n(All 15 Types)', fontsize=13)
    ax.set_ylabel('')
    ax.set_xlabel('')
    plt.tight_layout()
    
    fig.savefig(save_dir / 'Fig_S_all_celltypes_heatmap.png', dpi=300, bbox_inches='tight')
    fig.savefig(save_dir / 'Fig_S_all_celltypes_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: Fig_S_all_celltypes_heatmap.pdf")


def run_comprehensive_celltype_analysis(adata: ad.AnnData, save_dir: Path = None) -> Dict:
    """Pipeline completo."""
    if save_dir is None:
        save_dir = OUTPUT_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    df = compute_all_effects(adata)
    
    if not df.empty:
        df.to_csv(save_dir / 'all_celltypes_by_phenotype.csv', index=False)
        plot_heatmap(df, save_dir)
    
    summary = {
        'n_cell_types': int(df['cell_type'].nunique()) if not df.empty else 0,
        'n_comparisons': len(df),
        'n_fdr_significant': int(df['fdr_significant'].sum()) if not df.empty else 0,
    }
    
    with open(save_dir / 'comprehensive_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


if __name__ == '__main__':
    adata = sc.read_h5ad(PATHS.PROCESSED_DIR / 'adata_with_phenotypes.h5ad')
    run_comprehensive_celltype_analysis(adata)
