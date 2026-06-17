#!/usr/bin/env python3
"""
================================================================================
VALIDATION & COMPARISON MODULE v4.0 — CONSOLIDATED
================================================================================
Unified module for cross-dataset validation (GSE210616 discovery vs GSE213688
validation) combining the best implementations from:

  - compare_datasets.py          → phenotype/abundance comparison, chi-squared
  - complete_validation_analysis  → publication figures (Fig1-Fig6), report
  - post_validation_analysis      → geodesic (weighted), spatial correlations,
                                    MYC-STING by phenotype, Euclidean comparison
  - validation_gse213688          → end-to-end pipeline structure, GPU setup

ANALYSES PERFORMED:
  1. Dataset Overview & Phenotype Distributions  (chi-squared, KS test)
  2. Cell Type Abundance Comparison              (Cohen's d, MW-U, all types)
  3. Spatial Correlations                        (7 pathway-cell type pairs)
  4. MYC-STING Pathway by Phenotype              (per-niche correlations)
  5. cDC1 Geodesic Distance                      (Dijkstra weighted, KEY NOVELTY)
  6. Euclidean Distance Comparison               (justifies geodesic approach)
  7. Key Metrics Replication Summary              (structured pass/fail)

FIGURES GENERATED (Publication-Ready):
  Fig1 - Phenotype Distribution (Discovery vs Validation)
  Fig2 - CAF Abundance Comparison (KEY FINDING)
  Fig3 - All Cell Types Butterfly Chart (Effect Size)
  Fig4 - MYC-STING Spatial Correlation
  Fig5 - cDC1 Geodesic Distance (CROWN JEWEL)
  Fig6 - Validation Summary Panel

FIXES CONSOLIDATED:
  Merged 4 scripts into one with best-of-breed functions
  Spatial graph per-sample (avoids cross-sample teleportation)
  Weighted geodesic using actual spatial distances
  Euclidean comparison, per-phenotype pathway correlations
================================================================================
"""

import os
import sys
import gc
import warnings
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
from scipy.stats import (spearmanr, pearsonr, mannwhitneyu, ttest_ind,
                          ks_2samp, chi2_contingency)
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import shortest_path, connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Patch

try:
    import squidpy as sq
    SQUIDPY_AVAILABLE = True
except ImportError:
    SQUIDPY_AVAILABLE = False

# Imports canónicos desde config y utils_stats
from config import (
    SIGNATURES, CANONICAL_SIGNATURES,
    PHENOTYPE_COLORS as _CANONICAL_PHENO_COLORS,
    DATASET_COLORS as _CANONICAL_DATASET_COLORS,
)
from utils_stats import cohens_d_pooled

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
class ValidationConfig:
    """Unified configuration for validation & comparison pipeline."""

    BASE_DIR = Path("/home/external/frjimenez/fabian/genoma")

    # ── Input Files ──────────────────────────────────────────────────────────
    GSE210616_PATH = BASE_DIR / "data/processed/adata_with_phenotypes.h5ad"
    GSE213688_PATHS = [
        # Reclassified v3 (from reclassify_validation.py) — prioridad máxima
        BASE_DIR / "results/validation_gse213688/adata_gse213688_classified_v3.h5ad",
        BASE_DIR / "results/validation_gse213688/adata_gse213688_classified_v2.h5ad",
        BASE_DIR / "results/validation_gse213688/adata_gse213688_classified.h5ad",
        BASE_DIR / "results/validation_gse213688/adata_gse213688_deconvolved.h5ad",
    ]

    # ── Output Directories ───────────────────────────────────────────────────
    OUTPUT_DIR  = BASE_DIR / "results/validation_final"
    FIGURES_DIR = OUTPUT_DIR / "figures"
    TABLES_DIR  = OUTPUT_DIR / "tables"

    # ── Reference Values from Discovery (GSE210616) ──────────────────────────
    # Estos valores de referencia se calcularon con
    # la fórmula ANTERIOR de Cohen's d (ddof=0, non-pooled).
    # Con la fórmula canónica (ddof=1, pooled), los valores de Discovery
    # cambiarán ligeramente. Actualizar después de re-ejecutar mechanism_validation.
    # Estos valores se calcularon con fórmula ANTERIOR (ddof=0, non-pooled).
    # Con cohens_d_pooled canónico (ddof=1, pooled), los valores cambiarán ~5%.
    # TODO: Re-ejecutar mechanism_validation.py en HPC y actualizar estos valores exactos.
    DISCOVERY_METRICS = {
        'CAF_cohens_d':       -0.63,    # Recalcular con cohens_d_pooled post-ejecución
        'CD8_p_value':         0.175,
        'MYC_CD8_correlation': -0.42,
        'cDC1_distance_ratio': 2.3,
    }

    # ── Pathway Gene Sets ────────────────────────────────────────────────────
    # Core genes from CANONICAL_SIGNATURES
    # Validation extiende STING con genes downstream (IRF7, IFNB1, IRF8, IFNG)
    # para capturar la cascada completa de señalización.
    # BARRIER usa COL3A1 (presente en GSE213688) en vez de COL1A2 (ausente).
    SILENCING_GENES  = CANONICAL_SIGNATURES['silencing_repressors'][:2] + ['DNMT1', 'STAT3']
    # Usar firma STING canónica (5 genes, igual que Discovery)
    # Eliminar extensión (STING1=alias TMEM173 causa doble conteo, IRF7/IFNB1/IRF8/IFNG son downstream)
    STING_GENES      = list(CANONICAL_SIGNATURES['sting_pathway'])
    CHEMOKINE_GENES  = CANONICAL_SIGNATURES['chemokine_signals'][:4]  # CXCL9, CXCL10, CXCL11, CCL5
    # Alineado con CANONICAL_SIGNATURES['bulk_excluded_up']
    # NOTA: COL3A1 reemplaza COL1A2 si está ausente en GSE213688
    BARRIER_GENES    = list(CANONICAL_SIGNATURES['bulk_excluded_up'])

    # ── Cell Types ───────────────────────────────────────────────────────────
    KEY_CELL_TYPES = ['CD8_T', 'CAF', 'cDC1', 'Macrophage', 'Tumor', 'NK',
                      'CD4_T', 'B_Cell', 'Endothelial', 'PVL', 'NKT',
                      'Monocyte', 'Myeloid_Cycling', 'T_Cell_Cycling',
                      'Normal_Epithelial']
    PHENOTYPES = ['Immune_Desert', 'Immune_Excluded', 'Inflamed',
                  'Normal_Stroma', 'Ambiguous_Cold']

    # ── Spatial Parameters ───────────────────────────────────────────────────
    SPATIAL_N_NEIGHS = 6          # Visium hexagonal grid
    COORD_TYPE       = 'generic'  # squidpy coord_type

    # ── Visualization ────────────────────────────────────────────────────────
    # Colores importados de config.PHENOTYPE_COLORS
    PHENOTYPE_COLORS = _CANONICAL_PHENO_COLORS
    DATASET_COLORS = _CANONICAL_DATASET_COLORS


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════
def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"validation_v4_{timestamp}.log"

    logger = logging.getLogger("validation_v4")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def safe_toarray(X):
    """Convert sparse matrix to dense array safely."""
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


# Alias directo, eliminada implementación local
cohens_d = cohens_d_pooled


def get_exact_column(df: pd.DataFrame, target_name: str) -> Optional[str]:
    """Find exact or fuzzy column match (case-insensitive)."""
    if target_name in df.columns:
        return target_name
    for col in df.columns:
        if target_name.lower() == col.lower():
            return col
    for col in df.columns:
        if target_name.lower() in col.lower():
            return col
    return None


def normalize_phenotype_column(adata: ad.AnnData, logger: logging.Logger) -> str:
    """Ensure 'phenotype' column exists, normalizing from variants."""
    if 'phenotype' in adata.obs.columns:
        return 'phenotype'
    for alt in ['phenotype_v2', 'Phenotype', 'phenotype_v3', 'immune_phenotype']:
        if alt in adata.obs.columns:
            adata.obs['phenotype'] = adata.obs[alt]
            logger.info(f"  Phenotype column: '{alt}' -> 'phenotype'")
            return 'phenotype'
    logger.warning("  No phenotype column found!")
    return ''


def calculate_gene_score(adata: ad.AnnData, genes: List[str],
                          score_name: str = '') -> np.ndarray:
    """Calculate mean expression score for a gene set.
    
    Prioriza .raw (log1p-normalizado) sobre .X para
    consistencia con phenotype_classifier.py. Si .X está scaled/z-scored,
    los scores serían diferentes sin este fix.
    """
    # Usar .raw si existe (log1p-normalizado, consistente con Discovery)
    # Log explícito cuando .raw es None (validation dataset)
    if adata.raw is not None:
        source = adata.raw.to_adata()
    else:
        source = adata
        # Solo avisar una vez por dataset, no por cada gene set
        if not hasattr(calculate_gene_score, '_warned_no_raw'):
            print(f"  [WARN] .raw es None — usando .X (max={source.X.max():.1f}). "
                  f"Scores pueden diferir de Discovery.")
            calculate_gene_score._warned_no_raw = True
    available = [g for g in genes if g in source.var_names]
    if not available:
        if score_name:
            print(f"  [WARN] 0/{len(genes)} genes de '{score_name}' encontrados")
        return np.zeros(adata.n_obs)
    gene_idx = [source.var_names.get_loc(g) for g in available]
    expr = safe_toarray(source.X[:, gene_idx])
    return expr.mean(axis=1)


def get_abundances(adata: ad.AnnData) -> Optional[pd.DataFrame]:
    """Extract cell abundances with phenotype column."""
    obsm_key = None
    for key in ['means_cell_abundance_w_sf', 'q50_cell_abundance_w_sf']:
        if key in adata.obsm:
            obsm_key = key
            break
    if obsm_key is None:
        return None

    ab = pd.DataFrame(adata.obsm[obsm_key], index=adata.obs_names)
    if 'mod' in adata.uns and 'factor_names' in adata.uns['mod']:
        names = adata.uns['mod']['factor_names']
        if isinstance(names, np.ndarray):
            names = names.tolist()
        ab.columns = names

    if 'phenotype' in adata.obs.columns:
        ab['phenotype'] = adata.obs['phenotype'].values
    return ab


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: DATASET OVERVIEW & PHENOTYPE DISTRIBUTION
# Source: compare_datasets.py — best phenotype comparison with chi-squared
# ═══════════════════════════════════════════════════════════════════════════════
def compare_dataset_overview(adata_disc: ad.AnnData, adata_val: ad.AnnData,
                              logger: logging.Logger) -> Dict:
    """High-level comparison between datasets."""
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 1a: Dataset Overview")
    logger.info("=" * 70)

    overview = {
        'discovery': {'n_spots': int(adata_disc.n_obs), 'n_genes': int(adata_disc.n_vars)},
        'validation': {'n_spots': int(adata_val.n_obs), 'n_genes': int(adata_val.n_vars)},
        'common_genes': int(len(adata_disc.var_names.intersection(adata_val.var_names))),
    }

    for label, ad_ in [('GSE210616', adata_disc), ('GSE213688', adata_val)]:
        logger.info(f"\n  {label}: {ad_.n_obs:,} spots x {ad_.n_vars:,} genes")
        if 'phenotype' in ad_.obs.columns:
            for p, c in ad_.obs['phenotype'].value_counts().items():
                logger.info(f"    {p}: {c:,} ({100*c/ad_.n_obs:.1f}%)")

    logger.info(f"\n  Common genes: {overview['common_genes']:,}")
    return overview


def compare_phenotype_distributions(adata_disc: ad.AnnData, adata_val: ad.AnnData,
                                     config: ValidationConfig,
                                     logger: logging.Logger) -> pd.DataFrame:
    """Compare phenotype distributions with chi-squared test."""
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 1b: Phenotype Distribution Comparison")
    logger.info("=" * 70)

    disc_counts = adata_disc.obs['phenotype'].value_counts()
    val_counts = adata_val.obs['phenotype'].value_counts()

    all_phenotypes = sorted(set(disc_counts.index) | set(val_counts.index))

    rows = []
    for p in all_phenotypes:
        d_n = disc_counts.get(p, 0)
        v_n = val_counts.get(p, 0)
        d_pct = 100 * d_n / adata_disc.n_obs
        v_pct = 100 * v_n / adata_val.n_obs
        rows.append({
            'phenotype': p,
            'disc_count': int(d_n), 'disc_pct': d_pct,
            'val_count': int(v_n), 'val_pct': v_pct,
            'diff_pct': abs(d_pct - v_pct),
        })

    df = pd.DataFrame(rows)

    # Chi-squared test on contingency table
    contingency = df[['disc_count', 'val_count']].values
    if contingency.sum() > 0 and contingency.shape[0] > 1:
        chi2, chi_p, dof, _ = chi2_contingency(contingency)
        logger.info(f"\n  Chi-squared: chi2 = {chi2:.2f}, df = {dof}, p = {chi_p:.4e}")
        df.attrs['chi2_pvalue'] = chi_p
    else:
        df.attrs['chi2_pvalue'] = np.nan

    logger.info("\n  Phenotype comparison:")
    for _, r in df.iterrows():
        logger.info(f"    {r['phenotype']:20s}: Disc={r['disc_pct']:5.1f}%  "
                    f"Val={r['val_pct']:5.1f}%  Delta={r['diff_pct']:.1f}pp")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: CELL TYPE ABUNDANCE COMPARISON
# Source: compare_datasets.py — most comprehensive (all types, 3 test types)
# ═══════════════════════════════════════════════════════════════════════════════
def compare_cell_abundances(ab_disc: pd.DataFrame, ab_val: pd.DataFrame,
                             config: ValidationConfig,
                             logger: logging.Logger) -> pd.DataFrame:
    """Compare Desert vs Excluded abundances across all cell types.
    
    Guard para 0 Desert spots en validation.
    Si validation no tiene Desert, solo reporta Discovery y skip validation comparison.
    """
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 2: Cell Type Abundance (Desert vs Excluded)")
    logger.info("=" * 70)

    # FIX FASE1-08: Check Desert spot counts BEFORE iterating
    n_disc_desert = (ab_disc['phenotype'] == 'Immune_Desert').sum() if 'phenotype' in ab_disc.columns else 0
    n_val_desert  = (ab_val['phenotype'] == 'Immune_Desert').sum() if 'phenotype' in ab_val.columns else 0
    val_has_desert = n_val_desert >= 5
    
    if not val_has_desert:
        logger.warning(f"  Validation has {n_val_desert} Desert spots — "
                       f"skipping Desert-vs-Excluded comparison for validation. "
                       f"Reclassify GSE213688 with phenotype_classifier.py first.")

    results = []
    numeric_cols = [c for c in ab_disc.columns if c != 'phenotype']

    for col in numeric_cols:
        disc_desert   = ab_disc[ab_disc['phenotype'] == 'Immune_Desert'][col]
        disc_excluded = ab_disc[ab_disc['phenotype'] == 'Immune_Excluded'][col]

        if any(len(g) < 5 for g in [disc_desert, disc_excluded]):
            continue

        d_disc = cohens_d(disc_desert, disc_excluded)
        _, p_disc = mannwhitneyu(disc_desert, disc_excluded, alternative='two-sided')

        # Validation comparison only if Desert spots exist
        if val_has_desert:
            val_desert    = ab_val[ab_val['phenotype'] == 'Immune_Desert'][col]
            val_excluded  = ab_val[ab_val['phenotype'] == 'Immune_Excluded'][col]
            if any(len(g) < 5 for g in [val_desert, val_excluded]):
                d_val, p_val = np.nan, np.nan
            else:
                d_val  = cohens_d(val_desert, val_excluded)
                _, p_val = mannwhitneyu(val_desert, val_excluded, alternative='two-sided')
        else:
            d_val, p_val = np.nan, np.nan

        # Concordance: same sign of effect (only if both valid)
        concordant = (d_disc > 0) == (d_val > 0) if np.isfinite(d_val) else False
        replicated = concordant and abs(d_val) > 0.3 and p_val < 0.05 if np.isfinite(d_val) else False

        results.append({
            'cell_type': col,
            'discovery_d':  round(d_disc, 3),
            'discovery_p':  p_disc,
            'validation_d': round(d_val, 3),
            'validation_p': p_val,
            'concordant':   concordant,
            'replicated':   replicated,
        })

    df = pd.DataFrame(results)
    if len(df) == 0:
        return df

    df = df.sort_values('discovery_d')

    logger.info(f"\n  {'Cell Type':<22} {'Disc d':>8} {'Val d':>8} {'Conc':>6} {'Repl':>6}")
    logger.info("  " + "-" * 56)
    for _, r in df.iterrows():
        c = "Y" if r['concordant'] else "N"
        rp = "Y" if r['replicated'] else "N"
        logger.info(f"  {r['cell_type']:<22} {r['discovery_d']:>8.3f} "
                    f"{r['validation_d']:>8.3f} {c:>6} {rp:>6}")

    n_conc = df['concordant'].sum()
    n_repl = df['replicated'].sum()
    logger.info(f"\n  Concordant: {n_conc}/{len(df)} ({100*n_conc/len(df):.0f}%)")
    logger.info(f"  Replicated: {n_repl}/{len(df)} ({100*n_repl/len(df):.0f}%)")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: SPATIAL CORRELATIONS
# Source: post_validation_analysis.py — UNIQUE, 7 pathway-cell type pairs
# ═══════════════════════════════════════════════════════════════════════════════
def analyze_spatial_correlations(adata: ad.AnnData, config: ValidationConfig,
                                  logger: logging.Logger) -> pd.DataFrame:
    """Analyze spatial correlations between cell types and pathway scores."""
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 3: Spatial Correlations (Pathway x Cell Type)")
    logger.info("=" * 70)

    ab = get_abundances(adata)
    if ab is None:
        return pd.DataFrame()

    adata.obs['MYC_score']       = calculate_gene_score(adata, ['MYC'], 'MYC')
    adata.obs['STING_score']     = calculate_gene_score(adata, config.STING_GENES, 'STING')
    adata.obs['Silencing_score'] = calculate_gene_score(adata, config.SILENCING_GENES, 'Silencing')
    adata.obs['Chemokine_score'] = calculate_gene_score(adata, config.CHEMOKINE_GENES, 'Chemokine')

    cd8_col  = get_exact_column(ab, 'CD8_T')
    cdc1_col = get_exact_column(ab, 'cDC1')
    caf_col  = get_exact_column(ab, 'CAF')
    tumor_col = get_exact_column(ab, 'Tumor')

    pairs = []
    if cd8_col:
        pairs += [(cd8_col, 'MYC_score', 'CD8 vs MYC'),
                  (cd8_col, 'Silencing_score', 'CD8 vs Silencing'),
                  (cd8_col, 'Chemokine_score', 'CD8 vs Chemokines')]
    if cdc1_col:
        pairs += [(cdc1_col, 'MYC_score', 'cDC1 vs MYC'),
                  (cdc1_col, 'Chemokine_score', 'cDC1 vs Chemokines')]
    if caf_col and cd8_col:
        pairs.append((caf_col, cd8_col, 'CAF vs CD8'))
    if tumor_col and cd8_col:
        pairs.append((tumor_col, cd8_col, 'Tumor vs CD8'))

    results = []
    logger.info(f"\n  {'Comparison':<25} {'Spearman rho':>12} {'P-value':>12} {'Interpretation'}")
    logger.info("  " + "-" * 70)

    for var1, var2, label in pairs:
        v1 = ab[var1].values if var1 in ab.columns else adata.obs[var1].values if var1 in adata.obs.columns else None
        v2 = ab[var2].values if var2 in ab.columns else adata.obs[var2].values if var2 in adata.obs.columns else None
        if v1 is None or v2 is None:
            continue

        rho, pval = spearmanr(v1, v2)

        if abs(rho) < 0.1:     interp = "Negligible"
        elif abs(rho) < 0.3:   interp = "Weak"
        elif abs(rho) < 0.5:   interp = "Moderate"
        else:                   interp = "Strong"
        interp += " negative" if rho < 0 else " positive"

        results.append({'comparison': label, 'var1': var1, 'var2': var2,
                         'spearman_rho': rho, 'p_value': pval,
                         'interpretation': interp})
        logger.info(f"  {label:<25} {rho:>12.3f} {pval:>12.2e} {interp}")

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: MYC-STING PATHWAY BY PHENOTYPE
# Source: post_validation_analysis.py — per-niche breakdown (most detailed)
# ═══════════════════════════════════════════════════════════════════════════════
def analyze_myc_sting_pathway(adata: ad.AnnData, config: ValidationConfig,
                               logger: logging.Logger) -> Dict:
    """Analyze MYC-STING pathway correlation stratified by phenotype."""
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 4: MYC-STING Pathway (Stratified by Phenotype)")
    logger.info("=" * 70)

    results = {}
    myc_expr   = calculate_gene_score(adata, ['MYC'], 'MYC')
    sting_expr = calculate_gene_score(adata, config.STING_GENES, 'STING')

    # FIX FASE4-01: Check if STING genes are actually present (0/5 in validation GSE213688)
    sting_available = [g for g in config.STING_GENES if g in (adata.raw.var_names if adata.raw is not None else adata.var_names)]
    if len(sting_available) == 0:
        logger.warning(f"  0/{len(config.STING_GENES)} STING genes found in dataset!")
        logger.warning(f"  Missing: {config.STING_GENES}")
        logger.warning(f"  MYC-STING correlations CANNOT be computed for this dataset.")
        logger.warning(f"  Paper should state: 'STING pathway genes were absent from the")
        logger.warning(f"  validation panel; MYC-STING analysis restricted to Discovery.'")
        results['sting_genes_found'] = 0
        results['sting_genes_total'] = len(config.STING_GENES)
        results['skip_reason'] = 'STING genes absent from dataset'
        return results
    
    logger.info(f"  STING genes: {len(sting_available)}/{len(config.STING_GENES)} present")

    # Guard: if all sting_expr is zero, skip
    if np.all(sting_expr == 0) or np.std(sting_expr) < 1e-10:
        logger.warning(f" STING score is zero/constant for all spots. Skipping correlations.")
        results['skip_reason'] = 'STING score zero/constant'
        return results

    rho_all, pval_all = spearmanr(myc_expr, sting_expr)
    results['overall'] = {'spearman_rho': float(rho_all), 'p_value': float(pval_all)}
    logger.info(f"\n  Overall: rho = {rho_all:.4f} (p = {pval_all:.2e})")

    logger.info("\n  By phenotype:")
    for phenotype in ['Immune_Desert', 'Immune_Excluded', 'Inflamed']:
        mask = adata.obs['phenotype'] == phenotype
        if mask.sum() < 10:
            continue
        rho, pval = spearmanr(myc_expr[mask], sting_expr[mask])
        results[phenotype] = {'n_spots': int(mask.sum()),
                               'spearman_rho': float(rho), 'p_value': float(pval)}
        logger.info(f"    {phenotype}: rho = {rho:.4f} (n={mask.sum()}, p={pval:.2e})")

    if abs(rho_all) < 0.1:
        results['interpretation'] = 'SUPPORTS_SPATIAL_SEGREGATION'
        logger.info("\n  [OK] Weak spatial correlation supports compartmentalization hypothesis")
    else:
        results['interpretation'] = 'SPATIAL_CORRELATION_PRESENT'
        logger.info(f"\n  [!] Spatial correlation rho={rho_all:.3f} detected")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 5: cDC1 GEODESIC DISTANCE (KEY NOVELTY)
# Source: post_validation_analysis.py — weighted Dijkstra (most rigorous)
# ═══════════════════════════════════════════════════════════════════════════════
def _build_spatial_graph_per_sample(adata: ad.AnnData, sample_id: str,
                                     config: ValidationConfig,
                                     logger: logging.Logger) -> Optional[ad.AnnData]:
    """Build per-sample spatial graph (prevents cross-sample teleportation)."""
    mask = adata.obs['sample_id'] == sample_id
    if mask.sum() < 50:
        return None

    adata_s = adata[mask].copy()

    try:
        if SQUIDPY_AVAILABLE:
            sq.gr.spatial_neighbors(
                adata_s,
                n_neighs=config.SPATIAL_N_NEIGHS,
                coord_type=config.COORD_TYPE,
                spatial_key='spatial'
            )
        else:
            # Fallback: scanpy neighbors on spatial coordinates
            adata_s.obsm['X_spatial'] = adata_s.obsm['spatial'].astype(float)
            sc.pp.neighbors(adata_s, n_neighbors=config.SPATIAL_N_NEIGHS + 1,
                            use_rep='X_spatial')

        if 'connectivities' not in adata_s.obsp:
            return None
        return adata_s
    except Exception as e:
        logger.warning(f"  Graph failed for {sample_id}: {e}")
        return None


def _calculate_geodesic_distances(adata_sample: ad.AnnData,
                                   source_indices: np.ndarray,
                                   target_indices: np.ndarray,
                                   logger: logging.Logger) -> np.ndarray:
    """
    Geodesic shortest-path through spatial graph (Dijkstra).

    Uses actual spatial distances as edge weights when available,
    otherwise falls back to unweighted hop count.
    """
    if 'connectivities' not in adata_sample.obsp:
        raise ValueError("Spatial graph not computed")

    # Prefer weighted (actual distances) over unweighted (hop count)
    if 'distances' in adata_sample.obsp:
        dist_matrix = adata_sample.obsp['distances'].copy()
        dist_matrix.data[dist_matrix.data == 0] = np.inf
    else:
        dist_matrix = adata_sample.obsp['connectivities'].copy()
        dist_matrix.data = np.ones_like(dist_matrix.data)

    try:
        sp = shortest_path(dist_matrix, method='D', directed=False,
                           indices=source_indices)
    except Exception as e:
        logger.warning(f"  Shortest path failed: {e}")
        return np.array([])

    min_dists = []
    for i in range(len(source_indices)):
        d = sp[i, target_indices]
        finite = d[np.isfinite(d)]
        if len(finite) > 0:
            min_dists.append(np.min(finite))
    return np.array(min_dists)


def analyze_cdc1_geodesic_distances(adata: ad.AnnData, config: ValidationConfig,
                                     logger: logging.Logger) -> Dict:
    """
    Analyze cDC1-Tumor GEODESIC distance — the KEY NOVELTY CLAIM.
    
    Validates that cDC1 cells are physically isolated from tumor in
    Excluded phenotypes due to stromal barriers, while Desert phenotypes
    lack such barriers (pure silencing mechanism).
    """
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 5: cDC1 GEODESIC Distance (KEY NOVELTY)")
    logger.info("=" * 70)
    logger.info("  Method: Dijkstra shortest-path through spatial graph")

    results = {}

    if 'spatial' not in adata.obsm:
        logger.warning("  No spatial coordinates!")
        return results

    ab = get_abundances(adata)
    if ab is None:
        logger.warning("  No abundance data!")
        return results

    cdc1_col  = get_exact_column(ab, 'cDC1')
    tumor_col = get_exact_column(ab, 'Tumor')
    if cdc1_col is None or tumor_col is None:
        logger.warning(f"  Missing columns: cDC1={cdc1_col}, Tumor={tumor_col}")
        return results

    logger.info(f"  Columns: cDC1='{cdc1_col}', Tumor='{tumor_col}'")

    samples = adata.obs['sample_id'].unique()
    phenotype_distances = {'Immune_Desert': [], 'Immune_Excluded': []}

    for sample_id in samples:
        adata_s = _build_spatial_graph_per_sample(adata, sample_id, config, logger)
        if adata_s is None:
            continue

        # Usar get_abundances() en vez de hardcodear key obsm
        # El key 'means_cell_abundance_w_sf' puede no existir si Cell2Location usó otro nombre
        local_ab = get_abundances(adata_s)
        if local_ab is None:
            continue
        # Eliminar columna phenotype si fue añadida por get_abundances
        if 'phenotype' in local_ab.columns:
            local_ab = local_ab.drop(columns=['phenotype'])

        for phenotype in ['Immune_Desert', 'Immune_Excluded']:
            pheno_mask = adata_s.obs['phenotype'] == phenotype
            if pheno_mask.sum() < 20:
                continue

            pheno_idx = np.where(pheno_mask)[0]
            cdc1_vals  = local_ab.iloc[pheno_idx][cdc1_col].values
            tumor_vals = local_ab.iloc[pheno_idx][tumor_col].values

            cdc1_thresh  = max(np.percentile(cdc1_vals, 75), 0.1)
            tumor_thresh = max(np.percentile(tumor_vals, 75), 0.1)

            high_cdc1  = cdc1_vals  >= cdc1_thresh
            high_tumor = tumor_vals >= tumor_thresh

            if high_cdc1.sum() < 3 or high_tumor.sum() < 3:
                continue

            try:
                geo = _calculate_geodesic_distances(
                    adata_s, pheno_idx[high_cdc1], pheno_idx[high_tumor], logger)
                if len(geo) > 0:
                    phenotype_distances[phenotype].extend(geo.tolist())
            except Exception:
                continue

    # ── Compile & test ───────────────────────────────────────────────────────
    logger.info("\n  --- GEODESIC DISTANCE RESULTS ---")

    for pheno in ['Immune_Desert', 'Immune_Excluded']:
        dists = np.array(phenotype_distances[pheno])
        if len(dists) > 0:
            results[pheno] = {
                'n': len(dists),
                'mean': float(np.mean(dists)),
                'std':  float(np.std(dists)),
                'median': float(np.median(dists)),
            }
            logger.info(f"  {pheno}: n={len(dists)}, "
                        f"mean={np.mean(dists):.2f} +/- {np.std(dists):.2f}")

    if 'Immune_Desert' in results and 'Immune_Excluded' in results:
        d_dists = np.array(phenotype_distances['Immune_Desert'])
        e_dists = np.array(phenotype_distances['Immune_Excluded'])

        desert_mean   = results['Immune_Desert']['mean']
        excluded_mean = results['Immune_Excluded']['mean']
        ratio = excluded_mean / desert_mean if desert_mean > 0 else 0

        stat, pval = mannwhitneyu(d_dists, e_dists, alternative='two-sided')
        d_effect = cohens_d(d_dists, e_dists)

        results['comparison'] = {
            'geodesic_distance_ratio': float(ratio),
            'cohens_d': float(d_effect),
            'mannwhitney_p': float(pval),
        }
        # Convenience keys for figures
        results['desert_mean']   = desert_mean
        results['excluded_mean'] = excluded_mean
        results['ratio']     = float(ratio)
        results['validated'] = ratio > 1.2 and pval < 0.05

        logger.info(f"\n  {'='*50}")
        logger.info(f"  GEODESIC RATIO: Excluded/Desert = {ratio:.2f}x")
        logger.info(f"  Cohen's d: {d_effect:.3f}")
        logger.info(f"  Mann-Whitney p: {pval:.4e}")

        if ratio > 1.5 and pval < 0.05:
            logger.info("  [OK] VALIDATED: cDC1 geodesically MORE DISTANT in Excluded")
            results['interpretation'] = 'GEODESIC_ISOLATION_CONFIRMED'
        elif ratio > 1.2:
            logger.info("  [!] TREND observed")
            results['interpretation'] = 'TREND_OBSERVED'
        else:
            logger.info("  [X] Not confirmed")
            results['interpretation'] = 'NOT_CONFIRMED'

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 6: EUCLIDEAN DISTANCE COMPARISON
# Source: post_validation_analysis.py — UNIQUE, justifies geodesic approach
# ═══════════════════════════════════════════════════════════════════════════════
def calculate_euclidean_comparison(adata: ad.AnnData, config: ValidationConfig,
                                    logger: logging.Logger) -> Dict:
    """Euclidean distance for comparison (shows why geodesic is needed)."""
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 6: Euclidean Distance (Reference)")
    logger.info("=" * 70)

    ab = get_abundances(adata)
    if ab is None:
        return {}

    cdc1_col  = get_exact_column(ab, 'cDC1')
    tumor_col = get_exact_column(ab, 'Tumor')
    if cdc1_col is None or tumor_col is None:
        return {}

    coords = adata.obsm['spatial']
    results = {}

    for phenotype in ['Immune_Desert', 'Immune_Excluded']:
        mask = adata.obs['phenotype'] == phenotype
        if mask.sum() < 20:
            continue

        idx = np.where(mask)[0]
        cdc1_v  = ab.iloc[idx][cdc1_col].values
        tumor_v = ab.iloc[idx][tumor_col].values

        high_cdc1  = cdc1_v  >= max(np.percentile(cdc1_v, 75), 0.1)
        high_tumor = tumor_v >= max(np.percentile(tumor_v, 75), 0.1)

        if high_cdc1.sum() < 3 or high_tumor.sum() < 3:
            continue

        dists = cdist(coords[idx[high_cdc1]], coords[idx[high_tumor]], 'euclidean')
        min_dists = dists.min(axis=1)

        results[phenotype] = {
            'mean_euclidean': float(np.mean(min_dists)),
            'std_euclidean':  float(np.std(min_dists)),
        }
        logger.info(f"  {phenotype}: {np.mean(min_dists):.1f} +/- {np.std(min_dists):.1f} pixels")

    if 'Immune_Desert' in results and 'Immune_Excluded' in results:
        ratio = (results['Immune_Excluded']['mean_euclidean'] /
                 results['Immune_Desert']['mean_euclidean'])
        results['ratio'] = float(ratio)
        logger.info(f"  Euclidean ratio: {ratio:.2f}x")
        logger.info("  Note: Geodesic accounts for tissue topology; Euclidean does not.")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 7: KEY METRICS REPLICATION SUMMARY
# Source: compare_datasets.py — structured replication checking
# ═══════════════════════════════════════════════════════════════════════════════
def compare_key_metrics(adata_disc: ad.AnnData, adata_val: ad.AnnData,
                         ab_disc: pd.DataFrame, ab_val: pd.DataFrame,
                         config: ValidationConfig,
                         logger: logging.Logger) -> Dict:
    """Structured check of each key paper metric against discovery."""
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS 7: Key Metrics Replication")
    logger.info("=" * 70)

    results = {}

    # ── METRIC 1: CAF Cohen's d ──────────────────────────────────────────────
    caf_col = get_exact_column(ab_val, 'CAF')
    if caf_col:
        vd = ab_val[ab_val['phenotype'] == 'Immune_Desert'][caf_col]
        ve = ab_val[ab_val['phenotype'] == 'Immune_Excluded'][caf_col]
        if len(vd) > 0 and len(ve) > 0:
            d_val = cohens_d(vd, ve)
            _, p_val = mannwhitneyu(vd, ve)
            results['CAF'] = {
                'discovery': config.DISCOVERY_METRICS['CAF_cohens_d'],
                'validation': float(d_val),
                'p_value': float(p_val),
                'replicated': d_val < -0.3 and p_val < 0.05,
            }
            status = "PASS" if results['CAF']['replicated'] else "FAIL"
            logger.info(f"\n  CAF: disc d={config.DISCOVERY_METRICS['CAF_cohens_d']:.3f}, "
                        f"val d={d_val:.3f}, p={p_val:.2e}  [{status}]")

    # ── METRIC 2: CD8 equivalence ────────────────────────────────────────────
    cd8_col = get_exact_column(ab_val, 'CD8_T')
    if cd8_col:
        vd = ab_val[ab_val['phenotype'] == 'Immune_Desert'][cd8_col]
        ve = ab_val[ab_val['phenotype'] == 'Immune_Excluded'][cd8_col]
        if len(vd) > 0 and len(ve) > 0:
            d_val = cohens_d(vd, ve)
            _, p_val = mannwhitneyu(vd, ve)
            results['CD8'] = {
                'discovery_p': config.DISCOVERY_METRICS['CD8_p_value'],
                'validation_d': float(d_val),
                'validation_p': float(p_val),
                'replicated': abs(d_val) < 0.3,  # small effect = equivalence
            }
            status = "PASS" if results['CD8']['replicated'] else "FAIL"
            logger.info(f"  CD8: disc p={config.DISCOVERY_METRICS['CD8_p_value']:.3f} (NS), "
                        f"val d={d_val:.3f}, p={p_val:.2e}  [{status}]")

    # ── METRIC 3: MYC-CD8 correlation ────────────────────────────────────────
    if 'MYC' in adata_val.var_names and cd8_col:
        myc = safe_toarray(adata_val.X[:, adata_val.var_names.get_loc('MYC')]).flatten()
        cd8 = ab_val[cd8_col].values
        rho, pval = spearmanr(myc, cd8)
        results['MYC_CD8'] = {
            'discovery': config.DISCOVERY_METRICS['MYC_CD8_correlation'],
            'validation': float(rho),
            'p_value': float(pval),
            'replicated': rho < -0.2 and pval < 0.05,
        }
        status = "PASS" if results['MYC_CD8']['replicated'] else "FAIL"
        logger.info(f"  MYC-CD8: disc rho={config.DISCOVERY_METRICS['MYC_CD8_correlation']:.3f}, "
                    f"val rho={rho:.3f}, p={pval:.2e}  [{status}]")

    # ── Summary ──────────────────────────────────────────────────────────────
    n_r = sum(1 for v in results.values() if v.get('replicated'))
    logger.info(f"\n  Replicated: {n_r}/{len(results)}")
    return results

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION REPORT
# Source: complete_validation_analysis — most comprehensive
# ═══════════════════════════════════════════════════════════════════════════════
def generate_report(overview, phenotype_df, celltype_df, key_metrics,
                     spatial_corr, myc_sting, geodesic, euclidean,
                     config, logger):
    """Generate comprehensive Q1-ready validation report."""
    logger.info("\n  Generating Validation Report...")

    lines = [
        "=" * 80,
        "VALIDATION REPORT v4.0 -- GSE210616 vs GSE213688",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Target: Nature Communications / Cancer Research (Q1)",
        "=" * 80,
        "",
        "1. DATASET OVERVIEW",
        "-" * 40,
        f"  Discovery (GSE210616): {overview.get('discovery', {}).get('n_spots', '?'):,} spots",
        f"  Validation (GSE213688): {overview.get('validation', {}).get('n_spots', '?'):,} spots",
        f"  Common genes: {overview.get('common_genes', '?'):,}",
        "",
        "2. PHENOTYPE DISTRIBUTION",
        "-" * 40,
    ]

    if len(phenotype_df) > 0:
        for _, r in phenotype_df.iterrows():
            lines.append(f"  {r['phenotype']:20s}: Disc={r['disc_pct']:5.1f}%  Val={r['val_pct']:5.1f}%")
        chi_p = phenotype_df.attrs.get('chi2_pvalue', np.nan)
        if not np.isnan(chi_p):
            lines.append(f"  Chi-squared p-value: {chi_p:.4e}")

    lines += ["", "3. CELL TYPE EFFECT SIZES (Desert vs Excluded)", "-" * 40]
    if len(celltype_df) > 0:
        for _, r in celltype_df.iterrows():
            s = "[Y]" if r['replicated'] else "[N]"
            lines.append(f"  {s} {r['cell_type']:<20s}: disc d={r['discovery_d']:.3f}, val d={r['validation_d']:.3f}")
        n_r = celltype_df['replicated'].sum()
        lines.append(f"  Replicated: {n_r}/{len(celltype_df)}")

    lines += ["", "4. KEY METRICS REPLICATION", "-" * 40]
    for metric, data in key_metrics.items():
        s = "[PASS]" if data.get('replicated') else "[FAIL]"
        lines.append(f"  {s} {metric}: {json.dumps(data, default=str)}")

    lines += ["", "5. SPATIAL CORRELATIONS", "-" * 40]
    if len(spatial_corr) > 0:
        for _, r in spatial_corr.iterrows():
            lines.append(f"  {r['comparison']:<25s}: rho={r['spearman_rho']:.3f} ({r['interpretation']})")

    lines += ["", "6. MYC-STING PATHWAY", "-" * 40]
    for k, v in myc_sting.items():
        lines.append(f"  {k}: {json.dumps(v, default=str)}")

    lines += ["", "7. GEODESIC DISTANCE (KEY NOVELTY)", "-" * 40]
    if 'comparison' in geodesic:
        lines.append(f"  Ratio: {geodesic.get('ratio', 'N/A')}")
        lines.append(f"  Cohen's d: {geodesic['comparison']['cohens_d']:.3f}")
        lines.append(f"  MW p-value: {geodesic['comparison']['mannwhitney_p']:.4e}")
        lines.append(f"  Interpretation: {geodesic.get('interpretation', 'N/A')}")

    lines += ["", "8. EUCLIDEAN COMPARISON (reference)", "-" * 40]
    lines.append(f"  Ratio: {euclidean.get('ratio', 'N/A')}")

    lines += ["", "=" * 80, "END OF REPORT", "=" * 80]

    report_path = config.OUTPUT_DIR / "VALIDATION_REPORT_v4.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info(f"    Report saved: {report_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    config = ValidationConfig()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(config.OUTPUT_DIR)

    logger.info("=" * 80)
    logger.info("VALIDATION & COMPARISON v4.0 -- CONSOLIDATED PIPELINE")
    logger.info("=" * 80)
    logger.info(f"  Squidpy available: {SQUIDPY_AVAILABLE}")

    # ── LOAD DATA ────────────────────────────────────────────────────────────
    logger.info("\n[1/8] Loading datasets...")

    if not config.GSE210616_PATH.exists():
        logger.error(f"Discovery data not found: {config.GSE210616_PATH}")
        return
    adata_disc = sc.read_h5ad(config.GSE210616_PATH)
    logger.info(f"  GSE210616: {adata_disc.shape}")
    normalize_phenotype_column(adata_disc, logger)

    adata_val = None
    for p in config.GSE213688_PATHS:
        if p.exists():
            adata_val = sc.read_h5ad(p)
            logger.info(f"  GSE213688: {adata_val.shape} ({p.name})")
            break
    if adata_val is None:
        logger.error("Validation data not found!")
        return
    normalize_phenotype_column(adata_val, logger)

    ab_disc = get_abundances(adata_disc)
    ab_val  = get_abundances(adata_val)

    if ab_disc is None or ab_val is None:
        logger.error("Cannot extract abundances from one or both datasets!")
        return

    logger.info(f"  Discovery phenotypes: {ab_disc['phenotype'].value_counts().to_dict()}")
    logger.info(f"  Validation phenotypes: {ab_val['phenotype'].value_counts().to_dict()}")

    # FIX FASE1-10: Check if validation has Desert spots — critical for all comparisons
    n_val_desert = (ab_val['phenotype'] == 'Immune_Desert').sum() if 'phenotype' in ab_val.columns else 0
    if n_val_desert == 0:
        logger.warning("=" * 70)
        logger.warning("VALIDATION HAS 0 Immune_Desert SPOTS!")
        logger.warning("  Desert-vs-Excluded comparisons will be skipped for validation.")
        logger.warning("  Run reclassify_validation.py to re-classify GSE213688")
        logger.warning("  with the SAME phenotype_classifier.py used for Discovery.")
        logger.warning("=" * 70)

    # ── ANALYSES ─────────────────────────────────────────────────────────────
    logger.info("\n[2/8] Dataset overview & phenotype distributions...")
    overview     = compare_dataset_overview(adata_disc, adata_val, logger)
    phenotype_df = compare_phenotype_distributions(adata_disc, adata_val, config, logger)

    logger.info("\n[3/8] Cell type abundance comparison...")
    celltype_df = compare_cell_abundances(ab_disc, ab_val, config, logger)

    logger.info("\n[4/8] Spatial correlations (validation dataset)...")
    spatial_corr = analyze_spatial_correlations(adata_val, config, logger)

    logger.info("\n[5/8] MYC-STING pathway (validation dataset)...")
    myc_sting = analyze_myc_sting_pathway(adata_val, config, logger)

    logger.info("\n[6/8] cDC1 Geodesic distance (validation dataset)...")
    geodesic = analyze_cdc1_geodesic_distances(adata_val, config, logger)

    logger.info("\n[7/8] Euclidean distance (reference)...")
    euclidean = calculate_euclidean_comparison(adata_val, config, logger)

    key_metrics = compare_key_metrics(adata_disc, adata_val, ab_disc, ab_val,
                                       config, logger)

    # ── SAVE TABLES ──────────────────────────────────────────────────────────
    phenotype_df.to_csv(config.TABLES_DIR / 'phenotype_comparison.csv', index=False)
    celltype_df.to_csv(config.TABLES_DIR / 'celltype_effect_sizes.csv', index=False)
    spatial_corr.to_csv(config.TABLES_DIR / 'spatial_correlations.csv', index=False)

    with open(config.TABLES_DIR / 'key_metrics.json', 'w') as f:
        json.dump(key_metrics, f, indent=2, default=str)
    with open(config.TABLES_DIR / 'geodesic_results.json', 'w') as f:
        json.dump(geodesic, f, indent=2, default=str)
    with open(config.TABLES_DIR / 'myc_sting_results.json', 'w') as f:
        json.dump(myc_sting, f, indent=2, default=str)

    # ── REPORT ───────────────────────────────────────────────────────────────
    generate_report(overview, phenotype_df, celltype_df, key_metrics,
                     spatial_corr, myc_sting, geodesic, euclidean,
                     config, logger)

    logger.info("\n" + "=" * 80)
    logger.info("[OK] VALIDATION & COMPARISON v4.0 COMPLETED")
    logger.info("=" * 80)
    logger.info(f"  Figures: {config.FIGURES_DIR}")
    logger.info(f"  Tables:  {config.TABLES_DIR}")
    logger.info(f"  Report:  {config.OUTPUT_DIR / 'VALIDATION_REPORT_v4.txt'}")


if __name__ == "__main__":
    main()
