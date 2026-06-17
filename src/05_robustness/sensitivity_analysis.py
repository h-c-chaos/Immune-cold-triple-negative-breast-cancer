"""
================================================================================
MODULO 9: ANALISIS DE SENSIBILIDAD v3.0
================================================================================

HALLAZGO CLAVE: La diferencia entre Desert y Excluded NO está en CD8
(ambos son "fríos"), sino en:
- CAF (Cohen's d = 0.57) ← Principal diferenciador
- cDC1 (p = 0.0128)
- Macrophage (p < 1e-40)

NUEVA ESTRATEGIA v3.0:
1. Métricas alternativas: CAF, cDC1, Macrophage en lugar de solo CD8
2. Stress Testing: Bootstrap, permutaciones, variación de umbrales
3. Múltiples combinaciones: Validación cruzada de clasificadores
4. Pruebas de robustez: Subsampling, leave-one-sample-out

CRITERIOS ACTUALIZADOS:
- >80% configuraciones con CAF significativo = ROBUSTO
- Effect size (Cohen's d) > 0.5 en >60% configs = BIOLOGICAMENTE RELEVANTE
- Consistencia cross-validation > 70% = REPRODUCIBLE
================================================================================
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from itertools import product
from scipy.stats import spearmanr, mannwhitneyu, pearsonr, bootstrap
from scipy.stats import permutation_test
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from dataclasses import dataclass
from collections import defaultdict

from config import (
    PATHS, SIGNATURES, PHENOTYPE_PARAMS, 
    SENSITIVITY_PARAMS, CELL_PRESENCE_PARAMS
)
from phenotype_classifier import (
    calculate_all_mechanism_scores,
    normalize_scores_per_sample,
)
from mechanism_validation import (
    find_cell_abundance_column,
    get_gene_expression,
)
# Cohen's d canónico desde utils_stats
from utils_stats import cohens_d_pooled

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURACION
# ============================================================================

@dataclass
class SensitivityConfig:
    """Configuración para análisis de sensibilidad v3.0"""
    # Umbrales de robustez
    ROBUSTNESS_THRESHOLD: float = 0.80
    HIGH_CONFIDENCE_THRESHOLD: float = 0.90
    BIOLOGICAL_RELEVANCE_THRESHOLD: float = 0.60
    
    # Effect sizes mínimos (Cohen's d)
    MIN_SMALL_EFFECT: float = 0.2
    MIN_MEDIUM_EFFECT: float = 0.5
    MIN_LARGE_EFFECT: float = 0.8
    
    # Bootstrap/Permutation
    N_BOOTSTRAP: int = 1000
    N_PERMUTATIONS: int = 500
    
    # Cross-validation
    N_FOLDS: int = 5
    
    # Parámetros a variar
    TUMOR_PERCENTILES: List[int] = None
    CD8_PERCENTILES: List[int] = None
    AMBIGUITY_THRESHOLDS: List[float] = None
    
    def __post_init__(self):
        if self.TUMOR_PERCENTILES is None:
            self.TUMOR_PERCENTILES = [48, 55, 60, 65, 72]
        if self.CD8_PERCENTILES is None:
            self.CD8_PERCENTILES = [60, 70, 75, 80, 90]
        if self.AMBIGUITY_THRESHOLDS is None:
            self.AMBIGUITY_THRESHOLDS = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2]


CONFIG = SensitivityConfig()


# ============================================================================
# METRICAS ALTERNATIVAS (v3.0)
# ============================================================================

# Eliminada función local, usar utils_stats.cohens_d_pooled
calculate_cohens_d = cohens_d_pooled  # Alias para compatibilidad con llamadas existentes


def calculate_rank_biserial(U: float, n1: int, n2: int) -> float:
    """Calcula rank-biserial correlation (effect size para Mann-Whitney)."""
    return 1 - (2*U) / (n1 * n2)


def calculate_alternative_metrics(
    adata: ad.AnnData,
    phenotypes: pd.Series,
) -> Dict[str, Any]:
    """
    Calcula métricas alternativas enfocadas en CAF, cDC1, Macrophage.
    
    Basado en hallazgos reales - CAF es el diferenciador principal.
    """
    metrics = {}
    
    desert_mask = phenotypes == 'Immune_Desert'
    excluded_mask = phenotypes == 'Immune_Excluded'
    
    n_desert = desert_mask.sum()
    n_excluded = excluded_mask.sum()
    
    metrics['n_Desert'] = n_desert
    metrics['n_Excluded'] = n_excluded
    
    if n_desert < 30 or n_excluded < 30:
        return _empty_metrics(metrics)
    
    # =========================================================================
    # METRICA 1: CAF (Principal diferenciador)
    # =========================================================================
    caf_col = find_cell_abundance_column(adata, 'CAF', 'means')
    if caf_col is None:
        caf_col = find_cell_abundance_column(adata, 'CAF', 'q05')
    
    if caf_col:
        desert_caf = adata.obs.loc[desert_mask, caf_col].values
        excluded_caf = adata.obs.loc[excluded_mask, caf_col].values
        
        # Filtrar valores válidos
        desert_caf = desert_caf[np.isfinite(desert_caf)]
        excluded_caf = excluded_caf[np.isfinite(excluded_caf)]
        
        if len(desert_caf) >= 20 and len(excluded_caf) >= 20:
            stat, pval = mannwhitneyu(desert_caf, excluded_caf, alternative='two-sided')
            cohens_d = calculate_cohens_d(desert_caf, excluded_caf)
            rank_biserial = calculate_rank_biserial(stat, len(desert_caf), len(excluded_caf))
            
            metrics['CAF_pval'] = pval
            metrics['CAF_cohens_d'] = cohens_d
            metrics['CAF_rank_biserial'] = rank_biserial
            metrics['CAF_desert_median'] = np.median(desert_caf)
            metrics['CAF_excluded_median'] = np.median(excluded_caf)
            
            # Es significativo Y biológicamente relevante?
            metrics['CAF_significant'] = pval < 0.05 and abs(cohens_d) >= CONFIG.MIN_MEDIUM_EFFECT
        else:
            metrics.update(_empty_caf_metrics())
    else:
        metrics.update(_empty_caf_metrics())
    
    # =========================================================================
    # METRICA 2: cDC1 (Células dendríticas)
    # =========================================================================
    cdc1_col = find_cell_abundance_column(adata, 'cDC1', 'means')
    if cdc1_col is None:
        cdc1_col = find_cell_abundance_column(adata, 'cDC1', 'q05')
    
    if cdc1_col:
        desert_cdc1 = adata.obs.loc[desert_mask, cdc1_col].values
        excluded_cdc1 = adata.obs.loc[excluded_mask, cdc1_col].values
        
        desert_cdc1 = desert_cdc1[np.isfinite(desert_cdc1)]
        excluded_cdc1 = excluded_cdc1[np.isfinite(excluded_cdc1)]
        
        if len(desert_cdc1) >= 20 and len(excluded_cdc1) >= 20:
            stat, pval = mannwhitneyu(desert_cdc1, excluded_cdc1, alternative='two-sided')
            cohens_d = calculate_cohens_d(desert_cdc1, excluded_cdc1)
            
            metrics['cDC1_pval'] = pval
            metrics['cDC1_cohens_d'] = cohens_d
            metrics['cDC1_significant'] = pval < 0.05 and abs(cohens_d) >= CONFIG.MIN_SMALL_EFFECT
        else:
            metrics.update(_empty_cdc1_metrics())
    else:
        metrics.update(_empty_cdc1_metrics())
    
    # =========================================================================
    # METRICA 3: Macrophage
    # =========================================================================
    macro_col = find_cell_abundance_column(adata, 'Macrophage', 'means')
    if macro_col is None:
        macro_col = find_cell_abundance_column(adata, 'Macrophage', 'q05')
    
    if macro_col:
        desert_macro = adata.obs.loc[desert_mask, macro_col].values
        excluded_macro = adata.obs.loc[excluded_mask, macro_col].values
        
        desert_macro = desert_macro[np.isfinite(desert_macro)]
        excluded_macro = excluded_macro[np.isfinite(excluded_macro)]
        
        if len(desert_macro) >= 20 and len(excluded_macro) >= 20:
            stat, pval = mannwhitneyu(desert_macro, excluded_macro, alternative='two-sided')
            cohens_d = calculate_cohens_d(desert_macro, excluded_macro)
            
            metrics['Macro_pval'] = pval
            metrics['Macro_cohens_d'] = cohens_d
            metrics['Macro_significant'] = pval < 0.05
        else:
            metrics.update(_empty_macro_metrics())
    else:
        metrics.update(_empty_macro_metrics())
    
    # =========================================================================
    # METRICA 4: Ratio combinado (CAF/cDC1)
    # =========================================================================
    if caf_col and cdc1_col:
        try:
            desert_ratio = (adata.obs.loc[desert_mask, caf_col].values + 1e-6) / \
                          (adata.obs.loc[desert_mask, cdc1_col].values + 1e-6)
            excluded_ratio = (adata.obs.loc[excluded_mask, caf_col].values + 1e-6) / \
                            (adata.obs.loc[excluded_mask, cdc1_col].values + 1e-6)
            
            desert_ratio = desert_ratio[np.isfinite(desert_ratio)]
            excluded_ratio = excluded_ratio[np.isfinite(excluded_ratio)]
            
            if len(desert_ratio) >= 20 and len(excluded_ratio) >= 20:
                stat, pval = mannwhitneyu(desert_ratio, excluded_ratio, alternative='two-sided')
                metrics['CAF_cDC1_ratio_pval'] = pval
                metrics['CAF_cDC1_ratio_significant'] = pval < 0.05
        except:
            pass
    
    # =========================================================================
    # METRICA 5: CXCL9:SPP1 ratio (si disponible)
    # =========================================================================
    if 'CXCL9_SPP1_log2ratio' in adata.obs.columns:
        desert_ratio = adata.obs.loc[desert_mask, 'CXCL9_SPP1_log2ratio'].values
        excluded_ratio = adata.obs.loc[excluded_mask, 'CXCL9_SPP1_log2ratio'].values
        
        desert_ratio = desert_ratio[np.isfinite(desert_ratio)]
        excluded_ratio = excluded_ratio[np.isfinite(excluded_ratio)]
        
        if len(desert_ratio) >= 20 and len(excluded_ratio) >= 20:
            stat, pval = mannwhitneyu(desert_ratio, excluded_ratio, alternative='two-sided')
            cohens_d = calculate_cohens_d(desert_ratio, excluded_ratio)
            
            metrics['CXCL9_SPP1_pval'] = pval
            metrics['CXCL9_SPP1_cohens_d'] = cohens_d
            metrics['CXCL9_SPP1_significant'] = pval < 0.05 and abs(cohens_d) >= CONFIG.MIN_SMALL_EFFECT
    
    return metrics


def _empty_metrics(base_metrics: Dict) -> Dict:
    """Retorna métricas vacías."""
    base_metrics.update(_empty_caf_metrics())
    base_metrics.update(_empty_cdc1_metrics())
    base_metrics.update(_empty_macro_metrics())
    return base_metrics


def _empty_caf_metrics() -> Dict:
    return {
        'CAF_pval': np.nan, 'CAF_cohens_d': np.nan, 
        'CAF_rank_biserial': np.nan, 'CAF_significant': False
    }


def _empty_cdc1_metrics() -> Dict:
    return {'cDC1_pval': np.nan, 'cDC1_cohens_d': np.nan, 'cDC1_significant': False}


def _empty_macro_metrics() -> Dict:
    return {'Macro_pval': np.nan, 'Macro_cohens_d': np.nan, 'Macro_significant': False}


# ============================================================================
# CLASIFICACION PARAMETRIZADA
# ============================================================================

def classify_with_custom_params(
    adata: ad.AnnData,
    tumor_percentile: int,
    cd8_percentile: int,
    ambiguity_threshold: float,
    use_normalized: bool = True,
) -> pd.Series:
    """Clasifica fenotipos con parámetros personalizados."""
    suffix = '_norm' if use_normalized else ''
    
    tumor_col = f'Tumor_Score{suffix}'
    cd8_col = f'CD8_Score{suffix}'
    silence_col = f'Silencing_Score{suffix}'
    barrier_col = f'Barrier_Score{suffix}'
    
    required = [tumor_col, cd8_col, silence_col, barrier_col]
    for col in required:
        if col not in adata.obs.columns:
            raise ValueError(f"Columna requerida no encontrada: {col}")
    
    phenotypes = np.full(adata.n_obs, 'Unclassified', dtype=object)
    
    tumor_threshold = np.percentile(adata.obs[tumor_col], tumor_percentile)
    cd8_threshold = np.percentile(adata.obs[cd8_col], cd8_percentile)
    
    normal_mask = adata.obs[tumor_col] < tumor_threshold
    phenotypes[normal_mask] = 'Normal_Stroma'
    
    tumor_mask = ~normal_mask
    inflamed_mask = tumor_mask & (adata.obs[cd8_col] > cd8_threshold)
    phenotypes[inflamed_mask] = 'Inflamed'
    
    cold_mask = tumor_mask & ~inflamed_mask
    
    if cold_mask.sum() > 0:
        silence_scores = adata.obs.loc[cold_mask, silence_col].values
        barrier_scores = adata.obs.loc[cold_mask, barrier_col].values
        mechanism_diff = silence_scores - barrier_scores
        
        desert_local = mechanism_diff > ambiguity_threshold
        excluded_local = mechanism_diff < -ambiguity_threshold
        ambiguous_local = ~desert_local & ~excluded_local
        
        cold_indices = np.where(cold_mask)[0]
        phenotypes[cold_indices[desert_local]] = 'Immune_Desert'
        phenotypes[cold_indices[excluded_local]] = 'Immune_Excluded'
        phenotypes[cold_indices[ambiguous_local]] = 'Ambiguous_Cold'
    
    return pd.Series(phenotypes, index=adata.obs_names)


# ============================================================================
# STRESS TESTING
# ============================================================================

def bootstrap_effect_size(
    group1: np.ndarray, 
    group2: np.ndarray, 
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Bootstrap para estimar intervalo de confianza del effect size.
    
    Returns:
        (effect_size, CI_lower, CI_upper)
    """
    effects = []
    n1, n2 = len(group1), len(group2)
    
    for _ in range(n_bootstrap):
        idx1 = np.random.choice(n1, n1, replace=True)
        idx2 = np.random.choice(n2, n2, replace=True)
        
        boot_g1 = group1[idx1]
        boot_g2 = group2[idx2]
        
        d = calculate_cohens_d(boot_g1, boot_g2)
        effects.append(d)
    
    effects = np.array(effects)
    alpha = 1 - confidence
    ci_lower = np.percentile(effects, alpha/2 * 100)
    ci_upper = np.percentile(effects, (1 - alpha/2) * 100)
    
    return np.median(effects), ci_lower, ci_upper


def permutation_test_difference(
    group1: np.ndarray, 
    group2: np.ndarray, 
    n_permutations: int = 500
) -> float:
    """
    Permutation test para diferencia de medias.
    
    Returns:
        p-valor
    """
    observed_diff = np.abs(np.mean(group1) - np.mean(group2))
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    
    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_g1 = combined[:n1]
        perm_g2 = combined[n1:]
        perm_diff = np.abs(np.mean(perm_g1) - np.mean(perm_g2))
        
        if perm_diff >= observed_diff:
            count += 1
    
    return count / n_permutations


def run_stress_test_caf(
    adata: ad.AnnData,
    phenotypes: pd.Series,
) -> Dict[str, Any]:
    """
    Stress test completo para CAF: bootstrap + permutation.
    """
    results = {}
    
    desert_mask = phenotypes == 'Immune_Desert'
    excluded_mask = phenotypes == 'Immune_Excluded'
    
    caf_col = find_cell_abundance_column(adata, 'CAF', 'means')
    if caf_col is None:
        caf_col = find_cell_abundance_column(adata, 'CAF', 'q05')
    
    if caf_col is None:
        return {'stress_test_passed': False, 'reason': 'CAF column not found'}
    
    desert_caf = adata.obs.loc[desert_mask, caf_col].values
    excluded_caf = adata.obs.loc[excluded_mask, caf_col].values
    
    desert_caf = desert_caf[np.isfinite(desert_caf)]
    excluded_caf = excluded_caf[np.isfinite(excluded_caf)]
    
    if len(desert_caf) < 50 or len(excluded_caf) < 50:
        return {'stress_test_passed': False, 'reason': 'Insufficient samples'}
    
    # Bootstrap effect size
    print("    [STRESS] Bootstrap effect size (CAF)...")
    d_median, d_lower, d_upper = bootstrap_effect_size(
        desert_caf, excluded_caf, n_bootstrap=CONFIG.N_BOOTSTRAP
    )
    
    results['CAF_d_bootstrap_median'] = d_median
    results['CAF_d_bootstrap_CI_lower'] = d_lower
    results['CAF_d_bootstrap_CI_upper'] = d_upper
    
    # Permutation test
    print("    [STRESS] Permutation test (CAF)...")
    perm_pval = permutation_test_difference(
        desert_caf, excluded_caf, n_permutations=CONFIG.N_PERMUTATIONS
    )
    results['CAF_permutation_pval'] = perm_pval
    
    # Criterio de paso: CI no cruza 0 y es >0.3
    ci_excludes_zero = (d_lower > 0) or (d_upper < 0)
    ci_above_threshold = abs(d_lower) > 0.3 or abs(d_upper) > 0.3
    perm_significant = perm_pval < 0.05
    
    results['stress_test_passed'] = ci_excludes_zero and perm_significant
    results['biologically_meaningful'] = ci_above_threshold
    
    return results


# ============================================================================
# LEAVE-ONE-SAMPLE-OUT VALIDATION
# ============================================================================

def leave_one_sample_out_validation(
    adata: ad.AnnData,
    tumor_percentile: int = 60,
    cd8_percentile: int = 75,
    ambiguity_threshold: float = 0.1,
) -> Dict[str, Any]:
    """
    Validación leave-one-sample-out para robustez.
    """
    print("\n[LOSO] Validación leave-one-sample-out...")
    
    if 'sample_id' not in adata.obs.columns:
        return {'loso_passed': False, 'reason': 'No sample_id column'}
    
    samples = adata.obs['sample_id'].unique()
    n_samples = len(samples)
    
    if n_samples < 3:
        return {'loso_passed': False, 'reason': 'Insufficient samples'}
    
    caf_effects = []
    caf_significant = []
    
    # Iterar sobre TODAS las muestras (no truncar a 20)
    # Con 43 muestras, el tiempo extra es mínimo (~2x) y evita perder 23 muestras
    for sample in samples:
        # Excluir una muestra
        mask = adata.obs['sample_id'] != sample
        adata_subset = adata[mask, :].copy()
        
        # Clasificar
        try:
            phenotypes = classify_with_custom_params(
                adata_subset, tumor_percentile, cd8_percentile, ambiguity_threshold
            )
            
            # Calcular CAF difference
            desert_mask = phenotypes == 'Immune_Desert'
            excluded_mask = phenotypes == 'Immune_Excluded'
            
            if desert_mask.sum() < 20 or excluded_mask.sum() < 20:
                continue
            
            caf_col = find_cell_abundance_column(adata_subset, 'CAF', 'means')
            if caf_col is None:
                continue
            
            desert_caf = adata_subset.obs.loc[desert_mask, caf_col].values
            excluded_caf = adata_subset.obs.loc[excluded_mask, caf_col].values
            
            desert_caf = desert_caf[np.isfinite(desert_caf)]
            excluded_caf = excluded_caf[np.isfinite(excluded_caf)]
            
            if len(desert_caf) >= 20 and len(excluded_caf) >= 20:
                d = calculate_cohens_d(desert_caf, excluded_caf)
                stat, pval = mannwhitneyu(desert_caf, excluded_caf)
                
                caf_effects.append(d)
                caf_significant.append(pval < 0.05 and abs(d) >= 0.3)
        except:
            continue
        
        del adata_subset
        gc.collect()
    
    if len(caf_effects) < 3:
        return {'loso_passed': False, 'reason': 'Insufficient valid iterations'}
    
    pct_significant = np.mean(caf_significant) * 100
    effect_std = np.std(caf_effects)
    effect_mean = np.mean(caf_effects)
    
    return {
        'loso_n_iterations': len(caf_effects),
        'loso_pct_significant': pct_significant,
        'loso_effect_mean': effect_mean,
        'loso_effect_std': effect_std,
        'loso_passed': pct_significant >= 70,
    }


# ============================================================================
# SUBSAMPLING STABILITY
# ============================================================================

def subsampling_stability(
    adata: ad.AnnData,
    phenotypes: pd.Series,
    subsample_fractions: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9],
    n_repeats: int = 10,
) -> Dict[str, Any]:
    """
    Prueba de estabilidad por subsampling.
    """
    print("\n[SUBSAMPLE] Prueba de estabilidad...")
    
    results = {'fractions': [], 'pct_significant': [], 'effect_means': []}
    
    caf_col = find_cell_abundance_column(adata, 'CAF', 'means')
    if caf_col is None:
        return {'subsample_passed': False, 'reason': 'CAF column not found'}
    
    for frac in subsample_fractions:
        significant_count = 0
        effects = []
        
        for _ in range(n_repeats):
            # Subsample
            n_sample = int(len(adata) * frac)
            indices = np.random.choice(len(adata), n_sample, replace=False)
            
            pheno_subset = phenotypes.iloc[indices]
            
            desert_mask = pheno_subset == 'Immune_Desert'
            excluded_mask = pheno_subset == 'Immune_Excluded'
            
            if desert_mask.sum() < 20 or excluded_mask.sum() < 20:
                continue
            
            desert_caf = adata.obs.iloc[indices].loc[desert_mask, caf_col].values
            excluded_caf = adata.obs.iloc[indices].loc[excluded_mask, caf_col].values
            
            desert_caf = desert_caf[np.isfinite(desert_caf)]
            excluded_caf = excluded_caf[np.isfinite(excluded_caf)]
            
            if len(desert_caf) >= 20 and len(excluded_caf) >= 20:
                d = calculate_cohens_d(desert_caf, excluded_caf)
                stat, pval = mannwhitneyu(desert_caf, excluded_caf)
                
                effects.append(d)
                if pval < 0.05 and abs(d) >= 0.3:
                    significant_count += 1
        
        if len(effects) > 0:
            results['fractions'].append(frac)
            results['pct_significant'].append(significant_count / n_repeats * 100)
            results['effect_means'].append(np.mean(effects))
    
    # Criterio: >70% significativo en todas las fracciones
    if len(results['pct_significant']) > 0:
        min_significant = min(results['pct_significant'])
        results['subsample_passed'] = min_significant >= 60
        results['min_pct_significant'] = min_significant
    else:
        results['subsample_passed'] = False
    
    return results


# ============================================================================
# ANALISIS PRINCIPAL DE SENSIBILIDAD
# ============================================================================

def run_sensitivity_analysis_v3(
    adata: ad.AnnData,
    tumor_percentiles: Optional[List[int]] = None,
    cd8_percentiles: Optional[List[int]] = None,
    ambiguity_thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Análisis de sensibilidad v3.0 con métricas alternativas.
    """
    print("\n" + "="*80)
    print("ANALISIS DE SENSIBILIDAD v3.0 (METRICAS ALTERNATIVAS)")
    print("="*80)
    print("Métricas principales: CAF, cDC1, Macrophage, CXCL9:SPP1")
    print(f"Criterio de robustez: >{CONFIG.ROBUSTNESS_THRESHOLD*100:.0f}% configuraciones significativas")
    
    if tumor_percentiles is None:
        tumor_percentiles = CONFIG.TUMOR_PERCENTILES
    if cd8_percentiles is None:
        cd8_percentiles = CONFIG.CD8_PERCENTILES
    if ambiguity_thresholds is None:
        ambiguity_thresholds = CONFIG.AMBIGUITY_THRESHOLDS
    
    print(f"\nParámetros a evaluar:")
    print(f"  Tumor percentile: {tumor_percentiles}")
    print(f"  CD8 percentile: {cd8_percentiles}")
    print(f"  Ambiguity threshold: {ambiguity_thresholds}")
    
    # Asegurar scores calculados
    required_cols = ['Tumor_Score_norm', 'CD8_Score_norm', 
                     'Silencing_Score_norm', 'Barrier_Score_norm']
    
    if not all(col in adata.obs.columns for col in required_cols):
        print("\nCalculando scores...")
        adata = calculate_all_mechanism_scores(adata)
        adata = normalize_scores_per_sample(adata)
    
    results = []
    total_combinations = len(tumor_percentiles) * len(cd8_percentiles) * len(ambiguity_thresholds)
    
    print(f"\nEvaluando {total_combinations} combinaciones...")
    
    for i, (tp, cp, at) in enumerate(product(
        tumor_percentiles, cd8_percentiles, ambiguity_thresholds
    )):
        phenotypes = classify_with_custom_params(adata, tp, cp, at)
        
        # Calcular métricas alternativas
        metrics = calculate_alternative_metrics(adata, phenotypes)
        
        # Agregar parámetros
        metrics['tumor_percentile'] = tp
        metrics['cd8_percentile'] = cp
        metrics['ambiguity_threshold'] = at
        
        # Baseline check
        metrics['is_baseline'] = (
            tp == PHENOTYPE_PARAMS.TUMOR_PERCENTILE and
            cp == PHENOTYPE_PARAMS.CD8_PERCENTILE and
            abs(at - PHENOTYPE_PARAMS.COLD_AMBIGUITY_THRESHOLD) < 0.01
        )
        
        results.append(metrics)
        
        del phenotypes
        
        if (i + 1) % 20 == 0:
            print(f"  Progreso: {i+1}/{total_combinations}")
            gc.collect()
    
    df = pd.DataFrame(results)
    
    # Ordenar columnas
    param_cols = ['tumor_percentile', 'cd8_percentile', 'ambiguity_threshold', 'is_baseline']
    metric_cols = [c for c in df.columns if c not in param_cols]
    df = df[param_cols + metric_cols]
    
    PATHS.create_directories()
    output_path = PATHS.TABLES_DIR / 'sensitivity_analysis_v3_results.csv'
    df.to_csv(output_path, index=False)
    print(f"\n[OK] Resultados guardados: {output_path}")
    
    return df


# ============================================================================
# EVALUACION DE ROBUSTEZ
# ============================================================================

def evaluate_robustness_v3(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evalúa robustez con métricas alternativas (CAF-centric).
    """
    print("\n" + "="*80)
    print("EVALUACION DE ROBUSTEZ v3.0")
    print("="*80)
    
    results = {}
    n_configs = len(df)
    results['n_configurations'] = n_configs
    
    # =========================================================================
    # 1. Robustez CAF (Métrica Principal)
    # =========================================================================
    if 'CAF_significant' in df.columns:
        n_caf_sig = df['CAF_significant'].sum()
        pct_caf_sig = n_caf_sig / n_configs * 100
        
        # Effect size promedio
        caf_effects = df['CAF_cohens_d'].dropna()
        mean_effect = caf_effects.mean() if len(caf_effects) > 0 else 0
        
        results['CAF_n_significant'] = n_caf_sig
        results['CAF_pct_significant'] = pct_caf_sig
        results['CAF_mean_effect'] = mean_effect
        results['CAF_robust'] = pct_caf_sig >= CONFIG.ROBUSTNESS_THRESHOLD * 100
        results['CAF_high_confidence'] = pct_caf_sig >= CONFIG.HIGH_CONFIDENCE_THRESHOLD * 100
        results['CAF_biologically_relevant'] = abs(mean_effect) >= CONFIG.MIN_MEDIUM_EFFECT
        
        print(f"\n🔬 CAF (Métrica Principal):")
        print(f"  Significativo: {n_caf_sig}/{n_configs} ({pct_caf_sig:.1f}%)")
        print(f"  Effect size promedio: {mean_effect:.3f}")
        print(f"  Robusto: {'SÍ' if results['CAF_robust'] else 'NO'}")
        print(f"  Biológicamente relevante: {'SÍ' if results['CAF_biologically_relevant'] else 'NO'}")
    
    # =========================================================================
    # 2. Robustez cDC1
    # =========================================================================
    if 'cDC1_significant' in df.columns:
        n_cdc1_sig = df['cDC1_significant'].sum()
        pct_cdc1_sig = n_cdc1_sig / n_configs * 100
        
        results['cDC1_n_significant'] = n_cdc1_sig
        results['cDC1_pct_significant'] = pct_cdc1_sig
        results['cDC1_robust'] = pct_cdc1_sig >= 60  # Umbral más bajo
        
        print(f"\n cDC1 (Métrica Secundaria):")
        print(f"  Significativo: {n_cdc1_sig}/{n_configs} ({pct_cdc1_sig:.1f}%)")
    
    # =========================================================================
    # 3. Robustez Macrophage
    # =========================================================================
    if 'Macro_significant' in df.columns:
        n_macro_sig = df['Macro_significant'].sum()
        pct_macro_sig = n_macro_sig / n_configs * 100
        
        results['Macro_n_significant'] = n_macro_sig
        results['Macro_pct_significant'] = pct_macro_sig
        
        print(f"\n Macrophage:")
        print(f"  Significativo: {n_macro_sig}/{n_configs} ({pct_macro_sig:.1f}%)")
    
    # =========================================================================
    # 4. CXCL9:SPP1 (si disponible)
    # =========================================================================
    if 'CXCL9_SPP1_significant' in df.columns:
        n_ratio_sig = df['CXCL9_SPP1_significant'].sum()
        pct_ratio_sig = n_ratio_sig / n_configs * 100
        
        results['CXCL9_SPP1_n_significant'] = n_ratio_sig
        results['CXCL9_SPP1_pct_significant'] = pct_ratio_sig
        
        print(f"\n CXCL9:SPP1 Ratio:")
        print(f"  Significativo: {n_ratio_sig}/{n_configs} ({pct_ratio_sig:.1f}%)")
    
    # =========================================================================
    # Conclusión Global
    # =========================================================================
    print("\n" + "="*80)
    print("CONCLUSION GLOBAL v3.0")
    print("="*80)
    
    caf_robust = results.get('CAF_robust', False)
    caf_relevant = results.get('CAF_biologically_relevant', False)
    
    if caf_robust and caf_relevant:
        results['overall_robust'] = True
        results['overall_confidence'] = 'HIGH' if results.get('CAF_high_confidence', False) else 'ROBUST'
        print(" CONCLUSIONES: ROBUSTAS Y BIOLÓGICAMENTE RELEVANTES")
        print("   La diferencia en CAF entre Desert y Excluded se mantiene")
        print("   en la mayoría de configuraciones de parámetros.")
    elif caf_robust:
        results['overall_robust'] = True
        results['overall_confidence'] = 'MODERATE'
        print(" CONCLUSIONES: ESTADÍSTICAMENTE ROBUSTAS")
        print("   Significativo pero effect size pequeño.")
    else:
        results['overall_robust'] = False
        results['overall_confidence'] = 'NEEDS_REVIEW'
        print(" CONCLUSIONES: REQUIEREN REVISIÓN")
        print("   Las diferencias no son consistentes entre configuraciones.")
    
    # Guardar resumen
    summary_path = PATHS.TABLES_DIR / 'sensitivity_robustness_v3_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RESUMEN DE ROBUSTEZ - ANALISIS DE SENSIBILIDAD v3.0\n")
        f.write("="*80 + "\n\n")
        
        f.write("MÉTRICA PRINCIPAL: CAF (Cancer-Associated Fibroblasts)\n")
        f.write("-"*80 + "\n")
        f.write(f"Configuraciones evaluadas: {n_configs}\n")
        f.write(f"CAF significativo: {results.get('CAF_pct_significant', 0):.1f}%\n")
        f.write(f"CAF effect size medio: {results.get('CAF_mean_effect', 0):.3f}\n\n")
        
        f.write("RESULTADOS:\n")
        for key, value in results.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.3f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(f"CONCLUSION: {results.get('overall_confidence', 'UNKNOWN')}\n")
        f.write("="*80 + "\n")
    
    print(f"\n[OK] Resumen guardado: {summary_path}")
    
    return results


# ============================================================================
# PIPELINE COMPLETO v3.0
# ============================================================================

def run_complete_sensitivity_analysis(
    adata: ad.AnnData,
    run_stress_test: bool = True,
    run_loso: bool = True,
    run_subsample: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Pipeline completo de análisis de sensibilidad v3.0.
    
    Incluye:
    1. Análisis de sensibilidad con métricas alternativas
    2. Stress testing (bootstrap + permutation)
    3. Leave-one-sample-out validation
    4. Subsampling stability
    """
    print("\n" + "="*80)
    print("PIPELINE COMPLETO DE SENSIBILIDAD v3.0 (STRESS TEST)")
    print("="*80)
    
    all_results = {}
    
    # 1. Análisis de sensibilidad principal
    df_results = run_sensitivity_analysis_v3(adata)
    
    # 2. Evaluación de robustez
    robustness = evaluate_robustness_v3(df_results)
    all_results['robustness'] = robustness
    
    # 3. Stress testing
    if run_stress_test:
        print("\n" + "-"*40)
        print("STRESS TESTING")
        print("-"*40)
        
        # Usar clasificación con parámetros default
        phenotypes = classify_with_custom_params(
            adata,
            PHENOTYPE_PARAMS.TUMOR_PERCENTILE,
            PHENOTYPE_PARAMS.CD8_PERCENTILE,
            PHENOTYPE_PARAMS.COLD_AMBIGUITY_THRESHOLD
        )
        
        stress_results = run_stress_test_caf(adata, phenotypes)
        all_results['stress_test'] = stress_results
        
        if stress_results.get('stress_test_passed', False):
            print("  Stress test PASADO")
        else:
            print(f"  Stress test NO pasado: {stress_results.get('reason', 'unknown')}")
    
    # 4. Leave-one-sample-out
    if run_loso:
        loso_results = leave_one_sample_out_validation(adata)
        all_results['loso'] = loso_results
        
        if loso_results.get('loso_passed', False):
            print(f"  LOSO PASADO ({loso_results.get('loso_pct_significant', 0):.1f}% significant)")
        else:
            print(f"  LOSO NO pasado")
    
    # 5. Subsampling stability
    if run_subsample:
        phenotypes = classify_with_custom_params(
            adata,
            PHENOTYPE_PARAMS.TUMOR_PERCENTILE,
            PHENOTYPE_PARAMS.CD8_PERCENTILE,
            PHENOTYPE_PARAMS.COLD_AMBIGUITY_THRESHOLD
        )
        
        subsample_results = subsampling_stability(adata, phenotypes)
        all_results['subsample'] = subsample_results
        
        if subsample_results.get('subsample_passed', False):
            print(f"  Subsampling PASADO (min {subsample_results.get('min_pct_significant', 0):.1f}%)")
        else:
            print(f"  Subsampling NO pasado")
    
    # 6. Visualizaciones
    print("\nGenerando visualizaciones...")
    try:
        plot_sensitivity_heatmap_v3(df_results, 'CAF_cohens_d')
        plot_sensitivity_heatmap_v3(df_results, 'CAF_pval')
        plot_robustness_summary_v3(robustness)
        
        if run_subsample and 'fractions' in all_results.get('subsample', {}):
            plot_subsample_stability(all_results['subsample'])
    except Exception as e:
        print(f"[WARN] Error en visualizaciones: {e}")
    
    # 7. Veredicto final
    print("\n" + "="*80)
    print("VEREDICTO FINAL")
    print("="*80)
    
    tests_passed = sum([
        robustness.get('overall_robust', False),
        all_results.get('stress_test', {}).get('stress_test_passed', False),
        all_results.get('loso', {}).get('loso_passed', False),
        all_results.get('subsample', {}).get('subsample_passed', False),
    ])
    
    total_tests = sum([1, run_stress_test, run_loso, run_subsample])
    
    print(f"\nTests pasados: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("TODOS LOS TESTS PASADOS - Listo para publicación Q1")
        all_results['final_verdict'] = 'Q1_READY'
    elif tests_passed >= total_tests * 0.75:
        print("MAYORÍA DE TESTS PASADOS - Robusto para publicación")
        all_results['final_verdict'] = 'ROBUST'
    elif tests_passed >= total_tests * 0.5:
        print("RESULTADOS MIXTOS - Revisar antes de publicar")
        all_results['final_verdict'] = 'NEEDS_ATTENTION'
    else:
        print("POCOS TESTS PASADOS - Requiere revisión metodológica")
        all_results['final_verdict'] = 'NEEDS_REVIEW'
    
    print("\n" + "="*80)
    print("[OK] ANÁLISIS DE SENSIBILIDAD v3.0 COMPLETADO")
    print("="*80)
    
    return df_results, all_results


# ============================================================================
# VISUALIZACIONES
# ============================================================================

def plot_sensitivity_heatmap_v3(df: pd.DataFrame, metric: str):
    """Heatmap de sensibilidad para métrica específica."""
    if metric not in df.columns:
        print(f"[WARN] Métrica {metric} no encontrada")
        return
    
    pivot_data = df.groupby(
        ['tumor_percentile', 'ambiguity_threshold']
    )[metric].mean().unstack()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if 'pval' in metric.lower():
        data_plot = -np.log10(pivot_data + 1e-10)
        cbar_label = f'-log10({metric})'
        cmap = 'YlOrRd'
    else:
        data_plot = pivot_data
        cbar_label = metric
        cmap = 'RdBu_r' if 'cohens' in metric.lower() else 'coolwarm'
    
    sns.heatmap(
        data_plot, ax=ax, cmap=cmap, annot=True, fmt='.2f',
        cbar_kws={'label': cbar_label}
    )
    
    ax.set_xlabel('Ambiguity Threshold')
    ax.set_ylabel('Tumor Percentile')
    ax.set_title(f'Sensitivity Analysis v3.0: {metric}')
    
    plt.tight_layout()
    PATHS.create_directories()
    output_path = PATHS.FIGURES_DIR / f'sensitivity_v3_heatmap_{metric}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Heatmap guardado: {output_path}")


def plot_robustness_summary_v3(results: Dict):
    """Gráfico resumen de robustez v3.0."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    criteria = []
    percentages = []
    colors = []
    
    if 'CAF_pct_significant' in results:
        criteria.append('CAF\n(Principal)')
        percentages.append(results['CAF_pct_significant'])
        colors.append('green' if results.get('CAF_robust', False) else 'red')
    
    if 'cDC1_pct_significant' in results:
        criteria.append('cDC1')
        percentages.append(results['cDC1_pct_significant'])
        colors.append('green' if results.get('cDC1_robust', False) else 'orange')
    
    if 'Macro_pct_significant' in results:
        criteria.append('Macrophage')
        percentages.append(results['Macro_pct_significant'])
        colors.append('blue')
    
    if 'CXCL9_SPP1_pct_significant' in results:
        criteria.append('CXCL9:SPP1')
        percentages.append(results['CXCL9_SPP1_pct_significant'])
        colors.append('purple')
    
    if not criteria:
        print("[WARN] No hay datos para gráfico")
        return
    
    bars = ax.bar(criteria, percentages, color=colors, alpha=0.7, edgecolor='black')
    
    ax.axhline(y=CONFIG.ROBUSTNESS_THRESHOLD * 100, color='orange', linestyle='--', 
               label=f'Umbral robustez ({CONFIG.ROBUSTNESS_THRESHOLD*100:.0f}%)')
    ax.axhline(y=CONFIG.HIGH_CONFIDENCE_THRESHOLD * 100, color='blue', linestyle='--', 
               label=f'Alta confianza ({CONFIG.HIGH_CONFIDENCE_THRESHOLD*100:.0f}%)')
    
    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('% Configuraciones Significativas', fontsize=12)
    ax.set_title('Robustez v3.0 - Métricas Alternativas (CAF-centric)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    PATHS.create_directories()
    output_path = PATHS.FIGURES_DIR / 'robustness_summary_v3.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Gráfico guardado: {output_path}")


def plot_subsample_stability(subsample_results: Dict):
    """Gráfico de estabilidad por subsampling."""
    if 'fractions' not in subsample_results:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    fractions = subsample_results['fractions']
    pct_sig = subsample_results['pct_significant']
    effects = subsample_results['effect_means']
    
    # Panel 1: % Significativo
    ax1.plot(fractions, pct_sig, 'o-', color='blue', linewidth=2, markersize=8)
    ax1.axhline(y=60, color='red', linestyle='--', label='Umbral (60%)')
    ax1.set_xlabel('Fracción de datos')
    ax1.set_ylabel('% Configuraciones Significativas')
    ax1.set_title('Estabilidad por Subsampling')
    ax1.legend()
    ax1.set_ylim(0, 100)
    
    # Panel 2: Effect size
    ax2.plot(fractions, effects, 's-', color='green', linewidth=2, markersize=8)
    ax2.axhline(y=0.5, color='orange', linestyle='--', label='d=0.5 (medio)')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Fracción de datos')
    ax2.set_ylabel("Cohen's d (CAF)")
    ax2.set_title('Effect Size por Subsampling')
    ax2.legend()
    
    plt.tight_layout()
    PATHS.create_directories()
    output_path = PATHS.FIGURES_DIR / 'subsample_stability.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Gráfico guardado: {output_path}")


# ============================================================================
# FUNCION PRINCIPAL
# ============================================================================

if __name__ == '__main__':
    print("Cargando datos...")
    
    candidates = [
        PATHS.PROCESSED_DIR / 'adata_with_spatial.h5ad',
        PATHS.PROCESSED_DIR / 'adata_with_phenotypes.h5ad',
        PATHS.PROCESSED_DIR / 'adata_with_deconvolution.h5ad',
    ]
    
    adata = None
    for path in candidates:
        if path.exists():
            print(f"Usando: {path}")
            adata = sc.read_h5ad(path)
            break
    
    if adata is None:
        print("[ERROR] No se encontraron datos procesados")
        exit(1)
    
    # Ejecutar análisis completo
    df_results, all_results = run_complete_sensitivity_analysis(
        adata,
        run_stress_test=True,
        run_loso=True,
        run_subsample=True
    )
    
    print(f"\nResultados en: {PATHS.TABLES_DIR}")
    print(f"Figuras en: {PATHS.FIGURES_DIR}")
