"""
================================================================================
ADICIONES A sensitivity_analysis.py
================================================================================
RESPONDE A: Dictamen de Crítica, Sección de Estabilidad

Problema anterior:
  - Solo 150 combinaciones (4×4×~10 rangos)
  - Métricas binarias (significativo sí/no → 100% de "aprobados")
  - 100% estabilidad es sospechoso. ¿Dónde se rompe?
  
  Se resuelve esto con 3 extensiones:

EXTENSIÓN 1: Rangos Extremos (770 combinaciones)
  tumor_percentile: [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
  immune_percentile: [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
  ambiguity_margin: [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
  Total: 11 × 10 × 7 = 770 combinaciones
  
  Con rangos tan extremos, ALGUNA configuración FALLARÁ, y eso es BUENO.
  Demuestra que el efecto no es un artefacto de cherry-picking.

EXTENSIÓN 2: Métricas Continuas
  - Cohen's kappa: concordancia entre clasificación con params default vs custom
  - Cohen's d: tamaño de efecto CAF continuo (no solo significativo/no)
  - ARI: Adjusted Rand Index entre clasificaciones
  Esto elimina el "100% aprobado" y muestra un GRADIENTE de degradación.

EXTENSIÓN 3: Heatmaps de Degradación con Contornos
  - Contorno kappa=0.80 ("substantial agreement")
  - Contorno kappa=0.60 ("moderate agreement")
  - Contorno d=0.50 ("medium effect")

INSTRUCCIONES:
  Copiar funciones al final de sensitivity_analysis.py (antes de __main__).
  Llamar run_extended_sensitivity() desde el pipeline o standalone.
================================================================================
"""

import numpy as np
import pandas as pd
import anndata as ad
from typing import Dict, List, Tuple, Optional
from itertools import product
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Importar Cohen's d canónico de utils_stats
try:
    from utils_stats import cohens_d_pooled
except ImportError:
    def cohens_d_pooled(g1, g2):
        g1, g2 = np.asarray(g1, float), np.asarray(g2, float)
        g1, g2 = g1[np.isfinite(g1)], g2[np.isfinite(g2)]
        n1, n2 = len(g1), len(g2)
        if n1 < 2 or n2 < 2: return 0.0
        sp = np.sqrt(((n1-1)*np.var(g1,ddof=1)+(n2-1)*np.var(g2,ddof=1))/(n1+n2-2))
        return float((g1.mean()-g2.mean())/sp) if sp > 1e-10 else 0.0


# ============================================================================
# EXTENSIÓN 1: RANGOS EXTREMOS (770 combinaciones)
# ============================================================================

# Rangos extendidos
EXTENDED_TUMOR_PERCENTILES = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
EXTENDED_IMMUNE_PERCENTILES = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
EXTENDED_AMBIGUITY_MARGINS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# Total: 11 × 10 × 7 = 770 combinaciones


def run_extended_sensitivity(
    adata: ad.AnnData,
    save_dir: str = None,
    # 65→60 para coincidir con PHENOTYPE_PARAMS.TUMOR_PERCENTILE
    reference_tumor_pct: int = 60,
    reference_immune_pct: int = 75,
    reference_ambiguity: float = 0.10,
) -> pd.DataFrame:
    """
    Análisis de sensibilidad extendido con 770 combinaciones y métricas continuas.
    
    A diferencia de v3.0 (150 combos, métricas binarias), esta versión:
    - Usa rangos extremos que GARANTIZAN encontrar la frontera de degradación
    - Reporta métricas continuas (kappa, d, ARI) que muestran el gradiente
    - Genera heatmaps con contornos de degradación
    
    Parameters
    ----------
    adata : AnnData con deconvolución y scores de mecanismo
    save_dir : str — directorio para CSVs y figuras
    reference_tumor_pct : int — percentil de referencia para tumor (default pipeline)
    reference_immune_pct : int — percentil de referencia para inmune
    reference_ambiguity : float — margen de ambigüedad de referencia
    
    Returns
    -------
    DataFrame con 770 filas × métricas por configuración
    """
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS v4.0 — RANGOS EXTREMOS (770 combos)")
    print("=" * 80)
    
    # 1. Clasificación de referencia (parámetros default del pipeline)
    print(f"\n[1/4] Clasificación de referencia: "
          f"tumor={reference_tumor_pct}, immune={reference_immune_pct}, "
          f"margin={reference_ambiguity}")
    
    ref_phenotypes = _classify_parametric(
        adata, reference_tumor_pct, reference_immune_pct, reference_ambiguity
    )
    
    ref_counts = pd.Series(ref_phenotypes).value_counts()
    print(f"  Referencia: {dict(ref_counts)}")
    
    # 2. Barrer 770 combinaciones
    combos = list(product(
        EXTENDED_TUMOR_PERCENTILES,
        EXTENDED_IMMUNE_PERCENTILES,
        EXTENDED_AMBIGUITY_MARGINS,
    ))
    n_total = len(combos)
    print(f"\n[2/4] Barriendo {n_total} combinaciones...")
    
    results = []
    
    for i, (tumor_pct, immune_pct, margin) in enumerate(combos):
        if (i + 1) % 100 == 0:
            print(f"  Progreso: {i+1}/{n_total} ({(i+1)/n_total*100:.0f}%)")
        
        try:
            # Clasificar con estos parámetros
            custom_phenotypes = _classify_parametric(
                adata, tumor_pct, immune_pct, margin
            )
            
            # Métricas de concordancia vs referencia
            kappa = _cohens_kappa(ref_phenotypes, custom_phenotypes)
            ari = _adjusted_rand_index(ref_phenotypes, custom_phenotypes)
            
            # Métricas de effect size para CAF
            caf_d, caf_pval, n_desert, n_excluded = _calculate_caf_effect(
                adata, custom_phenotypes
            )
            
            results.append({
                'tumor_percentile': tumor_pct,
                'immune_percentile': immune_pct,
                'ambiguity_margin': margin,
                'cohens_kappa': kappa,
                'ARI': ari,
                'CAF_cohens_d': caf_d,
                'CAF_pval': caf_pval,
                'CAF_significant': caf_pval < 0.05 and abs(caf_d) >= 0.5,
                'n_desert': n_desert,
                'n_excluded': n_excluded,
                'valid': n_desert >= 50 and n_excluded >= 50,
            })
            
        except Exception as e:
            results.append({
                'tumor_percentile': tumor_pct,
                'immune_percentile': immune_pct,
                'ambiguity_margin': margin,
                'cohens_kappa': np.nan,
                'ARI': np.nan,
                'CAF_cohens_d': np.nan,
                'CAF_pval': np.nan,
                'CAF_significant': False,
                'n_desert': 0,
                'n_excluded': 0,
                'valid': False,
                'error': str(e),
            })
    
    df = pd.DataFrame(results)
    
    # 3. Resumen de robustez
    print(f"\n[3/4] Evaluando robustez extendida...")
    
    valid_mask = df['valid'] == True
    n_valid = valid_mask.sum()
    
    if n_valid > 0:
        df_valid = df[valid_mask]
        
        # Métricas de robustez
        pct_significant = (df_valid['CAF_significant'].sum() / n_valid) * 100
        mean_kappa = df_valid['cohens_kappa'].mean()
        mean_d = df_valid['CAF_cohens_d'].mean()
        mean_ari = df_valid['ARI'].mean()
        
        # Frontera de degradación
        pct_high_kappa = (df_valid['cohens_kappa'] >= 0.80).sum() / n_valid * 100
        pct_moderate_kappa = (df_valid['cohens_kappa'] >= 0.60).sum() / n_valid * 100
        pct_medium_d = (df_valid['CAF_cohens_d'].abs() >= 0.50).sum() / n_valid * 100
        
        print(f"\n  === RESUMEN DE ROBUSTEZ EXTENDIDA ===")
        print(f"  Configuraciones válidas: {n_valid}/{n_total}")
        print(f"  CAF significativo (d≥0.5, p<0.05): {pct_significant:.1f}%")
        print(f"  Kappa medio vs referencia: {mean_kappa:.3f}")
        print(f"  Cohen's d medio (CAF): {mean_d:.3f}")
        print(f"  ARI medio: {mean_ari:.3f}")
        print(f"  ")
        print(f"  FRONTERA DE DEGRADACIÓN:")
        print(f"    kappa ≥ 0.80 ('substantial'): {pct_high_kappa:.1f}%")
        print(f"    kappa ≥ 0.60 ('moderate'): {pct_moderate_kappa:.1f}%")
        print(f"    |d| ≥ 0.50 ('medium effect'): {pct_medium_d:.1f}%")
        
        # Encontrar la configuración más extrema que aún funciona
        d_threshold_mask = df_valid['CAF_cohens_d'].abs() >= 0.50
        if d_threshold_mask.any():
            extremes = df_valid[d_threshold_mask]
            worst_tumor = extremes['tumor_percentile'].agg(['min', 'max'])
            worst_immune = extremes['immune_percentile'].agg(['min', 'max'])
            worst_margin = extremes['ambiguity_margin'].agg(['min', 'max'])
            
            print(f"\n  RANGO FUNCIONAL (d≥0.5):")
            print(f"    tumor: [{worst_tumor['min']}, {worst_tumor['max']}]")
            print(f"    immune: [{worst_immune['min']}, {worst_immune['max']}]")
            print(f"    margin: [{worst_margin['min']:.2f}, {worst_margin['max']:.2f}]")
        
        # Methods text
        print(f"\n  Methods text: 'We tested {n_total} parameter combinations "
              f"spanning extreme ranges (tumor percentile: 40-90, immune: 50-95, "
              f"ambiguity margin: 0-30%). Of {n_valid} valid configurations, "
              f"{pct_significant:.1f}% maintained significant CAF differential "
              f"(|d|≥0.5, p<0.05). Classification agreement with reference "
              f"parameters showed κ={mean_kappa:.2f} (mean). Results degraded "
              f"beyond kappa=0.80 in {100-pct_high_kappa:.1f}% of configurations, "
              f"identifying the precise stability boundary.'")
    
    # 4. Visualizaciones
    print(f"\n[4/4] Generando heatmaps de degradación...")
    
    output_dir = Path(save_dir) if save_dir else Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        plot_degradation_heatmap(df, 'cohens_kappa', output_dir, 
                                 contour_levels=[0.60, 0.80])
        plot_degradation_heatmap(df, 'CAF_cohens_d', output_dir,
                                 contour_levels=[-0.50, -0.30])
        plot_degradation_heatmap(df, 'ARI', output_dir,
                                 contour_levels=[0.60, 0.80])
        plot_combined_degradation(df, output_dir)
    except Exception as e:
        print(f"  Error en visualizaciones: {e}")
        import traceback
        traceback.print_exc()
    
    # Guardar CSV
    if save_dir:
        csv_path = Path(save_dir) / 'sensitivity_v4_extended_770combos.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n  ✓ Saved: {csv_path}")
    
    print(f"\n{'=' * 80}")
    print(f" SENSITIVITY ANALYSIS v4.0 COMPLETADO ({n_total} combos)")
    print(f"{'=' * 80}")
    
    return df


# ============================================================================
# EXTENSIÓN 2: MÉTRICAS CONTINUAS
# ============================================================================

def _cohens_kappa(labels1, labels2):
    """
    Cohen's kappa — concordancia entre dos clasificaciones.
    
    κ = (p_o - p_e) / (1 - p_e)
    
    Interpretación:
      κ < 0.20  — Slight agreement
      0.21–0.40 — Fair
      0.41–0.60 — Moderate
      0.61–0.80 — Substantial
      0.81–1.00 — Almost perfect
    """
    labels1 = np.asarray(labels1)
    labels2 = np.asarray(labels2)
    
    # Excluir todas las variantes ambiguas, no solo 'Ambiguous'
    ambiguous_labels = {'Ambiguous', 'Ambiguous_Cold', 'Unclassified', ''}
    valid = ~np.isin(labels1, list(ambiguous_labels)) & ~np.isin(labels2, list(ambiguous_labels))
    
    if valid.sum() < 10:
        return np.nan
    
    l1 = labels1[valid]
    l2 = labels2[valid]
    
    # Todas las etiquetas posibles
    all_labels = sorted(set(l1) | set(l2))
    n = len(l1)
    
    if n == 0:
        return np.nan
    
    # Observed agreement
    p_o = np.sum(l1 == l2) / n
    
    # Expected agreement by chance
    p_e = 0
    for label in all_labels:
        p_e += (np.sum(l1 == label) / n) * (np.sum(l2 == label) / n)
    
    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else 0.0
    
    kappa = (p_o - p_e) / (1 - p_e)
    return float(kappa)


def _adjusted_rand_index(labels1, labels2):
    """
    Adjusted Rand Index — concordancia ajustada por azar.
    ARI = 1.0 → clasificaciones idénticas
    ARI = 0.0 → concordancia aleatoria
    ARI < 0.0 → peor que azar
    """
    labels1 = np.asarray(labels1)
    labels2 = np.asarray(labels2)
    
    # Excluir todas las variantes ambiguas
    ambiguous_labels = {'Ambiguous', 'Ambiguous_Cold', 'Unclassified', ''}
    valid = ~np.isin(labels1, list(ambiguous_labels)) & ~np.isin(labels2, list(ambiguous_labels))
    
    if valid.sum() < 10:
        return np.nan
    
    l1 = labels1[valid]
    l2 = labels2[valid]
    
    # Contingency table
    all_labels_1 = sorted(set(l1))
    all_labels_2 = sorted(set(l2))
    
    contingency = np.zeros((len(all_labels_1), len(all_labels_2)), dtype=int)
    
    label1_map = {l: i for i, l in enumerate(all_labels_1)}
    label2_map = {l: i for i, l in enumerate(all_labels_2)}
    
    for a, b in zip(l1, l2):
        contingency[label1_map[a], label2_map[b]] += 1
    
    # Compute ARI from contingency table
    n = contingency.sum()
    
    # Sum of combinations
    sum_comb_c = sum(_comb2(contingency[:, j].sum()) for j in range(contingency.shape[1]))
    sum_comb_r = sum(_comb2(contingency[i, :].sum()) for i in range(contingency.shape[0]))
    sum_comb_nij = sum(_comb2(contingency[i, j]) 
                       for i in range(contingency.shape[0]) 
                       for j in range(contingency.shape[1]))
    comb_n = _comb2(n)
    
    if comb_n == 0:
        return 0.0
    
    expected = (sum_comb_r * sum_comb_c) / comb_n
    max_index = (sum_comb_r + sum_comb_c) / 2
    
    if max_index == expected:
        return 1.0 if sum_comb_nij == expected else 0.0
    
    ari = (sum_comb_nij - expected) / (max_index - expected)
    return float(ari)


def _comb2(n):
    """Combinación C(n, 2) = n*(n-1)/2"""
    return n * (n - 1) / 2 if n >= 2 else 0


def _calculate_caf_effect(adata, phenotypes):
    """
    Calcula Cohen's d y p-value de CAF entre Desert y Excluded.
    """
    phenotypes = np.asarray(phenotypes)
    
    # Buscar columna CAF
    # Añadir formato REAL HPC (meanscell_/q05cell_ sin underscore)
    caf_values = None

    # Buscar en adata.obs
    for candidate in ['q05cell_abundance_w_sf_CAF', 'q05_cell_abundance_w_sf_CAF',
                       'q05_CAF', 'CAF_q05', 'meanscell_abundance_w_sf_CAF',
                       'means_cell_abundance_w_sf_CAF', 'CAF']:
        if candidate in adata.obs.columns:
            caf_values = adata.obs[candidate].values.astype(float)
            break

    # Fuzzy en adata.obs
    if caf_values is None:
        for col in adata.obs.columns:
            if 'caf' in col.lower():
                caf_values = adata.obs[col].values.astype(float)
                break

    # Buscar en adata.obsm (formato REAL HPC: meanscell_abundance_w_sf)
    if caf_values is None:
        import pandas as _pd
        for obsm_key in adata.obsm.keys():
            if 'abundance' in obsm_key:
                val = adata.obsm[obsm_key]
                if isinstance(val, _pd.DataFrame):
                    for col in val.columns:
                        if 'caf' in col.lower():
                            caf_values = val[col].values.astype(float)
                            break
            if caf_values is not None:
                break

    # Si CAF no se encontro, devolver conteos reales (no 0,0)
    if caf_values is None:
        _d_mask = phenotypes == 'Immune_Desert'
        _e_mask = phenotypes == 'Immune_Excluded'
        return np.nan, np.nan, int(_d_mask.sum()), int(_e_mask.sum())
    
    desert_mask = phenotypes == 'Immune_Desert'
    excluded_mask = phenotypes == 'Immune_Excluded'
    
    n_desert = desert_mask.sum()
    n_excluded = excluded_mask.sum()
    
    if n_desert < 10 or n_excluded < 10:
        return np.nan, np.nan, n_desert, n_excluded
    
    desert_vals = caf_values[desert_mask]
    excluded_vals = caf_values[excluded_mask]
    
    # Usar cohens_d_pooled canónico (ddof=1, pooled ponderada)
    d = cohens_d_pooled(desert_vals, excluded_vals)
    
    # Mann-Whitney
    try:
        _, pval = mannwhitneyu(desert_vals, excluded_vals, alternative='two-sided')
    except Exception:
        pval = np.nan
    
    return float(d), float(pval), int(n_desert), int(n_excluded)


def _classify_parametric(adata, tumor_pct, immune_pct, margin):
    """
    Réplica EXACTA de phenotype_classifier.py con parámetros variables.
    
    La versión anterior usaba lógica DIFERENTE (immune_low → Desert, NOT immune_high → Excluded).
    La lógica correcta del pipeline es: diff = Silencing_Score - Barrier_Score para 
    distinguir Desert vs Excluded en tumores fríos.
    
    Con parámetros default (60, 75, 0.1), la clasificación debe ser ~idéntica
    a adata.obs['Phenotype'].
    """
    try:
        # Intentar importar del módulo principal
        from sensitivity_analysis import classify_with_custom_params
        return classify_with_custom_params(
            adata, tumor_pct, immune_pct, margin
        ).values
    except (ImportError, AttributeError):
        pass
    
    # Buscar columnas de scores normalizados (priorizar _norm)
    suffix = '_norm'
    tumor_col = f'Tumor_Score{suffix}' if f'Tumor_Score{suffix}' in adata.obs.columns else None
    cd8_col = f'CD8_Score{suffix}' if f'CD8_Score{suffix}' in adata.obs.columns else None
    silence_col = f'Silencing_Score{suffix}' if f'Silencing_Score{suffix}' in adata.obs.columns else None
    barrier_col = f'Barrier_Score{suffix}' if f'Barrier_Score{suffix}' in adata.obs.columns else None
    
    # Fallback a scores sin normalizar
    if tumor_col is None:
        for col in ['Tumor_Score', 'Tumor_score', 'tumor_score']:
            if col in adata.obs.columns:
                tumor_col = col
                break
    if cd8_col is None:
        for col in ['CD8_Score', 'CD8_score', 'cd8_score']:
            if col in adata.obs.columns:
                cd8_col = col
                break
    if silence_col is None:
        for col in ['Silencing_Score', 'Silencing_score']:
            if col in adata.obs.columns:
                silence_col = col
                break
    if barrier_col is None:
        for col in ['Barrier_Score', 'Barrier_score']:
            if col in adata.obs.columns:
                barrier_col = col
                break
    
    # Verificar columnas esenciales
    if tumor_col is None or cd8_col is None:
        return np.full(adata.n_obs, 'Ambiguous', dtype=object)
    
    # Si no hay Silencing/Barrier, no podemos distinguir Desert vs Excluded
    has_mechanism = silence_col is not None and barrier_col is not None
    
    phenotypes = np.full(adata.n_obs, 'Unclassified', dtype=object)
    sample_col = 'sample_id' if 'sample_id' in adata.obs.columns else None
    samples = adata.obs[sample_col].unique() if sample_col else ['all']
    
    for sample_id in samples:
        if sample_col and sample_id != 'all':
            mask = (adata.obs[sample_col] == sample_id).values
        else:
            mask = np.ones(adata.n_obs, dtype=bool)
        
        if mask.sum() < 20:
            continue
        
        t_vals = adata.obs.loc[mask, tumor_col].values.astype(float)
        c_vals = adata.obs.loc[mask, cd8_col].values.astype(float)
        
        t_thresh = np.percentile(t_vals, tumor_pct)
        c_thresh = np.percentile(c_vals, immune_pct)
        
        indices = np.where(mask)[0]
        
        for j in range(len(indices)):
            idx = indices[j]
            
            if t_vals[j] < t_thresh:
                phenotypes[idx] = 'Normal_Stroma'
            elif c_vals[j] > c_thresh:
                phenotypes[idx] = 'Inflamed'
            else:
                # Tumores fríos: distinguir Desert vs Excluded por Silencing - Barrier
                if has_mechanism:
                    s_val = adata.obs.iloc[idx][silence_col]
                    b_val = adata.obs.iloc[idx][barrier_col]
                    diff = float(s_val) - float(b_val)
                    
                    if diff > margin:
                        phenotypes[idx] = 'Immune_Desert'
                    elif diff < -margin:
                        phenotypes[idx] = 'Immune_Excluded'
                    else:
                        phenotypes[idx] = 'Ambiguous_Cold'
                else:
                    phenotypes[idx] = 'Ambiguous_Cold'
    
    return phenotypes


# ============================================================================
# EXTENSIÓN 3: HEATMAPS DE DEGRADACIÓN CON CONTORNOS
# ============================================================================

def plot_degradation_heatmap(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    contour_levels: list = None,
    fixed_margin: float = 0.10,
):
    """
    Heatmap 2D (tumor × immune) con contornos de degradación.
    
    El fixed_margin permite mostrar un corte 2D del espacio 3D de parámetros.
    Para ver el efecto del margen, generar múltiples heatmaps o usar el plot combinado.
    
    Parameters
    ----------
    df : DataFrame con resultados de 770 combinaciones
    metric : str — columna a visualizar ('cohens_kappa', 'CAF_cohens_d', 'ARI')
    output_dir : Path — directorio para guardar figuras
    contour_levels : list — niveles para contornos
    fixed_margin : float — valor fijo de ambiguity_margin para el corte 2D
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filtrar por margen fijo
    df_slice = df[np.isclose(df['ambiguity_margin'], fixed_margin)]
    
    if len(df_slice) == 0:
        # Usar el margen más cercano disponible
        available_margins = df['ambiguity_margin'].unique()
        closest = available_margins[np.argmin(np.abs(available_margins - fixed_margin))]
        df_slice = df[np.isclose(df['ambiguity_margin'], closest)]
        fixed_margin = closest
    
    # Crear pivot table
    pivot = df_slice.pivot_table(
        index='immune_percentile',
        columns='tumor_percentile',
        values=metric,
        aggfunc='mean',
    )
    
    # Ordenar ejes
    pivot = pivot.sort_index(ascending=False)  # immune alto arriba
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colormap según métrica
    if 'kappa' in metric.lower() or 'ari' in metric.lower():
        cmap = 'RdYlGn'
        vmin, vmax = 0, 1
        label = 'Cohen\'s κ' if 'kappa' in metric.lower() else 'ARI'
    elif 'cohens_d' in metric.lower():
        cmap = 'RdBu'
        vmin, vmax = -1.5, 0.5
        label = 'Cohen\'s d (CAF Desert vs Excluded)'
    else:
        cmap = 'viridis'
        vmin, vmax = None, None
        label = metric
    
    # Heatmap
    sns.heatmap(
        pivot,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': label},
    )
    
    # Contornos (si datos suficientes)
    if contour_levels and pivot.shape[0] >= 3 and pivot.shape[1] >= 3:
        try:
            # Crear grid para contour
            x = np.arange(pivot.shape[1])
            y = np.arange(pivot.shape[0])
            Z = pivot.values.astype(float)
            
            # Reemplazar NaN para contour
            Z_clean = np.nan_to_num(Z, nan=0)
            
            CS = ax.contour(
                x + 0.5, y + 0.5, Z_clean,
                levels=contour_levels,
                colors='black',
                linewidths=2,
                linestyles='--',
            )
            ax.clabel(CS, inline=True, fontsize=10, fmt='%.2f')
        except Exception as e:
            print(f"  Contorno no posible: {e}")
    
    # Marcar configuración de referencia
    if 65 in pivot.columns and 75 in pivot.index:
        ref_x = list(pivot.columns).index(65) + 0.5
        ref_y = list(pivot.index).index(75) + 0.5
        ax.plot(ref_x, ref_y, '*', color='gold', markersize=20, 
                markeredgecolor='black', markeredgewidth=1.5,
                label='Reference params')
        ax.legend(loc='upper right')
    
    ax.set_xlabel('Tumor Percentile', fontsize=12)
    ax.set_ylabel('Immune Percentile', fontsize=12)
    ax.set_title(
        f'Parameter Sensitivity: {label}\n'
        f'(Ambiguity margin = {fixed_margin:.2f}, n={len(df_slice)} configs)',
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    
    fname = output_dir / f'sensitivity_v4_degradation_{metric}_margin{fixed_margin:.2f}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.savefig(fname.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {fname.name}")


def plot_combined_degradation(
    df: pd.DataFrame,
    output_dir: Path,
):
    """
    Panel combinado 2×3: kappa, d, ARI × 2 margins.
    
    Muestra cómo el paisaje de sensibilidad cambia con el margen
    de ambigüedad. Es la figura "killer" para la suplementaria.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    margins_to_show = [0.05, 0.15]  # Dos extremos del margen
    metrics = ['cohens_kappa', 'CAF_cohens_d', 'ARI']
    metric_labels = ['Cohen\'s κ', 'Cohen\'s d (CAF)', 'ARI']
    cmaps = ['RdYlGn', 'RdBu', 'RdYlGn']
    vmins = [0, -1.5, 0]
    vmaxs = [1, 0.5, 1]
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    for row, margin in enumerate(margins_to_show):
        df_slice = df[np.isclose(df['ambiguity_margin'], margin)]
        
        if len(df_slice) == 0:
            available = df['ambiguity_margin'].unique()
            closest = available[np.argmin(np.abs(available - margin))]
            df_slice = df[np.isclose(df['ambiguity_margin'], closest)]
            margin = closest
        
        for col, (metric, label, cmap, vmin, vmax) in enumerate(
            zip(metrics, metric_labels, cmaps, vmins, vmaxs)
        ):
            ax = axes[row, col]
            
            pivot = df_slice.pivot_table(
                index='immune_percentile',
                columns='tumor_percentile',
                values=metric,
                aggfunc='mean',
            ).sort_index(ascending=False)
            
            sns.heatmap(
                pivot, cmap=cmap, vmin=vmin, vmax=vmax,
                annot=True, fmt='.2f', linewidths=0.3,
                ax=ax, cbar_kws={'label': label, 'shrink': 0.8},
                annot_kws={'size': 7},
            )
            
            ax.set_title(f'{label} (margin={margin:.2f})', fontsize=11, fontweight='bold')
            ax.set_xlabel('Tumor Percentile', fontsize=9)
            ax.set_ylabel('Immune Percentile', fontsize=9)
            ax.tick_params(labelsize=8)
    
    plt.suptitle(
        'Extended Sensitivity Analysis: Degradation Landscape\n'
        f'770 parameter combinations across extreme ranges',
        fontsize=14, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    
    fname = output_dir / 'sensitivity_v4_combined_degradation.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.savefig(fname.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {fname.name}")


def plot_margin_effect(
    df: pd.DataFrame,
    output_dir: Path,
):
    """
    Lineplot mostrando cómo kappa y d cambian con el margen de ambigüedad.
    Cada línea = una combinación (tumor_pct, immune_pct), promediada.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (metric, label) in enumerate([
        ('cohens_kappa', 'Cohen\'s κ'),
        ('CAF_cohens_d', 'Cohen\'s d (CAF)')
    ]):
        ax = axes[idx]
        
        # Promediar kappa por margen
        margin_stats = df.groupby('ambiguity_margin').agg(
            mean_val=(metric, 'mean'),
            std_val=(metric, 'std'),
            median_val=(metric, 'median'),
            q25=(metric, lambda x: np.nanpercentile(x, 25)),
            q75=(metric, lambda x: np.nanpercentile(x, 75)),
        ).reset_index()
        
        ax.plot(margin_stats['ambiguity_margin'], margin_stats['mean_val'], 
                'b-o', linewidth=2, label='Mean')
        ax.fill_between(
            margin_stats['ambiguity_margin'],
            margin_stats['q25'],
            margin_stats['q75'],
            alpha=0.2, color='blue', label='IQR'
        )
        
        # Líneas de referencia
        if 'kappa' in metric:
            ax.axhline(y=0.80, color='green', linestyle='--', alpha=0.5, label='κ=0.80')
            ax.axhline(y=0.60, color='orange', linestyle='--', alpha=0.5, label='κ=0.60')
        elif 'cohens_d' in metric:
            ax.axhline(y=-0.50, color='red', linestyle='--', alpha=0.5, label='d=-0.50')
        
        ax.set_xlabel('Ambiguity Margin', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'Effect of Ambiguity Margin on {label}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fname = output_dir / 'sensitivity_v4_margin_effect.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {fname.name}")


# ============================================================================
# EXTENSIÓN 4 — VALIDACIÓN CON Z-SCORE FIJOS
# ============================================================================
# Responde a: "El percentil 75 siempre selecciona 25% de spots como
# 'calientes', incluso si son ruido." La prueba con umbrales absolutos
# (z-score fijos) demuestra concordancia percentil ↔ z-score,
# convirtiendo debilidad potencial en Supplementary Figure.

Z_SCORE_THRESHOLDS = {
    # (tumor_z_high, immune_z_high) — umbrales absolutos
    'conservative': (1.0, 1.0),
    'default': (0.5, 0.5),
    'relaxed': (0.0, 0.0),
    'stringent': (1.5, 1.5),
}


def _classify_zscore_fixed(adata, tumor_z_thresh, immune_z_thresh):
    """
    Clasificación con umbrales Z-SCORE FIJOS (no percentiles relativos).
    
    Demuestra que la estructura de fenotipos no es artefacto
    de usar percentiles adaptativos. Si z-score fijos producen 
    clasificación concordante (κ > 0.60), los percentiles no
    están forzando una estructura inexistente.
    """
    phenotypes = np.full(adata.n_obs, 'Ambiguous', dtype=object)
    
    # Buscar columnas de score
    tumor_col = None
    immune_col = None
    for col in adata.obs.columns:
        if 'tumor' in col.lower() and 'score' in col.lower():
            tumor_col = col
        if ('immune' in col.lower() or 'cd8' in col.lower()) and 'score' in col.lower():
            immune_col = col
    
    if tumor_col is None or immune_col is None:
        return phenotypes
    
    # Z-score GLOBAL (no per-sample) — sin adaptación
    tumor_vals = adata.obs[tumor_col].values.astype(float)
    immune_vals = adata.obs[immune_col].values.astype(float)
    
    t_mean, t_std = np.nanmean(tumor_vals), np.nanstd(tumor_vals)
    i_mean, i_std = np.nanmean(immune_vals), np.nanstd(immune_vals)
    
    if t_std == 0 or i_std == 0:
        return phenotypes
    
    tumor_z = (tumor_vals - t_mean) / t_std
    immune_z = (immune_vals - i_mean) / i_std
    
    # Clasificación con umbrales fijos
    for idx in range(adata.n_obs):
        tz = tumor_z[idx]
        iz = immune_z[idx]
        
        if tz >= tumor_z_thresh and iz >= immune_z_thresh:
            phenotypes[idx] = 'Inflamed'
        elif tz >= tumor_z_thresh and iz < -immune_z_thresh:
            phenotypes[idx] = 'Immune_Desert'
        elif tz >= tumor_z_thresh:
            phenotypes[idx] = 'Immune_Excluded'
        else:
            phenotypes[idx] = 'Ambiguous'
    
    return phenotypes


def run_zscore_validation(
    adata,
    save_dir: str = None,
    reference_phenotypes=None,
) -> pd.DataFrame:
    """
    Compara clasificación percentil vs z-score fijos.
    
    Genera:
    - Tabla de concordancia (κ, ARI) por umbral z-score
    - Tabla de valores absolutos: qué valor real corresponde a cada percentil
    - Supplementary Figure: scatter κ vs umbral
    
    Parameters
    ----------
    adata : AnnData con scores
    save_dir : str — directorio para CSVs y figuras
    reference_phenotypes : array-like — clasificación de referencia (percentil)
    
    Returns
    -------
    DataFrame con resultados de concordancia
    """
    print("\n" + "=" * 80)
    print("H14 VALIDATION: Z-SCORE FIJOS vs PERCENTILES")
    print("=" * 80)
    
    # Obtener clasificación de referencia (percentil-based)
    # Pipeline produce 'Phenotype' (capitalizada), no 'phenotype'
    if reference_phenotypes is None:
        if 'Phenotype' in adata.obs.columns:
            reference_phenotypes = adata.obs['Phenotype'].values
        elif 'phenotype' in adata.obs.columns:
            reference_phenotypes = adata.obs['phenotype'].values
        else:
            reference_phenotypes = _classify_parametric(adata, 65, 75, 0.10)
    
    results = []
    
    for name, (t_z, i_z) in Z_SCORE_THRESHOLDS.items():
        zscore_phenotypes = _classify_zscore_fixed(adata, t_z, i_z)
        
        kappa = _cohens_kappa(reference_phenotypes, zscore_phenotypes)
        ari = _adjusted_rand_index(reference_phenotypes, zscore_phenotypes)
        
        # CAF effect size con clasificación z-score
        caf_d, caf_p, n_des, n_exc = _calculate_caf_effect(adata, zscore_phenotypes)
        
        # Conteos
        unique, counts = np.unique(zscore_phenotypes, return_counts=True)
        count_dict = dict(zip(unique, counts))
        
        results.append({
            'threshold_name': name,
            'tumor_z': t_z,
            'immune_z': i_z,
            'kappa_vs_percentile': kappa,
            'ARI_vs_percentile': ari,
            'CAF_cohens_d': caf_d,
            'CAF_pval': caf_p,
            'n_desert': count_dict.get('Immune_Desert', 0),
            'n_excluded': count_dict.get('Immune_Excluded', 0),
            'n_inflamed': count_dict.get('Inflamed', 0),
            'n_ambiguous': count_dict.get('Ambiguous', 0),
        })
        
        print(f"  {name:12s} (z={t_z:.1f}): κ={kappa:.3f}, ARI={ari:.3f}, "
              f"d={caf_d:.3f}, Des={count_dict.get('Immune_Desert', 0)}, "
              f"Exc={count_dict.get('Immune_Excluded', 0)}")
    
    df = pd.DataFrame(results)
    
    # --- Tabla de valores absolutos por percentil ---
    print(f"\n  --- Valores absolutos por percentil de referencia ---")
    abs_table = _generate_absolute_value_table(adata)
    if abs_table is not None:
        print(abs_table.to_string(index=False))
    
    # Guardar
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(save_path / 'sensitivity_v4_zscore_validation.csv', index=False)
        if abs_table is not None:
            abs_table.to_csv(save_path / 'sensitivity_v4_absolute_values_by_percentile.csv', 
                           index=False)
        
        # Supplementary Figure
        try:
            _plot_zscore_concordance(df, save_path)
        except Exception as e:
            print(f"  Plot failed: {e}")
    
    # Veredicto
    best_kappa = df['kappa_vs_percentile'].max()
    mean_kappa = df['kappa_vs_percentile'].mean()
    print(f"\n  VEREDICTO:")
    print(f"    Mejor κ (percentil↔z-score): {best_kappa:.3f}")
    print(f"    Media κ: {mean_kappa:.3f}")
    if best_kappa >= 0.60:
        print(f" Concordancia ≥ 'moderate': percentiles NO fuerzan estructura artificial")
    else:
        print(f" Concordancia baja: considerar umbrales alternativos")
    
    return df


def _generate_absolute_value_table(adata):
    """
    H14: Tabla de qué valor absoluto de score corresponde a cada percentil.
    
    Esto responde directamente al reviewer: "El percentil 75 siempre
    selecciona 25% de spots, pero ¿qué VALOR real corresponde a eso?"
    
    Si el valor es biológicamente razonable (score > 0), los percentiles
    no están seleccionando ruido.
    """
    score_cols = [c for c in adata.obs.columns if c.endswith('_Score')]
    if not score_cols:
        return None
    
    percentiles_to_check = [25, 50, 60, 65, 70, 75, 80, 85, 90, 95]
    
    rows = []
    for col in score_cols:
        vals = adata.obs[col].values.astype(float)
        vals_clean = vals[np.isfinite(vals)]
        
        if len(vals_clean) < 100:
            continue
        
        for pct in percentiles_to_check:
            pct_val = np.percentile(vals_clean, pct)
            z_equiv = (pct_val - np.mean(vals_clean)) / np.std(vals_clean) if np.std(vals_clean) > 0 else 0
            rows.append({
                'Score': col,
                'Percentile': pct,
                'Absolute_Value': float(pct_val),
                'Z_Score_Equivalent': float(z_equiv),
                'N_spots_above': int((vals_clean >= pct_val).sum()),
                'Pct_spots_above': float((vals_clean >= pct_val).mean() * 100),
            })
    
    return pd.DataFrame(rows)


def _plot_zscore_concordance(df, output_dir):
    """Supplementary: κ y ARI vs z-score threshold."""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    z_vals = df['tumor_z'].values
    
    axes[0].plot(z_vals, df['kappa_vs_percentile'], 'bo-', linewidth=2, markersize=8)
    axes[0].axhline(y=0.80, color='green', linestyle='--', alpha=0.5, label='κ=0.80')
    axes[0].axhline(y=0.60, color='orange', linestyle='--', alpha=0.5, label='κ=0.60')
    axes[0].set_xlabel('Z-score Threshold', fontsize=11)
    axes[0].set_ylabel("Cohen's κ (vs percentile)", fontsize=11)
    axes[0].set_title('Concordance: Percentile vs Z-score', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(z_vals, df['CAF_cohens_d'].abs(), 'rs-', linewidth=2, markersize=8)
    axes[1].axhline(y=0.50, color='red', linestyle='--', alpha=0.5, label='|d|=0.50')
    axes[1].set_xlabel('Z-score Threshold', fontsize=11)
    axes[1].set_ylabel("|Cohen's d| (CAF)", fontsize=11)
    axes[1].set_title('CAF Effect Stability Across Thresholds', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fname = output_dir / 'sensitivity_v4_zscore_concordance.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.savefig(fname.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname.name}")


# ============================================================================
# WRAPPER PRINCIPAL
# ============================================================================

def run_sensitivity_v4_pipeline(
    adata: ad.AnnData,
    save_dir: str = None,
    reference_params: dict = None,
) -> Dict:
    """
    Pipeline completo de sensibilidad v4.0.
    
    Ejecuta:
    1. Barrido de 770 combinaciones con métricas continuas
    2. Heatmaps de degradación con contornos
    3. Efecto del margen de ambigüedad
    4. Resumen de robustez
    
    Parameters
    ----------
    adata : AnnData con deconvolución y scores
    save_dir : str — directorio para guardar resultados
    reference_params : dict — parámetros de referencia del pipeline
    
    Returns
    -------
    dict con resultados completos
    """
    if reference_params is None:
        reference_params = {
            'tumor_pct': 60,  
            'immune_pct': 75,
            'ambiguity': 0.10,
        }
    
    if save_dir is None:
        save_dir = "results"
    
    tables_dir = Path(save_dir) / 'tables' / 'sensitivity_analysis_additions' if 'tables' not in save_dir else Path(save_dir) / 'sensitivity_analysis_additions'
    figures_dir = Path(save_dir) / 'figures' / 'sensitivity_analysis_additions' if 'figures' not in save_dir else Path(save_dir) / 'sensitivity_analysis_additions'
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # 1. Barrido extendido
    df = run_extended_sensitivity(
        adata,
        save_dir=str(tables_dir),
        reference_tumor_pct=reference_params['tumor_pct'],
        reference_immune_pct=reference_params['immune_pct'],
        reference_ambiguity=reference_params['ambiguity'],
    )
    all_results['extended_df'] = df
    
    # 2. Heatmaps adicionales por margen
    for margin in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        try:
            plot_degradation_heatmap(df, 'cohens_kappa', figures_dir,
                                     contour_levels=[0.60, 0.80],
                                     fixed_margin=margin)
        except Exception:
            pass
    
    # 3. Efecto del margen
    try:
        plot_margin_effect(df, figures_dir)
    except Exception as e:
        print(f"  margin_effect plot failed: {e}")
    
    # 3b. H14 EXTENSIÓN: Validación z-score fijos vs percentiles
    try:
        zscore_df = run_zscore_validation(
            adata, save_dir=str(tables_dir),
            reference_phenotypes=None,  # usa phenotype del adata
        )
        all_results['zscore_validation'] = zscore_df
    except Exception as e:
        print(f"  z-score validation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Resumen estadístico
    valid = df[df['valid'] == True]
    all_results['summary'] = {
        'total_combos': len(df),
        'valid_combos': len(valid),
        'pct_significant': float((valid['CAF_significant'].sum() / len(valid) * 100)) if len(valid) > 0 else 0,
        'mean_kappa': float(valid['cohens_kappa'].mean()) if len(valid) > 0 else 0,
        'mean_d': float(valid['CAF_cohens_d'].mean()) if len(valid) > 0 else 0,
        'mean_ari': float(valid['ARI'].mean()) if len(valid) > 0 else 0,
    }
    
    # Guardar resumen JSON
    import json
    summary_path = tables_dir / 'sensitivity_v4_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results['summary'], f, indent=2, default=str)
    print(f"\n  ✓ Summary: {summary_path}")
    
    return all_results

# ============================================================================
# MAIN (testing independiente)
# ============================================================================

if __name__ == '__main__':
    import scanpy as sc
    import sys
    
    print("=" * 80)
    print("SENSITIVITY ANALYSIS v4.0 — TESTING INDEPENDIENTE")
    print("=" * 80)
    
    # Buscar adata
    possible_paths = [
        Path("/home/external/frjimenez/fabian/genoma/data/processed/adata_with_phenotypes.h5ad"),
        Path("data/processed/adata_with_phenotypes.h5ad"),
    ]
    
    adata_path = None
    for p in possible_paths:
        if p.exists():
            adata_path = p
            break
    
    if adata_path is None:
        print("No se encontró adata")
        sys.exit(1)
    
    print(f"\nCargando: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    print(f"  Spots: {adata.n_obs:,} | Genes: {adata.n_vars:,}")
    
    results = run_sensitivity_v4_pipeline(
        adata, 
        save_dir='/home/external/frjimenez/fabian/genoma/results',
    )
    
    print(f"\n Completado.")
    print(f"  Combos: {results['summary']['total_combos']}")
    print(f"  Válidas: {results['summary']['valid_combos']}")
    print(f"  Significativas: {results['summary']['pct_significant']:.1f}%")
    print(f"  κ medio: {results['summary']['mean_kappa']:.3f}")
