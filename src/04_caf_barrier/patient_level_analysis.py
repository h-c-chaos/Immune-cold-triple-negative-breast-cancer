"""
================================================================================
MODULO: PATIENT-LEVEL ANALYSIS — Responde a pseudoreplicación (A02/H7-01)
================================================================================

Este módulo agrega datos por N=43 y ejecuta tests a ese nivel.
El paper DEBE reportar AMBOS niveles (spot-level y patient-level).

Posiblemente se exigirá mixed-effects models o, como mínimo, análisis agregado
por paciente — este módulo provee lo segundo.

Métricas por paciente:
  - Mediana de CAF abundance por fenotipo
  - Mediana de cDC1 abundance por fenotipo
  - Ratio CXCL9:SPP1 mediano por fenotipo
  - Proporción de spots por fenotipo

Tests a nivel de paciente:
  - Wilcoxon signed-rank (pareado: pacientes con ambos fenotipos)
  - Mann-Whitney U (no pareado: todos los pacientes)
  - Cohen's d con CI bootstrapped

Output:
  - patient_level_summary.csv (agregados por paciente)
  - patient_level_tests.csv (resultados estadísticos)
  - patient_level_report.txt (resumen para Methods)
================================================================================
"""

import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.stats import wilcoxon, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Importar funciones canónicas
try:
    from config import PATHS
    RESULTS_DIR = PATHS.RESULTS_DIR
    PROCESSED_DIR = PATHS.PROCESSED_DIR
except ImportError:
    RESULTS_DIR = Path("results")
    PROCESSED_DIR = Path("data/processed")

try:
    from utils_stats import cohens_d_pooled, bootstrap_ci, apply_fdr
except ImportError:
    # Fallback mínimo
    def cohens_d_pooled(g1, g2):
        g1, g2 = np.asarray(g1, float), np.asarray(g2, float)
        g1, g2 = g1[np.isfinite(g1)], g2[np.isfinite(g2)]
        n1, n2 = len(g1), len(g2)
        if n1 < 2 or n2 < 2: return 0.0
        sp = np.sqrt(((n1-1)*np.var(g1,ddof=1)+(n2-1)*np.var(g2,ddof=1))/(n1+n2-2))
        return float((g1.mean()-g2.mean())/sp) if sp > 1e-10 else 0.0

    def bootstrap_ci(g1, g2, stat_func=None, n_boot=1000, ci=0.95, seed=42):
        if stat_func is None: stat_func = cohens_d_pooled
        obs = stat_func(g1, g2)
        rng = np.random.RandomState(seed)
        boot = [stat_func(rng.choice(g1, len(g1), True), rng.choice(g2, len(g2), True))
                for _ in range(n_boot)]
        lo = np.percentile(boot, 100*(1-ci)/2)
        hi = np.percentile(boot, 100*(1-(1-ci)/2))
        return obs, lo, hi

    def apply_fdr(pvals, method='fdr_bh', alpha=0.05):
        from statsmodels.stats.multitest import multipletests
        r, q, _, _ = multipletests(pvals, alpha=alpha, method=method)
        return r, q

try:
    from mechanism_validation import find_cell_abundance_column
except ImportError:
    find_cell_abundance_column = None


# ============================================================================
# CORE: AGREGAR POR PACIENTE
# ============================================================================

def aggregate_by_patient(
    adata,
    metric_col: str,
    phenotype_col: str = 'Phenotype',
    sample_col: str = 'sample_id',
    agg_func: str = 'median',
) -> pd.DataFrame:
    """
    Agrega una métrica por paciente y fenotipo.

    Para cada paciente, calcula la mediana (o media) de metric_col
    en los spots de cada fenotipo.

    Parameters
    ----------
    adata : AnnData
        Datos con fenotipos y métrica en .obs
    metric_col : str
        Columna a agregar (ej. 'CAF_q05', 'CXCL9_SPP1_log2ratio')
    phenotype_col : str
        Columna de fenotipos
    sample_col : str
        Columna de ID de paciente/muestra
    agg_func : str
        'median' o 'mean'

    Returns
    -------
    DataFrame con columnas: sample_id, phenotype, n_spots, agg_value
    """
    if metric_col not in adata.obs.columns:
        print(f"[WARN] Columna '{metric_col}' no encontrada en adata.obs")
        return pd.DataFrame()

    if phenotype_col not in adata.obs.columns:
        print(f"[WARN] Columna '{phenotype_col}' no encontrada")
        return pd.DataFrame()

    func = np.median if agg_func == 'median' else np.mean

    rows = []
    for sample_id in adata.obs[sample_col].unique():
        sample_mask = adata.obs[sample_col] == sample_id

        for pheno in ['Immune_Desert', 'Immune_Excluded', 'Inflamed']:
            pheno_mask = sample_mask & (adata.obs[phenotype_col] == pheno)
            n_spots = pheno_mask.sum()

            if n_spots > 0:
                vals = adata.obs.loc[pheno_mask, metric_col].values.astype(float)
                vals = vals[np.isfinite(vals)]

                if len(vals) > 0:
                    rows.append({
                        'sample_id': sample_id,
                        'phenotype': pheno,
                        'n_spots': int(n_spots),
                        'agg_value': float(func(vals)),
                        'metric': metric_col,
                    })

    return pd.DataFrame(rows)


def patient_level_test(
    patient_df: pd.DataFrame,
    group1: str = 'Immune_Desert',
    group2: str = 'Immune_Excluded',
    metric_name: str = '',
    min_patients: int = 5,
) -> Dict:
    """
    Tests a nivel de paciente — solo pacientes con AMBOS fenotipos.

    Reporta:
    - Wilcoxon signed-rank (pareado): pacientes que tienen ambos fenotipos
    - Mann-Whitney U (no pareado): para comparar
    - Cohen's d con bootstrap CI 95%
    """
    if patient_df.empty:
        return {'error': 'DataFrame vacío', 'metric': metric_name}

    # Pacientes con ambos fenotipos
    patients_g1 = set(patient_df[patient_df['phenotype'] == group1]['sample_id'])
    patients_g2 = set(patient_df[patient_df['phenotype'] == group2]['sample_id'])
    common = sorted(patients_g1 & patients_g2)

    result = {
        'metric': metric_name,
        'group1': group1,
        'group2': group2,
        'n_patients_g1': len(patients_g1),
        'n_patients_g2': len(patients_g2),
        'n_patients_paired': len(common),
    }

    if len(common) < min_patients:
        result['error'] = f'Solo {len(common)} pacientes pareados (mínimo {min_patients})'
        return result

    # Extraer valores pareados
    vals_g1 = []
    vals_g2 = []
    for pid in common:
        v1 = patient_df[(patient_df['sample_id'] == pid) &
                        (patient_df['phenotype'] == group1)]['agg_value'].values[0]
        v2 = patient_df[(patient_df['sample_id'] == pid) &
                        (patient_df['phenotype'] == group2)]['agg_value'].values[0]
        vals_g1.append(v1)
        vals_g2.append(v2)

    vals_g1 = np.array(vals_g1, dtype=float)
    vals_g2 = np.array(vals_g2, dtype=float)

    # Medianas
    result['g1_median'] = float(np.median(vals_g1))
    result['g2_median'] = float(np.median(vals_g2))
    result['g1_mean'] = float(np.mean(vals_g1))
    result['g2_mean'] = float(np.mean(vals_g2))

    # Cohen's d canónico
    d = cohens_d_pooled(vals_g1, vals_g2)
    result['cohens_d'] = d

    # Bootstrap CI para Cohen's d
    try:
        _, ci_lo, ci_hi = bootstrap_ci(vals_g1, vals_g2, n_boot=2000, seed=42)
        result['d_ci_lower'] = ci_lo
        result['d_ci_upper'] = ci_hi
    except Exception:
        result['d_ci_lower'] = np.nan
        result['d_ci_upper'] = np.nan

    # Wilcoxon signed-rank (pareado)
    try:
        # Necesita que las diferencias no sean todas cero
        diffs = vals_g1 - vals_g2
        if np.all(diffs == 0):
            result['wilcoxon_stat'] = np.nan
            result['wilcoxon_p'] = 1.0
        else:
            stat_w, pval_w = wilcoxon(vals_g1, vals_g2)
            result['wilcoxon_stat'] = float(stat_w)
            result['wilcoxon_p'] = float(pval_w)
    except Exception as e:
        result['wilcoxon_stat'] = np.nan
        result['wilcoxon_p'] = np.nan
        result['wilcoxon_error'] = str(e)

    # Mann-Whitney U (no pareado, para comparar)
    try:
        stat_mw, pval_mw = mannwhitneyu(vals_g1, vals_g2, alternative='two-sided')
        result['mannwhitney_stat'] = float(stat_mw)
        result['mannwhitney_p'] = float(pval_mw)
    except Exception:
        result['mannwhitney_stat'] = np.nan
        result['mannwhitney_p'] = np.nan

    return result


# ============================================================================
# BÚSQUEDA DE COLUMNAS DE ABUNDANCIA
# ============================================================================

def _find_abundance_col(adata, cell_type: str) -> Optional[str]:
    """Busca columna de abundancia celular en adata.obs."""
    if find_cell_abundance_column is not None:
        col = find_cell_abundance_column(adata, cell_type, 'q05')
        if col and not col.startswith('obsm:'):
            return col

    # Fallback: búsqueda manual
    # FIX FASE2-02: Añadir formato REAL HPC (meanscell_/q05cell_ sin underscore)
    patterns = [
        f'q05cell_abundance_w_sf_{cell_type}',      # Formato REAL HPC
        f'q05_cell_abundance_w_sf_{cell_type}',
        f'q05_{cell_type}',
        f'{cell_type}_q05',
        f'meanscell_abundance_w_sf_{cell_type}',     # Formato REAL HPC
        f'means_cell_abundance_w_sf_{cell_type}',
        f'{cell_type}_means',
    ]
    for p in patterns:
        if p in adata.obs.columns:
            return p

    # Fuzzy
    for col in adata.obs.columns:
        if cell_type.lower() in col.lower() and ('abundance' in col.lower() or 'q05' in col.lower()):
            return col

    return None


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def run_patient_level_analysis(
    adata,
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Pipeline completo de análisis a nivel de paciente.

    Agrega por paciente y ejecuta tests para:
    1. CAF abundance (hallazgo principal: d ~ -0.6)
    2. cDC1 abundance
    3. CD8_T abundance
    4. Ratio CXCL9:SPP1 (si disponible)

    Returns
    -------
    dict con DataFrames de resumen y tests
    """
    print("\n" + "=" * 80)
    print("PATIENT-LEVEL ANALYSIS (N=pacientes, no N=spots)")
    print("Responde a pseudoreplicación — Reviewer 2 pregunta #4")
    print("=" * 80)

    if 'sample_id' not in adata.obs.columns:
        print("[ERROR] No hay columna 'sample_id' — no se puede agregar por paciente")
        return {'error': 'No sample_id column'}

    if 'Phenotype' not in adata.obs.columns:
        print("[ERROR] No hay columna 'Phenotype'")
        return {'error': 'No Phenotype column'}

    n_patients = adata.obs['sample_id'].nunique()
    print(f"\n  Pacientes totales: {n_patients}")
    print(f"  Spots totales: {adata.n_obs:,}")

    # Identificar métricas disponibles
    metrics_to_test = {}

    # 1. CAF abundance
    caf_col = _find_abundance_col(adata, 'CAF')
    if caf_col:
        metrics_to_test['CAF_abundance'] = caf_col
        print(f"  ✓ CAF: {caf_col}")

    # 2. cDC1 abundance
    cdc1_col = _find_abundance_col(adata, 'cDC1')
    if cdc1_col is None:
        cdc1_col = _find_abundance_col(adata, 'DC')
    if cdc1_col:
        metrics_to_test['cDC1_abundance'] = cdc1_col
        print(f"  ✓ cDC1: {cdc1_col}")

    # 3. CD8_T abundance
    cd8_col = _find_abundance_col(adata, 'CD8_T')
    if cd8_col:
        metrics_to_test['CD8_T_abundance'] = cd8_col
        print(f"  ✓ CD8_T: {cd8_col}")

    # 4. Macrophage abundance
    macro_col = _find_abundance_col(adata, 'Macrophage')
    if macro_col:
        metrics_to_test['Macrophage_abundance'] = macro_col
        print(f"  ✓ Macrophage: {macro_col}")

    # 5. CXCL9:SPP1 ratio (si mechanism_validation lo calculó)
    if 'CXCL9_SPP1_log2ratio' in adata.obs.columns:
        metrics_to_test['CXCL9_SPP1_ratio'] = 'CXCL9_SPP1_log2ratio'
        print(f"  ✓ CXCL9:SPP1 ratio")

    if not metrics_to_test:
        print("[ERROR] No se encontraron métricas para agregar")
        return {'error': 'No metrics found'}

    # Agregar por paciente
    print(f"\n  Agregando {len(metrics_to_test)} métricas por paciente...")

    all_aggregated = []
    all_tests = []

    for metric_name, col_name in metrics_to_test.items():
        print(f"\n  --- {metric_name} ({col_name}) ---")

        patient_df = aggregate_by_patient(adata, col_name)

        if patient_df.empty:
            print(f"    [SKIP] Sin datos para agregar")
            continue

        patient_df['metric_name'] = metric_name
        all_aggregated.append(patient_df)

        # Contar pacientes por fenotipo
        for pheno in ['Immune_Desert', 'Immune_Excluded', 'Inflamed']:
            n = patient_df[patient_df['phenotype'] == pheno]['sample_id'].nunique()
            print(f"    {pheno}: {n} pacientes")

        # Test Desert vs Excluded
        test_result = patient_level_test(
            patient_df,
            group1='Immune_Desert',
            group2='Immune_Excluded',
            metric_name=metric_name,
        )
        all_tests.append(test_result)

        # Reportar resultado
        if 'error' not in test_result:
            d = test_result['cohens_d']
            p_w = test_result.get('wilcoxon_p', np.nan)
            p_mw = test_result.get('mannwhitney_p', np.nan)
            n_paired = test_result['n_patients_paired']
            ci_lo = test_result.get('d_ci_lower', np.nan)
            ci_hi = test_result.get('d_ci_upper', np.nan)

            print(f"    Cohen's d = {d:.3f} [95% CI: {ci_lo:.3f}, {ci_hi:.3f}]")
            print(f"    Wilcoxon (pareado, n={n_paired}): p = {p_w:.4f}")
            print(f"    Mann-Whitney (no pareado): p = {p_mw:.4f}")

            if not np.isnan(p_w):
                if p_w < 0.05:
                    print(f"    SIGNIFICATIVO a nivel de paciente")
                else:
                    print(f"    NO significativo a nivel de paciente (esperado con n={n_paired})")
                    if abs(d) >= 0.5:
                        print(f"      → Pero effect size |d|={abs(d):.2f} es medio/grande")
                        print(f"      → Limitación de poder estadístico con n={n_paired}")
        else:
            print(f"    [SKIP] {test_result['error']}")

    # Compilar resultados
    df_aggregated = pd.concat(all_aggregated, ignore_index=True) if all_aggregated else pd.DataFrame()
    df_tests = pd.DataFrame(all_tests)

    # FDR en los tests a nivel de paciente
    if len(df_tests) > 1 and 'wilcoxon_p' in df_tests.columns:
        pvals = df_tests['wilcoxon_p'].values
        valid_pvals = ~np.isnan(pvals)
        if valid_pvals.sum() > 1:
            try:
                reject, qvals = apply_fdr(pvals[valid_pvals])
                df_tests.loc[valid_pvals, 'wilcoxon_q'] = qvals
                df_tests.loc[valid_pvals, 'wilcoxon_fdr_significant'] = reject
                print(f"\n  FDR aplicado a {valid_pvals.sum()} tests")
            except Exception:
                pass

    # Resumen
    print(f"\n{'='*80}")
    print("RESUMEN PATIENT-LEVEL ANALYSIS")
    print(f"{'='*80}")

    if not df_tests.empty:
        print(f"\n  {'Métrica':<25} {'d':>8} {'p_Wilcoxon':>12} {'n_pareado':>10}")
        print(f"  {'-'*55}")
        for _, row in df_tests.iterrows():
            d = row.get('cohens_d', np.nan)
            p = row.get('wilcoxon_p', np.nan)
            n = row.get('n_patients_paired', 0)
            m = row.get('metric', '?')
            sig = '✓' if (not np.isnan(p) and p < 0.05) else '⚠'
            print(f"  {m:<25} {d:>8.3f} {p:>12.4f} {n:>10} {sig}")

    # Texto para Methods
    if not df_tests.empty and 'CAF_abundance' in df_tests['metric'].values:
        caf_row = df_tests[df_tests['metric'] == 'CAF_abundance'].iloc[0]
        d_caf = caf_row.get('cohens_d', np.nan)
        p_caf = caf_row.get('wilcoxon_p', np.nan)
        n_caf = caf_row.get('n_patients_paired', 0)
        ci_lo = caf_row.get('d_ci_lower', np.nan)
        ci_hi = caf_row.get('d_ci_upper', np.nan)

    # Guardar
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        if not df_aggregated.empty:
            df_aggregated.to_csv(save_path / 'patient_level_summary.csv', index=False)
            print(f"\n  ✓ Saved: {save_path / 'patient_level_summary.csv'}")

        if not df_tests.empty:
            df_tests.to_csv(save_path / 'patient_level_tests.csv', index=False)
            print(f"  ✓ Saved: {save_path / 'patient_level_tests.csv'}")

        # Report text
        with open(save_path / 'patient_level_report.txt', 'w') as f:
            f.write("PATIENT-LEVEL ANALYSIS REPORT\n")
            f.write(f"Generated by patient_level_analysis.py\n")
            f.write(f"Total patients: {n_patients}\n")
            f.write(f"Total spots: {adata.n_obs}\n\n")
            if not df_tests.empty:
                f.write(df_tests.to_string(index=False))
        print(f"  ✓ Saved: {save_path / 'patient_level_report.txt'}")

    results = {
        'aggregated': df_aggregated,
        'tests': df_tests,
        'n_patients': n_patients,
    }

    print(f"\n Patient-level analysis completado.")
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("PATIENT-LEVEL ANALYSIS — Standalone Execution")
    print("=" * 80)

    # Buscar adata
    possible_paths = [
        PROCESSED_DIR / 'adata_with_mechanism.h5ad',
        PROCESSED_DIR / 'adata_with_phenotypes.h5ad',
        PROCESSED_DIR / 'adata_with_deconvolution.h5ad',
    ]

    adata = None
    for p in possible_paths:
        if p.exists():
            print(f"Cargando: {p}")
            adata = sc.read_h5ad(p)
            print(f"  Spots: {adata.n_obs:,} | Genes: {adata.n_vars:,}")
            break

    if adata is None:
        print("ERROR: No se encontró archivo de datos.")
        print("Rutas buscadas:")
        for p in possible_paths:
            print(f"  {p}")
        import sys
        sys.exit(1)

    # Ejecutar
    save_dir = RESULTS_DIR / "patient_level"
    results = run_patient_level_analysis(adata, save_dir=str(save_dir))

    print(f"\n Completado. Resultados en: {save_dir}")
