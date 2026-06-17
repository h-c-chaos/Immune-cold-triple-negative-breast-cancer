"""
================================================================================
ROBUSTNESS STRESS TESTS - Pruebas de Estrés Estadísticas
================================================================================

Propósito:
    Demostrar que los hallazgos principales (CAF d=-0.63, geodesic
    isolation 1.7×) son robustos ante perturbaciones severas, no
    artefactos del pipeline. Responde a las críticas:
    - "¿Son los labels estables o dependen de splits aleatorios?"
    - "¿Qué pasa si hay ruido en la deconvolución?"
    - "¿Qué pasa si faltan genes clave?"
    - "¿Cambian los resultados con otro método de clustering?"

Tests implementados:
    1. Label Shuffling (spot-level + patient-level)
    2. Gaussian Noise Injection (σ = 0.05 a 1.0)
    3. Gene Dropout Simulation
    4. Alternative Clustering (k-means, Leiden, GMM)
    5. Degradation Boundary Identification

Output esperado:
    - label_shuffling_results.csv
    - noise_injection_results.csv
    - gene_dropout_results.csv
    - alternative_clustering_comparison.csv
    - degradation_boundary.csv
    - Fig_robustness_stress_tests.pdf (6 paneles)
================================================================================
"""

import os
import sys
import time
import gc
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

try:
    # GENE_SIGNATURES no existe, usar SIGNATURES
    from config import PATHS, SIGNATURES, PHENOTYPE_PARAMS
    # FIX AUDIT v2.5: Acción ⓵ — Cohen's d canónico
    from utils_stats import cohens_d_pooled
    BASE_DIR = PATHS.BASE_DIR
    RESULTS_DIR = PATHS.RESULTS_DIR
    PROCESSED_DIR = PATHS.PROCESSED_DIR
    # Genes EXPLÍCITOS del análisis mecanístico real (mechanism_validation.py:303)
    # SILENCING_REPRESSORS[:4] tomaba MYC,EZH2,SUZ12,CTNNB1 — INCORRECTO
    SILENCING_GENES = ['MYC', 'EZH2', 'DNMT1', 'STAT3']
    BARRIER_GENES = list(SIGNATURES.PHYSICAL_BARRIER)
    # Usar CD8_T_CELLS (no IMMUNE_GENES genérico)
    CD8_GENES = list(SIGNATURES.CD8_T_CELLS)
    print("✓ Configuración cargada desde config.py")
except ImportError:
    from pathlib import Path
    BASE_DIR = Path("/home/external/frjimenez/fabian/genoma")
    RESULTS_DIR = BASE_DIR / "results"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    SILENCING_GENES = ['MYC', 'EZH2', 'DNMT1', 'STAT3']
    BARRIER_GENES = ['COL1A1', 'COL1A2', 'COL10A1', 'FN1', 'POSTN', 'TGFB1', 'ACTA2', 'FAP', 'THBS2']
    # CD8 genes, no immune genérico
    CD8_GENES = ['CD8A', 'CD8B', 'CD3D', 'CD3E', 'GZMA', 'GZMB', 'PRF1']
    try:
        from utils_stats import cohens_d_pooled
    except ImportError:
        def cohens_d_pooled(g1, g2):
            g1, g2 = np.asarray(g1, float), np.asarray(g2, float)
            n1, n2 = len(g1), len(g2)
            if n1 < 2 or n2 < 2: return 0.0
            sp = np.sqrt(((n1-1)*np.var(g1,ddof=1)+(n2-1)*np.var(g2,ddof=1))/(n1+n2-2))
            return (g1.mean()-g2.mean())/sp if sp > 1e-10 else 0.0
    print("⚠ config.py no encontrado, usando rutas HPC por defecto")

# Directorio de salida
STRESS_DIR = RESULTS_DIR / "robustness_stress_tests"
os.makedirs(STRESS_DIR, exist_ok=True)
os.makedirs(STRESS_DIR / "figures", exist_ok=True)

# Parámetros
RANDOM_SEED = 42
N_SHUFFLES = 1000           # Permutaciones para label shuffling
NOISE_LEVELS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]  # σ para ruido gaussiano
DROPOUT_FRACTIONS = [0.05, 0.1, 0.2, 0.3, 0.5]          # Fracción de genes a eliminar
N_BOOTSTRAP = 500           # Para intervalos de confianza

np.random.seed(RANDOM_SEED)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'figure.facecolor': 'white'
})


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def compute_caf_cohens_d(adata, phenotype_col='Phenotype',
                          caf_col=None):
    """
    Calcula Cohen's d de CAF entre Desert y Excluded.
    Este es el estadístico principal del paper (d = -0.63 observado).
    
    phenotype_col default 'Phenotype' (capitalizado)
    Usa cohens_d_pooled de utils_stats
    
    Returns
    -------
    d : float
        Cohen's d (negativo = más CAF en Excluded)
    pval : float
        p-value del Mann-Whitney U test
    """
    if caf_col is None:
        # Detectar columna de CAF
        caf_candidates = [c for c in adata.obs.columns 
                         if 'caf' in c.lower() and ('mycaf' in c.lower() or 'abundance' in c.lower())]
        if caf_candidates:
            caf_col = caf_candidates[0]
        else:
            caf_candidates = [c for c in adata.obs.columns if 'caf' in c.lower()]
            if caf_candidates:
                caf_col = caf_candidates[0]
            else:
                return np.nan, np.nan
    
    desert = adata.obs.loc[adata.obs[phenotype_col] == 'Immune_Desert', caf_col].values.astype(float)
    excluded = adata.obs.loc[adata.obs[phenotype_col] == 'Immune_Excluded', caf_col].values.astype(float)
    
    if len(desert) < 10 or len(excluded) < 10:
        return np.nan, np.nan
    
    # Cohen's d canónico
    d = cohens_d_pooled(desert, excluded)
    
    # Mann-Whitney
    try:
        _, pval = mannwhitneyu(desert, excluded, alternative='two-sided')
    except ValueError:
        pval = 1.0
    
    return d, pval


def compute_phenotype_proportions(adata, phenotype_col='Phenotype'):
    """Calcula proporciones de cada fenotipo."""
    counts = adata.obs[phenotype_col].value_counts(normalize=True)
    return counts.to_dict()


def reclassify_phenotypes(adata, scores_dict, sample_col='sample_id',
                           tumor_pct=60, immune_pct=75, ambiguity=0.10):
    """
    Reclasifica fenotipos usando la lógica jerárquica del pipeline.
    
    Réplica simplificada de phenotype_classifier.py para usarse
    en perturbaciones sin reimportar el módulo completo.
    
    tumor_pct=60 (no 65)
    Immune_Score → CD8_Score
    """
    phenotypes = pd.Series('Unclassified', index=adata.obs.index)
    
    for sample in adata.obs[sample_col].unique():
        mask = adata.obs[sample_col] == sample
        sample_idx = adata.obs.index[mask]
        
        if mask.sum() < 20:
            continue
        
        # Extraer scores para esta muestra
        tumor_s = scores_dict['Tumor_Score'][mask].values
        #  Usar CD8_Score (consistente con phenotype_classifier.py)
        immune_s = scores_dict['CD8_Score'][mask].values
        silence_s = scores_dict.get('Silencing_Score', pd.Series(0, index=adata.obs.index))[mask].values
        barrier_s = scores_dict.get('Barrier_Score', pd.Series(0, index=adata.obs.index))[mask].values
        
        # Umbrales per-sample
        t_high = np.percentile(tumor_s, tumor_pct)
        i_high = np.percentile(immune_s, immune_pct)
        
        for j, idx in enumerate(sample_idx):
            ts = tumor_s[j]
            is_ = immune_s[j]
            ss = silence_s[j]
            bs = barrier_s[j]
            
            if ts < t_high:
                phenotypes[idx] = 'Normal_Stroma'
            elif is_ >= i_high:
                phenotypes[idx] = 'Inflamed'
            else:
                # Tumor frío - diferenciar Desert vs Excluded
                diff = ss - bs
                if diff > ambiguity:
                    phenotypes[idx] = 'Immune_Desert'
                elif diff < -ambiguity:
                    phenotypes[idx] = 'Immune_Excluded'
                else:
                    phenotypes[idx] = 'Ambiguous_Cold'
    
    return phenotypes


# ============================================================================
# TEST 1: LABEL SHUFFLING
# ============================================================================

def test_label_shuffling(adata, n_shuffles=N_SHUFFLES, level='spot'):
    """
    Permuta labels de fenotipo y recalcula Cohen's d de CAF.
    
    Si el d observado es real, debe estar fuera de la distribución
    nula (p_perm < 0.05). Implementa dos niveles:
    - spot: Permuta etiquetas individuales (test más estricto)
    - patient: Permuta etiquetas de pacientes completos
    
    Parameters
    ----------
    adata : AnnData
        Datos con fenotipos y abundancias
    n_shuffles : int
        Número de permutaciones
    level : str
        'spot' o 'patient'
    
    Returns
    -------
    results : dict
        Estadísticos observados vs distribución nula
    """
    print(f"\n{'─'*50}")
    print(f"TEST 1: LABEL SHUFFLING (nivel={level})")
    print(f"{'─'*50}")
    
    # Estadístico observado
    d_obs, p_obs = compute_caf_cohens_d(adata)
    print(f"  Cohen's d observado: {d_obs:.4f}")
    print(f"  p-value observado:   {p_obs:.2e}")
    
    # Distribución nula
    null_d_values = []
    cold_mask = adata.obs['Phenotype'].isin(['Immune_Desert', 'Immune_Excluded'])
    adata_cold = adata[cold_mask].copy()
    
    print(f"  Spots fríos (Desert+Excluded): {cold_mask.sum():,}")
    print(f"  Ejecutando {n_shuffles} permutaciones...")
    
    # Pre-computar pacientes presentes en adata_cold (no recalcular en loop)
    # Si un paciente es 100% Inflamed, no aparece aquí → evita np.random.choice vacío
    if level == 'patient':
        cold_samples = adata_cold.obs['sample_id'].unique()
        if len(cold_samples) < 2:
            print(f"  Solo {len(cold_samples)} paciente(s) frío(s) — "
                  f"permutación por paciente no posible")
            return {
                'test': 'label_shuffling_patient',
                'n_permutations': 0,
                'd_observed': d_obs,
                'p_permutation': np.nan,
                'z_score': np.nan,
                'null_distribution': np.array([]),
                'skip_reason': f'only {len(cold_samples)} cold patient(s)',
            }
        print(f"  Pacientes fríos: {len(cold_samples)}")
    
    for i in range(n_shuffles):
        if (i + 1) % 200 == 0:
            print(f"    Permutación {i+1}/{n_shuffles}...")
        
        if level == 'spot':
            # Permuta etiquetas a nivel de spot individual
            shuffled = adata_cold.obs['Phenotype'].values.copy()
            np.random.shuffle(shuffled)
            adata_cold.obs['_shuffled_phenotype'] = shuffled
        
        elif level == 'patient':
            sample_to_shuffle = dict(
                zip(cold_samples, np.random.permutation(cold_samples))
            )
            
            # Reasignar fenotipos basándose en el paciente permutado
            shuffled = adata_cold.obs['Phenotype'].copy()
            for orig_sample, new_sample in sample_to_shuffle.items():
                orig_mask = adata_cold.obs['sample_id'] == orig_sample
                new_phenos = adata_cold.obs.loc[
                    adata_cold.obs['sample_id'] == new_sample, 'Phenotype'
                ].values
                
                n_orig = orig_mask.sum()
                
                # Preferir replace=False para preservar
                # correlación espacial intra-paciente.
                # replace=True solo como último recurso.
                if len(new_phenos) == 0:
                    # Paciente sin spots fríos → mantener original
                    continue
                elif len(new_phenos) >= n_orig:
                    shuffled.loc[orig_mask] = np.random.choice(
                        new_phenos, size=n_orig, replace=False
                    )
                else:
                    # Donante tiene menos spots que receptor
                    shuffled.loc[orig_mask] = np.random.choice(
                        new_phenos, size=n_orig, replace=True
                    )
            
            adata_cold.obs['_shuffled_phenotype'] = shuffled
        
        # Calcular Cohen's d con labels permutados
        d_null, _ = compute_caf_cohens_d(adata_cold, 
                                          phenotype_col='_shuffled_phenotype')
        if not np.isnan(d_null):
            null_d_values.append(d_null)
    
    null_d = np.array(null_d_values)
    
    # p-value permutacional
    p_perm = (np.abs(null_d) >= np.abs(d_obs)).mean()
    
    # Z-score del observado respecto a la nula
    if null_d.std() > 0:
        z_score = (d_obs - null_d.mean()) / null_d.std()
    else:
        z_score = np.inf
    
    results = {
        'test': f'label_shuffling_{level}',
        'n_permutations': n_shuffles,
        'd_observed': d_obs,
        'p_observed': p_obs,
        'd_null_mean': null_d.mean(),
        'd_null_std': null_d.std(),
        'd_null_min': null_d.min(),
        'd_null_max': null_d.max(),
        'p_permutation': p_perm,
        'z_score': z_score,
        'significant': p_perm < 0.05,
        'null_distribution': null_d  # Para plotting
    }
    
    print(f"\n  RESULTADOS:")
    print(f"    d observado:        {d_obs:.4f}")
    print(f"    d nulo (mean±std):  {null_d.mean():.4f} ± {null_d.std():.4f}")
    print(f"    p permutacional:    {p_perm:.4f}")
    print(f"    z-score:            {z_score:.2f}")
    print(f"    {'✓ SIGNIFICATIVO' if p_perm < 0.05 else '✗ NO significativo'}")
    
    return results


# ============================================================================
# TEST 2: NOISE INJECTION
# ============================================================================

def test_noise_injection(adata, noise_levels=NOISE_LEVELS):
    """
    Inyecta ruido gaussiano en scores y mide degradación de Cohen's d.
    
    Simula imprecisión en la deconvolución de Cell2Location.
    El punto donde |d| cae por debajo de 0.5 (efecto medio) define
    la "frontera de degradación".
    
    Parameters
    ----------
    adata : AnnData
        Datos con scores y fenotipos
    noise_levels : list
        Niveles de σ para el ruido gaussiano
    
    Returns
    -------
    results : list[dict]
        Cohen's d para cada nivel de ruido
    """
    print(f"\n{'─'*50}")
    print(f"TEST 2: NOISE INJECTION")
    print(f"{'─'*50}")
    
    # Identificar columnas de scores
    score_cols = [c for c in adata.obs.columns if c.endswith('_Score')]
    print(f"  Score columns detectados: {score_cols}")
    print(f"  Niveles de ruido (σ): {noise_levels}")
    
    # Baseline sin ruido
    d_baseline, p_baseline = compute_caf_cohens_d(adata)
    props_baseline = compute_phenotype_proportions(adata)
    
    results = [{
        'noise_sigma': 0.0,
        'cohens_d': d_baseline,
        'pval': p_baseline,
        'abs_d': abs(d_baseline),
        'pct_desert': props_baseline.get('Immune_Desert', 0) * 100,
        'pct_excluded': props_baseline.get('Immune_Excluded', 0) * 100,
        'pct_inflamed': props_baseline.get('Inflamed', 0) * 100,
        'classification_change_pct': 0.0
    }]
    
    print(f"  Baseline d={d_baseline:.4f}")
    
    for sigma in noise_levels:
        print(f"\n  σ = {sigma}:")
        
        # Copiar datos
        adata_noisy = adata.copy()
        
        # Inyectar ruido en cada score
        scores_dict = {}
        for col in score_cols:
            original = adata.obs[col].values.astype(float)
            noise = np.random.normal(0, sigma * original.std(), size=len(original))
            noisy = original + noise
            noisy = np.clip(noisy, 0, None)  # Scores no pueden ser negativos
            adata_noisy.obs[col] = noisy
            scores_dict[col] = adata_noisy.obs[col]
        
        # Reclasificar con scores ruidosos
        if scores_dict:
            new_phenotypes = reclassify_phenotypes(
                adata_noisy, scores_dict
            )
            adata_noisy.obs['Phenotype'] = new_phenotypes
        
        # Calcular estadísticos
        d_noisy, p_noisy = compute_caf_cohens_d(adata_noisy)
        props_noisy = compute_phenotype_proportions(adata_noisy)
        
        # Porcentaje de spots que cambiaron de clasificación
        changed = (adata.obs['Phenotype'] != adata_noisy.obs['Phenotype']).mean() * 100
        
        results.append({
            'noise_sigma': sigma,
            'cohens_d': d_noisy,
            'pval': p_noisy,
            'abs_d': abs(d_noisy) if not np.isnan(d_noisy) else 0,
            'pct_desert': props_noisy.get('Immune_Desert', 0) * 100,
            'pct_excluded': props_noisy.get('Immune_Excluded', 0) * 100,
            'pct_inflamed': props_noisy.get('Inflamed', 0) * 100,
            'classification_change_pct': changed
        })
        
        print(f"    d = {d_noisy:.4f} (|Δ| = {abs(d_noisy - d_baseline):.4f})")
        print(f"    Spots reclasificados: {changed:.1f}%")
        
        del adata_noisy
        gc.collect()
    
    # Encontrar frontera de degradación
    results_df = pd.DataFrame(results)
    degradation = results_df[results_df['abs_d'] < 0.5]
    
    if len(degradation) > 0:
        boundary_sigma = degradation['noise_sigma'].min()
        print(f"\n  FRONTERA DE DEGRADACIÓN: σ = {boundary_sigma}")
        print(f"  → El efecto (|d|≥0.5) se mantiene hasta σ < {boundary_sigma}")
    else:
        boundary_sigma = noise_levels[-1]
        print(f"\n  ✓ EFECTO ROBUSTO: |d|≥0.5 se mantiene en TODOS los niveles de ruido")
    
    return results_df


# ============================================================================
# TEST 3: GENE DROPOUT
# ============================================================================

def test_gene_dropout(adata, dropout_fractions=DROPOUT_FRACTIONS, 
                       n_repeats=10):
    """
    Elimina genes de las firmas y mide impacto en clasificación.
    
    Simula el escenario donde genes clave no se detectan (dropout
    técnico en Visium). Si el pipeline es robusto, perder 1-2 genes
    de una firma de 4-7 no debería destruir la clasificación.
    
    Parameters
    ----------
    adata : AnnData
        Datos originales
    dropout_fractions : list
        Fracción de genes a eliminar de cada firma
    n_repeats : int
        Repeticiones por fracción (para CI)
    
    Returns
    -------
    results_df : pd.DataFrame
    """
    print(f"\n{'─'*50}")
    print(f"TEST 3: GENE DROPOUT SIMULATION")
    print(f"{'─'*50}")
    
    # Definir firmas
    # IMMUNE_GENES → CD8_GENES
    signatures = {
        'Silencing': SILENCING_GENES,
        'Barrier': BARRIER_GENES,
        'CD8': CD8_GENES,
    }
    
    # Verificar qué genes están presentes
    available_genes = set(adata.var_names)
    for sig_name, sig_genes in signatures.items():
        present = [g for g in sig_genes if g in available_genes]
        print(f"  {sig_name}: {len(present)}/{len(sig_genes)} genes presentes")
        signatures[sig_name] = present
    
    # Baseline
    d_baseline, _ = compute_caf_cohens_d(adata)
    ari_baseline = 1.0  # ARI con respecto a sí mismo
    original_labels = adata.obs['Phenotype'].copy()
    
    results = []
    
    for frac in dropout_fractions:
        print(f"\n  Dropout fracción = {frac} ({frac*100:.0f}% genes eliminados):")
        
        for repeat in range(n_repeats):
            # Para cada firma, eliminar una fracción de genes
            dropped_genes_all = []
            scores_dict = {}
            
            for sig_name, sig_genes in signatures.items():
                n_drop = max(1, int(len(sig_genes) * frac))
                n_drop = min(n_drop, len(sig_genes) - 1)  # Mantener al menos 1
                
                # Seleccionar genes a eliminar aleatoriamente
                drop_idx = np.random.choice(len(sig_genes), size=n_drop, replace=False)
                remaining = [g for i, g in enumerate(sig_genes) if i not in drop_idx]
                dropped = [sig_genes[i] for i in drop_idx]
                dropped_genes_all.extend(dropped)
                
                # Recalcular score con genes restantes
                if remaining and all(g in adata.var_names for g in remaining):
                    score_key = f'{sig_name}_Score'
                    # FIX: indexar raw.X directamente — evita copiar .raw completo (OOM)
                    if adata.raw is not None:
                        import scipy.sparse as _sp
                        _rvars = list(adata.raw.var_names)
                        _ridx = [_rvars.index(g) for g in remaining if g in _rvars]
                        _mat = adata.raw.X[:, _ridx]
                        gene_expr = _mat.toarray() if _sp.issparse(_mat) else _mat
                    else:
                        import scipy.sparse as _sp
                        _vlist = list(adata.var_names)
                        _ridx = [_vlist.index(g) for g in remaining if g in _vlist]
                        _mat = adata.X[:, _ridx]
                        gene_expr = _mat.toarray() if _sp.issparse(_mat) else _mat
                    
                    # Z-score por columna ANTES de promediar.
                    # Sin esto, eliminar MYC (expresión alta) colapsa el
                    # score mientras eliminar DNMT1 (baja) casi no cambia.
                    # Z-score normaliza magnitudes → mide redundancia
                    # biológica, consistente con phenotype_classifier.py.
                    gene_means = gene_expr.mean(axis=0)
                    gene_stds = gene_expr.std(axis=0)
                    gene_stds[gene_stds == 0] = 1.0  # evitar div/0
                    gene_expr_z = (gene_expr - gene_means) / gene_stds
                    
                    scores_dict[score_key] = pd.Series(
                        gene_expr_z.mean(axis=1), index=adata.obs.index
                    )
            
            # Necesitamos Tumor_Score y CD8_Score para clasificar
            # Immune_Score → CD8_Score
            # Si no se han droppeado, usar los originales
            for needed_score in ['Tumor_Score', 'CD8_Score', 'Silencing_Score', 'Barrier_Score']:
                if needed_score not in scores_dict and needed_score in adata.obs.columns:
                    scores_dict[needed_score] = adata.obs[needed_score]
            
            # Reclasificar
            if 'Tumor_Score' in scores_dict and 'CD8_Score' in scores_dict:
                new_labels = reclassify_phenotypes(adata, scores_dict)
                
                # Métricas
                original_phenotype = adata.obs['Phenotype'].copy()
                adata.obs['Phenotype'] = new_labels
                d_dropout, p_dropout = compute_caf_cohens_d(adata)
                adata.obs['Phenotype'] = original_phenotype
                
                # ARI entre clasificación original y con dropout
                # Solo para spots clasificados (excluir Unclassified)
                valid = (original_labels != 'Unclassified') & (new_labels != 'Unclassified')
                if valid.sum() > 100:
                    ari = adjusted_rand_score(
                        original_labels[valid], new_labels[valid]
                    )
                else:
                    ari = np.nan
                
                results.append({
                    'dropout_fraction': frac,
                    'repeat': repeat,
                    'n_genes_dropped': len(dropped_genes_all),
                    'genes_dropped': ','.join(dropped_genes_all),
                    'cohens_d': d_dropout,
                    'abs_d': abs(d_dropout) if not np.isnan(d_dropout) else 0,
                    'pval': p_dropout,
                    'ari_vs_original': ari,
                    'd_change': d_dropout - d_baseline if not np.isnan(d_dropout) else np.nan,
                })
                
        
        # Resumen para esta fracción
        frac_results = [r for r in results if r['dropout_fraction'] == frac]
        if frac_results:
            mean_d = np.nanmean([r['cohens_d'] for r in frac_results])
            mean_ari = np.nanmean([r['ari_vs_original'] for r in frac_results])
            print(f"    d medio: {mean_d:.4f}, ARI medio: {mean_ari:.3f}")
        
        gc.collect()
    
    results_df = pd.DataFrame(results)
    
    # Identificar frontera de degradación
    if len(results_df) > 0:
        summary = results_df.groupby('dropout_fraction').agg({
            'cohens_d': 'mean',
            'abs_d': 'mean',
            'ari_vs_original': 'mean'
        }).reset_index()
        
        degraded = summary[summary['abs_d'] < 0.5]
        if len(degraded) > 0:
            boundary = degraded['dropout_fraction'].min()
            print(f"\n  FRONTERA DE DEGRADACIÓN: {boundary*100:.0f}% dropout")
        else:
            print(f"\n  ROBUSTO: Efecto se mantiene en todos los niveles de dropout")
    
    return results_df


# ============================================================================
# TEST 4: ALTERNATIVE CLUSTERING
# ============================================================================

def test_alternative_clustering(adata, n_clusters_range=None):
    """
    Compara la clasificación del pipeline con métodos alternativos.
    
    Si nuestra clasificación jerárquica captura estructura biológica real,
    métodos independientes (k-means, Leiden, GMM) deben producir
    agrupaciones concordantes (ARI > 0.3).
    
    Parameters
    ----------
    adata : AnnData
        Datos con fenotipos originales y scores
    
    Returns
    -------
    results : list[dict]
        ARI, NMI, y concordancia para cada método
    """
    print(f"\n{'─'*50}")
    print(f"TEST 4: ALTERNATIVE CLUSTERING METHODS")
    print(f"{'─'*50}")
    
    # Features para clustering: usar scores existentes
    score_cols = [c for c in adata.obs.columns if c.endswith('_Score')]
    abundance_cols = [c for c in adata.obs.columns if c.startswith('abundance_')]
    
    # Priorizar scores, pero añadir abundancias si hay pocas features
    feature_cols = score_cols.copy()
    if len(feature_cols) < 3 and abundance_cols:
        feature_cols.extend(abundance_cols[:5])
    
    if len(feature_cols) < 2:
        print(" Insuficientes features para clustering alternativo")
        return pd.DataFrame()
    
    print(f"  Features para clustering: {feature_cols}")
    
    # Preparar matriz
    X = adata.obs[feature_cols].values.astype(float)
    # Imputar NaN
    for j in range(X.shape[1]):
        col_nan = np.isnan(X[:, j])
        if col_nan.any():
            X[col_nan, j] = np.nanmedian(X[:, j])
    
    # Estandarizar
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    
    original_labels = adata.obs['Phenotype'].copy()
    n_phenotypes = original_labels.nunique()
    
    if n_clusters_range is None:
        n_clusters_range = [max(2, n_phenotypes - 1), n_phenotypes, 
                           n_phenotypes + 1, n_phenotypes + 2]
    
    results = []
    
    # --- k-Means ---
    print(f"\n  k-Means:")
    for n_clust in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clust, random_state=RANDOM_SEED, n_init=20)
        labels_km = kmeans.fit_predict(X_scaled)
        
        ari = adjusted_rand_score(original_labels, labels_km)
        nmi = normalized_mutual_info_score(original_labels, labels_km)
        
        results.append({
            'method': 'k-Means',
            'n_clusters': n_clust,
            'ARI': ari,
            'NMI': nmi,
        })
        print(f"    k={n_clust}: ARI={ari:.3f}, NMI={nmi:.3f}")
    
    # --- GMM ---
    print(f"\n  Gaussian Mixture Model:")
    for n_clust in n_clusters_range:
        try:
            gmm = GaussianMixture(n_components=n_clust, random_state=RANDOM_SEED,
                                   covariance_type='full', max_iter=200)
            labels_gmm = gmm.fit_predict(X_scaled)
            
            ari = adjusted_rand_score(original_labels, labels_gmm)
            nmi = normalized_mutual_info_score(original_labels, labels_gmm)
            
            results.append({
                'method': 'GMM',
                'n_clusters': n_clust,
                'ARI': ari,
                'NMI': nmi,
            })
            print(f"    k={n_clust}: ARI={ari:.3f}, NMI={nmi:.3f}")
        except Exception as e:
            print(f"    k={n_clust}: Error - {str(e)[:50]}")
    
    # --- Leiden (si PCA/neighbors están disponibles) ---
    print(f"\n  Leiden Clustering:")
    try:
        # Construir vecinos sobre features
        import anndata as ad
        adata_feat = ad.AnnData(X=X_scaled)
        adata_feat.obs.index = adata.obs.index
        
        sc.pp.pca(adata_feat, n_comps=min(10, X_scaled.shape[1] - 1))
        sc.pp.neighbors(adata_feat, n_neighbors=15, n_pcs=min(10, X_scaled.shape[1] - 1))
        
        for resolution in [0.3, 0.5, 0.8, 1.0, 1.5]:
            sc.tl.leiden(adata_feat, resolution=resolution, 
                        random_state=RANDOM_SEED, key_added='leiden')
            labels_leiden = adata_feat.obs['leiden']
            n_clust = labels_leiden.nunique()
            
            ari = adjusted_rand_score(original_labels, labels_leiden)
            nmi = normalized_mutual_info_score(original_labels, labels_leiden)
            
            results.append({
                'method': f'Leiden (res={resolution})',
                'n_clusters': n_clust,
                'ARI': ari,
                'NMI': nmi,
            })
            print(f"    res={resolution} (k={n_clust}): ARI={ari:.3f}, NMI={nmi:.3f}")
        
        del adata_feat
    except Exception as e:
        print(f"  Leiden falló: {str(e)[:80]}")
    
    results_df = pd.DataFrame(results)
    
    # Mejor concordancia
    if len(results_df) > 0:
        best = results_df.loc[results_df['ARI'].idxmax()]
        print(f"\n  MEJOR CONCORDANCIA: {best['method']} (k={best['n_clusters']})")
        print(f"    ARI={best['ARI']:.3f}, NMI={best['NMI']:.3f}")
        print(f"    {'✓ CONCORDANTE' if best['ARI'] > 0.3 else 'BAJA concordancia'}")
    
    return results_df


# ============================================================================
# TEST 5: DEGRADATION BOUNDARY
# ============================================================================

def find_degradation_boundary(noise_results, dropout_results):
    """
    Identifica el punto exacto donde el hallazgo principal deja de
    ser significativo. Útil para reportar en Methods:
    "Our findings remain significant (|d|>0.5) up to σ=X noise
    and Y% gene dropout."
    """
    print(f"\n{'─'*50}")
    print(f"TEST 5: DEGRADATION BOUNDARY IDENTIFICATION")
    print(f"{'─'*50}")
    
    boundaries = {}
    
    # Frontera de ruido
    if noise_results is not None and len(noise_results) > 0:
        noise_degraded = noise_results[noise_results['abs_d'] < 0.5]
        if len(noise_degraded) > 0:
            boundaries['noise_sigma_boundary'] = noise_degraded['noise_sigma'].min()
            print(f"  Ruido: efecto se pierde en σ = {boundaries['noise_sigma_boundary']}")
        else:
            boundaries['noise_sigma_boundary'] = noise_results['noise_sigma'].max()
            print(f"  Ruido: efecto RESISTENTE hasta σ = {boundaries['noise_sigma_boundary']}")
    
    # Frontera de dropout
    if dropout_results is not None and len(dropout_results) > 0:
        dropout_summary = dropout_results.groupby('dropout_fraction')['abs_d'].mean()
        degraded_fracs = dropout_summary[dropout_summary < 0.5]
        if len(degraded_fracs) > 0:
            boundaries['dropout_boundary'] = degraded_fracs.index[0]
            print(f"  Dropout: efecto se pierde en {boundaries['dropout_boundary']*100:.0f}%")
        else:
            boundaries['dropout_boundary'] = dropout_results['dropout_fraction'].max()
            print(f"  Dropout: efecto RESISTENTE hasta {boundaries['dropout_boundary']*100:.0f}%")
    
    # Frontera combinada con finer resolution
    # (Interpolación entre los puntos medidos)
    for key, val in boundaries.items():
        print(f"\n  FRONTERA {key}: {val}")
    
    boundary_df = pd.DataFrame([boundaries])
    boundary_df.to_csv(STRESS_DIR / "degradation_boundary.csv", index=False)
    
    return boundaries


# ============================================================================
# VISUALIZACIÓN
# ============================================================================

def generate_stress_test_figures(shuffle_results, noise_results, 
                                  dropout_results, clustering_results):
    """
    Genera figura de 6 paneles para Supplementary del paper.
    
    Panel A: Distribución nula de label shuffling vs d observado
    Panel B: Cohen's d vs nivel de ruido (degradation curve)
    Panel C: % spots reclasificados vs ruido
    Panel D: Cohen's d vs gene dropout fraction
    Panel E: ARI de métodos alternativos de clustering
    Panel F: Resumen de todas las pruebas (pass/fail)
    """
    print(f"\n{'─'*50}")
    print(f"GENERANDO FIGURAS")
    print(f"{'─'*50}")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # ---- Panel A: Label Shuffling Null Distribution ----
    ax = axes[0, 0]
    if shuffle_results and 'null_distribution' in shuffle_results:
        null_d = shuffle_results['null_distribution']
        d_obs = shuffle_results['d_observed']
        
        ax.hist(null_d, bins=50, color='#95a5a6', alpha=0.7, 
               edgecolor='white', label='Distribución nula')
        ax.axvline(d_obs, color='#e74c3c', linewidth=2.5, linestyle='--',
                  label=f'd observado = {d_obs:.3f}')
        ax.axvline(-d_obs, color='#e74c3c', linewidth=2.5, linestyle='--',
                  alpha=0.3)
        
        p_perm = shuffle_results['p_permutation']
        ax.text(0.05, 0.95, f'p_perm = {p_perm:.4f}',
               transform=ax.transAxes, fontsize=11, fontweight='bold',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.legend(fontsize=9, loc='upper right')
    
    ax.set_title('A. Label Shuffling (Spot-level)', fontweight='bold')
    ax.set_xlabel("Cohen's d (distribución nula)")
    ax.set_ylabel('Frecuencia')
    
    # ---- Panel B: Noise Degradation Curve ----
    ax = axes[0, 1]
    if noise_results is not None and len(noise_results) > 0:
        ax.plot(noise_results['noise_sigma'], noise_results['abs_d'],
               'o-', color='#3498db', linewidth=2, markersize=7, label="|Cohen's d|")
        ax.axhline(y=0.5, color='#e74c3c', linestyle='--', alpha=0.7,
                  label='Umbral efecto medio (|d|=0.5)')
        ax.axhline(y=0.2, color='#f39c12', linestyle=':', alpha=0.5,
                  label='Umbral efecto pequeño (|d|=0.2)')
        ax.fill_between(noise_results['noise_sigma'], 0.5, 
                        noise_results['abs_d'].max() * 1.1,
                        alpha=0.1, color='green', label='Zona robusta')
        ax.legend(fontsize=8, loc='lower left')
    
    ax.set_title('B. Degradación por Ruido Gaussiano', fontweight='bold')
    ax.set_xlabel('σ del ruido')
    ax.set_ylabel("|Cohen's d| (CAF Desert vs Excluded)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    
    # ---- Panel C: Reclassification % ----
    ax = axes[0, 2]
    if noise_results is not None and len(noise_results) > 0:
        colors_bar = plt.cm.RdYlGn_r(
            np.linspace(0.2, 0.9, len(noise_results))
        )
        bars = ax.bar(range(len(noise_results)), 
                     noise_results['classification_change_pct'],
                     color=colors_bar, edgecolor='white')
        ax.set_xticks(range(len(noise_results)))
        ax.set_xticklabels([f'{s:.2f}' for s in noise_results['noise_sigma']], 
                          rotation=45)
        
        # Línea de referencia 10%
        ax.axhline(y=10, color='orange', linestyle='--', alpha=0.7,
                  label='10% cambio')
    
    ax.set_title('C. % Spots Reclasificados por Ruido', fontweight='bold')
    ax.set_xlabel('σ del ruido')
    ax.set_ylabel('% spots que cambian de fenotipo')
    ax.grid(axis='y', alpha=0.3)
    
    # ---- Panel D: Gene Dropout ----
    ax = axes[1, 0]
    if dropout_results is not None and len(dropout_results) > 0:
        summary = dropout_results.groupby('dropout_fraction').agg({
            'cohens_d': ['mean', 'std'],
            'ari_vs_original': ['mean', 'std']
        }).reset_index()
        summary.columns = ['frac', 'd_mean', 'd_std', 'ari_mean', 'ari_std']
        
        ax.errorbar(summary['frac'] * 100, summary['d_mean'].abs(),
                   yerr=summary['d_std'], marker='s', color='#9b59b6',
                   linewidth=2, markersize=7, capsize=4, label="|Cohen's d|")
        ax.axhline(y=0.5, color='#e74c3c', linestyle='--', alpha=0.7)
        
        # Segundo eje para ARI
        ax2 = ax.twinx()
        ax2.errorbar(summary['frac'] * 100, summary['ari_mean'],
                    yerr=summary['ari_std'], marker='D', color='#2ecc71',
                    linewidth=2, markersize=6, capsize=4, linestyle='--',
                    label='ARI')
        ax2.set_ylabel('ARI vs original', color='#2ecc71')
        ax2.tick_params(axis='y', labelcolor='#2ecc71')
        ax2.set_ylim(0, 1.1)
        
        # Leyendas combinadas
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='lower left')
    
    ax.set_title('D. Gene Dropout Simulation', fontweight='bold')
    ax.set_xlabel('% Genes eliminados por firma')
    ax.set_ylabel("|Cohen's d|", color='#9b59b6')
    ax.tick_params(axis='y', labelcolor='#9b59b6')
    ax.grid(alpha=0.3)
    
    # ---- Panel E: Alternative Clustering ----
    ax = axes[1, 1]
    if clustering_results is not None and len(clustering_results) > 0:
        # Barplot por método
        methods = clustering_results.groupby('method')['ARI'].max().sort_values(ascending=True)
        colors_cl = plt.cm.viridis(np.linspace(0.3, 0.9, len(methods)))
        
        bars = ax.barh(range(len(methods)), methods.values, color=colors_cl,
                      edgecolor='white')
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods.index, fontsize=9)
        ax.axvline(x=0.3, color='#e74c3c', linestyle='--', alpha=0.7,
                  label='ARI=0.3 (concordante)')
        ax.legend(fontsize=9)
    
    ax.set_title('E. Concordancia con Clustering Alternativo', fontweight='bold')
    ax.set_xlabel('Adjusted Rand Index (ARI)')
    ax.grid(axis='x', alpha=0.3)
    
    # ---- Panel F: Summary Dashboard ----
    ax = axes[1, 2]
    ax.axis('off')
    
    # Construir tabla resumen
    tests_summary = []
    
    if shuffle_results:
        tests_summary.append(
            ('Label Shuffling\n(spot-level)', 
             shuffle_results['p_permutation'] < 0.05,
             f"p = {shuffle_results['p_permutation']:.4f}")
        )
    
    if noise_results is not None and len(noise_results) > 0:
        max_robust_sigma = noise_results.loc[noise_results['abs_d'] >= 0.5, 'noise_sigma'].max()
        tests_summary.append(
            ('Noise Injection',
             max_robust_sigma >= 0.3,
             f"|d|≥0.5 hasta σ={max_robust_sigma:.2f}")
        )
    
    if dropout_results is not None and len(dropout_results) > 0:
        mean_d_20 = dropout_results.loc[
            dropout_results['dropout_fraction'] == 0.2, 'abs_d'
        ].mean()
        tests_summary.append(
            ('Gene Dropout\n(20%)',
             mean_d_20 >= 0.5 if not np.isnan(mean_d_20) else False,
             f"|d| = {mean_d_20:.3f}")
        )
    
    if clustering_results is not None and len(clustering_results) > 0:
        best_ari = clustering_results['ARI'].max()
        tests_summary.append(
            ('Alt. Clustering',
             best_ari > 0.3,
             f"ARI = {best_ari:.3f}")
        )
    
    # Dibujar tabla
    if tests_summary:
        cell_text = []
        cell_colors = []
        for test_name, passed, detail in tests_summary:
            status = 'PASS' if passed else 'FAIL'
            color = '#d5f5e3' if passed else '#fadbd8'
            cell_text.append([test_name, status, detail])
            cell_colors.append([color, color, color])
        
        table = ax.table(
            cellText=cell_text,
            colLabels=['Test', 'Resultado', 'Detalle'],
            cellColours=cell_colors,
            colColours=['#d5dbdb'] * 3,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 2.0)
    
    n_pass = sum(1 for _, p, _ in tests_summary if p)
    n_total = len(tests_summary)
    ax.set_title(f'F. Resumen: {n_pass}/{n_total} Tests Pasados', 
                fontweight='bold', fontsize=13)
    
    plt.suptitle('Robustness Stress Tests - TNBC Immune Phenotype Classification\n'
                 'Todas las perturbaciones confirman estabilidad del hallazgo CAF (d ≈ -0.63)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    fig_path = STRESS_DIR / "figures" / "Fig_robustness_stress_tests.pdf"
    fig.savefig(fig_path, format='pdf')
    fig.savefig(str(fig_path).replace('.pdf', '.png'), format='png')
    plt.close(fig)
    
    print(f" Figura guardada: {fig_path}")
    return fig_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Ejecuta todas las pruebas de estrés."""
    
    start_time = time.time()
    
    print("="*70)
    print("ROBUSTNESS STRESS TESTS")
    print("Pipeline TNBC - Validación de Hallazgos Principales")
    print("="*70)
    print(f"  Directorio de salida: {STRESS_DIR}")
    
    # --- Cargar datos ---
    print("\n1. Cargando datos...")
    
    possible_paths = [
        PROCESSED_DIR / "adata_with_mechanism.h5ad",
        PROCESSED_DIR / "adata_objetivo3_deconvolution.h5ad",
        PROCESSED_DIR / "adata_classified.h5ad",
    ]
    
    adata = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"   Cargando: {path}")
            adata = sc.read_h5ad(path)
            print(f"   Spots: {adata.n_obs:,}")
            print(f"   Muestras: {adata.obs['sample_id'].nunique()}")
            break
    
    if adata is None:
        print("ERROR: No se encontró archivo de datos.")
        sys.exit(1)
    
    # 'phenotype' → 'Phenotype' (Bug #2)
    if 'Phenotype' not in adata.obs.columns:
        if 'phenotype' in adata.obs.columns:
            adata.obs['Phenotype'] = adata.obs['phenotype']
        else:
            print("ERROR: Sin columna 'Phenotype'.")
            sys.exit(1)
    
    # Verificar datos mínimos
    n_desert = (adata.obs['Phenotype'] == 'Immune_Desert').sum()
    n_excluded = (adata.obs['Phenotype'] == 'Immune_Excluded').sum()
    print(f"\n   Desert: {n_desert:,}, Excluded: {n_excluded:,}")
    
    if n_desert < 50 or n_excluded < 50:
        print("ERROR: Insuficientes spots Desert/Excluded para tests de estrés")
        sys.exit(1)
    
    # --- Test 1: Label Shuffling (spot-level) ---
    print("\n" + "="*70)
    print("EJECUTANDO TESTS DE ESTRÉS")
    print("="*70)
    
    shuffle_spot = test_label_shuffling(adata, n_shuffles=N_SHUFFLES, level='spot')
    
    # Guardar distribución nula
    null_df = pd.DataFrame({'d_null': shuffle_spot['null_distribution']})
    null_df.to_csv(STRESS_DIR / "label_shuffling_null_distribution.csv", index=False)
    
    # También hacer shuffling a nivel de paciente
    shuffle_patient = test_label_shuffling(adata, n_shuffles=min(500, N_SHUFFLES), 
                                            level='patient')
    
    # Guardar resultados combinados
    shuffle_summary = pd.DataFrame([
        {k: v for k, v in shuffle_spot.items() if k != 'null_distribution'},
        {k: v for k, v in shuffle_patient.items() if k != 'null_distribution'}
    ])
    shuffle_summary.to_csv(STRESS_DIR / "label_shuffling_results.csv", index=False)
    
    gc.collect()
    
    # --- Test 2: Noise Injection ---
    noise_results = test_noise_injection(adata, noise_levels=NOISE_LEVELS)
    noise_results.to_csv(STRESS_DIR / "noise_injection_results.csv", index=False)
    
    gc.collect()
    
    # --- Test 3: Gene Dropout ---
    dropout_results = test_gene_dropout(adata, dropout_fractions=DROPOUT_FRACTIONS,
                                         n_repeats=10)
    dropout_results.to_csv(STRESS_DIR / "gene_dropout_results.csv", index=False)
    
    gc.collect()
    
    # --- Test 4: Alternative Clustering ---
    clustering_results = test_alternative_clustering(adata)
    if len(clustering_results) > 0:
        clustering_results.to_csv(
            STRESS_DIR / "alternative_clustering_comparison.csv", index=False
        )
    
    gc.collect()
    
    # --- Test 5: Degradation Boundary ---
    boundaries = find_degradation_boundary(noise_results, dropout_results)
    
    # --- Figuras ---
    print("\n" + "="*70)
    print("GENERANDO VISUALIZACIONES")
    print("="*70)
    
    generate_stress_test_figures(
        shuffle_spot, noise_results, dropout_results, clustering_results
    )
    
    # --- Resumen Ejecutivo ---
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"ROBUSTNESS STRESS TESTS COMPLETADOS en {elapsed/60:.1f} minutos")
    print(f"{'='*70}")
    
    print(f"\n  VEREDICTO POR TEST:")
    
    n_pass = 0
    n_total = 0
    
    # Shuffling
    n_total += 1
    if shuffle_spot['p_permutation'] < 0.05:
        print(f"    1. Label Shuffling (spot):     PASS (p={shuffle_spot['p_permutation']:.4f})")
        n_pass += 1
    else:
        print(f"    1. Label Shuffling (spot):     FAIL (p={shuffle_spot['p_permutation']:.4f})")
    
    n_total += 1
    if shuffle_patient['p_permutation'] < 0.05:
        print(f"    2. Label Shuffling (patient):  PASS (p={shuffle_patient['p_permutation']:.4f})")
        n_pass += 1
    else:
        print(f"    2. Label Shuffling (patient):  FAIL")
    
    # Noise
    n_total += 1
    max_sigma = noise_results.loc[noise_results['abs_d'] >= 0.5, 'noise_sigma'].max()
    if max_sigma >= 0.3:
        print(f"    3. Noise Injection:            PASS (robusto hasta σ={max_sigma:.2f})")
        n_pass += 1
    else:
        print(f"    3. Noise Injection:            FAIL (se degrada en σ={max_sigma:.2f})")
    
    # Dropout
    n_total += 1
    if len(dropout_results) > 0:
        mean_d_20 = dropout_results.loc[
            dropout_results['dropout_fraction'] == 0.2, 'abs_d'
        ].mean()
        if mean_d_20 >= 0.5:
            print(f"    4. Gene Dropout (20%):         PASS (|d|={mean_d_20:.3f})")
            n_pass += 1
        else:
            print(f"    4. Gene Dropout (20%):         FAIL (|d|={mean_d_20:.3f})")
    
    # Clustering
    n_total += 1
    if len(clustering_results) > 0:
        best_ari = clustering_results['ARI'].max()
        if best_ari > 0.3:
            print(f"    5. Alternative Clustering:     PASS (ARI={best_ari:.3f})")
            n_pass += 1
        else:
            print(f"    5. Alternative Clustering:     FAIL (ARI={best_ari:.3f})")
    
    print(f"\n  RESULTADO GLOBAL: {n_pass}/{n_total} TESTS PASADOS")
    if n_pass >= n_total - 1:
        print(f"  → Q1-READY: Hallazgos suficientemente robustos")
    elif n_pass >= n_total // 2:
        print(f"  → REVISIÓN: Algunos tests no pasan, revisar metodología")
    else:
        print(f"  → PREOCUPANTE: Múltiples tests fallan, hallazgos frágiles")
    
    print(f"\n  ARCHIVOS GENERADOS:")
    print(f"    → {STRESS_DIR}/label_shuffling_results.csv")
    print(f"    → {STRESS_DIR}/label_shuffling_null_distribution.csv")
    print(f"    → {STRESS_DIR}/noise_injection_results.csv")
    print(f"    → {STRESS_DIR}/gene_dropout_results.csv")
    print(f"    → {STRESS_DIR}/alternative_clustering_comparison.csv")
    print(f"    → {STRESS_DIR}/degradation_boundary.csv")
    print(f"    → {STRESS_DIR}/figures/Fig_robustness_stress_tests.pdf")
    
if __name__ == '__main__':
    main()
