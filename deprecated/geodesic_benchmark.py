"""
================================================================================
GEODESIC BENCHMARK - Comparación Formal Geodésica vs Euclidiana
================================================================================

Propósito:
    Demostrar formalmente que la distancia geodésica captura mejor la
    arquitectura tisular que la euclidiana, respondiendo a la crítica:
    "¿Por qué geodésica y no euclidiana?"

Pruebas implementadas:
    1. Distancia Euclidiana completa (baseline)
    2. Distancia Geodésica multi-k (k=4,6,8,10)
    3. Detour Penalty (ratio geodésica/euclidiana)
    4. Tests estadísticos pareados (Wilcoxon signed-rank)
    5. Comparación detour por fenotipo (Desert vs Excluded)

================================================================================
"""

import os
import sys
import time
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
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
    from config import PATHS, SIGNATURES
    BASE_DIR = PATHS.BASE_DIR
    RESULTS_DIR = PATHS.RESULTS_DIR
    PROCESSED_DIR = PATHS.PROCESSED_DIR
    print("Configuración cargada desde config.py")
except ImportError:
    # Fallback para ejecución standalone
    from pathlib import Path
    BASE_DIR = Path("/home/external/frjimenez/fabian/genoma")
    RESULTS_DIR = BASE_DIR / "results"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    print("config.py no encontrado, usando rutas HPC por defecto")

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

# Directorio de salida para este módulo
BENCHMARK_DIR = RESULTS_DIR / "geodesic_benchmark"
os.makedirs(BENCHMARK_DIR, exist_ok=True)
os.makedirs(BENCHMARK_DIR / "figures", exist_ok=True)

# Parámetros del benchmark
K_VALUES = [4, 6, 8, 10]          # Valores de k para sensibilidad
K_DEFAULT = 6                      # Geometría hexagonal Visium
ABUNDANCE_PERCENTILES = [70, 75, 80]  # Umbrales de abundancia
MAX_DISTANCE = 50                  # Límite computacional (hops)
N_PERMUTATIONS = 1000              # Para tests de significancia
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Configuración de figuras
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'figure.facecolor': 'white'
})


# ============================================================================
# 1. FUNCIONES DE DISTANCIA
# ============================================================================

def compute_euclidean_distances(coords_source, coords_target):
    """
    Calcula distancias euclidianas mínimas desde spots fuente a spots objetivo.
    
    Usa KDTree para eficiencia sobre datasets grandes (>70k spots).
    
    FIX H9: Usa k=2 para evitar autocorrelación (distancia=0) cuando un
    spot contiene tanto el tipo celular fuente como el objetivo.
    
    Parameters
    ----------
    coords_source : np.ndarray (N, 2)
        Coordenadas de spots fuente (e.g., tumor)
    coords_target : np.ndarray (M, 2)
        Coordenadas de spots objetivo (e.g., cDC1-high)
    
    Returns
    -------
    distances : np.ndarray (N,)
        Distancia euclidiana mínima de cada fuente al objetivo más cercano
    """
    if len(coords_target) == 0:
        return np.full(len(coords_source), np.inf)
    
    tree = KDTree(coords_target)
    
    # FIX H9: Consultar k=2 vecinos para evitar autocorrelación (dist=0)
    k_query = min(2, len(coords_target))
    distances, indices = tree.query(coords_source, k=k_query)
    
    if k_query == 1:
        return distances.ravel()
    
    # Para cada spot fuente, tomar el vecino más cercano que no sea sí mismo
    final_distances = np.empty(len(coords_source))
    for i in range(len(coords_source)):
        if distances[i, 0] > 1e-10:
            # Primer vecino no es sí mismo
            final_distances[i] = distances[i, 0]
        else:
            # Primer vecino es sí mismo (dist≈0), usar segundo
            final_distances[i] = distances[i, 1] if k_query > 1 else distances[i, 0]
    
    return final_distances


def build_spatial_graph_topological(coords, k_neighbors=6):
    """
    Construye grafo espacial k-NN TOPOLÓGICO (hop count, sin pesos).

    Parameters
    ----------
    coords : np.ndarray (N, 2)
        Coordenadas espaciales de todos los spots
    k_neighbors : int
        Número de vecinos para el grafo k-NN
    
    Returns
    -------
    graph : scipy.sparse.csr_matrix (N, N)
        Grafo de adyacencia topológico (pesos = 1)
    """
    n_spots = len(coords)
    tree = KDTree(coords)
    
    # Encontrar k vecinos más cercanos
    distances, indices = tree.query(coords, k=k_neighbors + 1)
    
    # Construir matriz de adyacencia sparse con pesos UNITARIOS
    rows = []
    cols = []
    weights = []
    
    for i in range(n_spots):
        for j_idx in range(1, k_neighbors + 1):  # Skip self (index 0)
            j = indices[i, j_idx]
            
            # FIX H1/H2: Peso = 1.0 (hop count topológico puro)
            # NO usamos abundance_weight ni CAF. La topología del tejido
            # per se captura barreras de exclusión.
            rows.append(i)
            cols.append(j)
            weights.append(1.0)
    
    graph = csr_matrix(
        (weights, (rows, cols)),
        shape=(n_spots, n_spots)
    )
    
    # Hacer simétrico (grafo no dirigido)
    graph = graph + graph.T
    # Para hop count, la simetría duplica pesos → normalizar a 1
    graph.data = np.minimum(graph.data, 1.0)
    
    return graph


def compute_geodesic_distances(graph, source_indices, target_indices,
                                max_distance=50):
    """
    Calcula distancias geodésicas mínimas usando Dijkstra sobre el grafo espacial.
    
    Con grafo topológico (hop count), max_distance=50 = 50 saltos,
    lo cual cubre prácticamente toda la muestra Visium.
    
    Parameters
    ----------
    graph : scipy.sparse.csr_matrix
        Grafo de adyacencia (topológico, pesos=1)
    source_indices : np.ndarray
        Índices de spots fuente
    target_indices : np.ndarray
        Índices de spots objetivo
    max_distance : float
        Límite de distancia en hops (para eficiencia)
    
    Returns
    -------
    min_distances : np.ndarray
        Distancia geodésica mínima de cada fuente al objetivo más cercano
    """
    if len(target_indices) == 0:
        return np.full(len(source_indices), np.inf)
    
    # Dijkstra desde spots objetivo (invertido para eficiencia)
    # Calculamos desde targets porque suelen ser menos que sources
    dist_matrix = dijkstra(
        graph,
        directed=False,
        indices=target_indices,
        limit=max_distance
    )
    
    # Para cada source, encontrar el target más cercano
    # dist_matrix shape: (n_targets, n_spots)
    min_distances = np.full(len(source_indices), np.inf)
    
    for i, src_idx in enumerate(source_indices):
        dists_to_targets = dist_matrix[:, src_idx]
        valid = dists_to_targets < np.inf
        if valid.any():
            min_distances[i] = dists_to_targets[valid].min()
    
    return min_distances


# ============================================================================
# 2. BENCHMARK PRINCIPAL
# ============================================================================

def run_distance_benchmark_per_sample(adata, sample_id, cell_type_col,
                                       target_cell='cDC1',
                                       k_values=K_VALUES,
                                       abundance_percentile=75):
    """
    Ejecuta benchmark geodésica vs euclidiana para una muestra.
    
    Para cada muestra:
    1. Identifica spots tumorales (fuente) y spots con alta abundancia
       del tipo celular objetivo (cDC1/CD8)
    2. Calcula distancia euclidiana (baseline)
    3. Calcula distancia geodésica topológica para múltiples valores de k
    4. Computa detour penalty SPOT-A-SPOT (ratio geo/euc)
    
    Parameters
    ----------
    adata : AnnData
        Objeto con datos espaciales y deconvolución
    sample_id : str
        ID de la muestra a analizar
    cell_type_col : str
        Columna en adata.obs con abundancia del tipo celular objetivo
    target_cell : str
        Nombre del tipo celular para logging
    k_values : list
        Valores de k para sensibilidad del grafo
    abundance_percentile : int
        Percentil para definir "alta abundancia"
    
    Returns
    -------
    results : dict
        Resultados del benchmark para esta muestra
    """
    # Filtrar muestra
    mask = adata.obs['sample_id'] == sample_id
    adata_sample = adata[mask].copy()
    
    # Verificar coordenadas espaciales
    if 'spatial' not in adata_sample.obsm:
        return None
    
    coords = adata_sample.obsm['spatial']
    n_spots = len(adata_sample)
    
    if n_spots < 50:  # Muestra muy pequeña
        return None
    
    # Identificar spots tumorales (fuente)
    phenotypes = adata_sample.obs.get('Phenotype', None)
    if phenotypes is None:
        print(f"  ⚠ {sample_id}: Sin columna 'Phenotype'")
        return None
    
    tumor_phenotypes = ['Immune_Desert', 'Immune_Excluded', 'Inflamed']
    tumor_mask = phenotypes.isin(tumor_phenotypes)
    tumor_indices = np.where(tumor_mask.values)[0]
    
    if len(tumor_indices) < 10:
        return None
    
    # Identificar spots con alta abundancia del tipo celular objetivo
    if cell_type_col not in adata_sample.obs.columns:
        # Buscar columna alternativa
        possible_cols = [c for c in adata_sample.obs.columns 
                        if target_cell.lower() in c.lower()]
        if possible_cols:
            cell_type_col = possible_cols[0]
        else:
            return None
    
    abundance = adata_sample.obs[cell_type_col].values.astype(float)
    threshold = np.percentile(abundance[abundance > 0], abundance_percentile) \
                if (abundance > 0).sum() > 10 else np.percentile(abundance, 90)
    
    target_mask = abundance >= threshold
    target_indices = np.where(target_mask)[0]
    
    if len(target_indices) < 3:
        return None
    
    # --- Distancia Euclidiana (con fix H9: k=2 para evitar autocorrelación) ---
    euc_distances = compute_euclidean_distances(
        coords[tumor_indices], coords[target_indices]
    )
    
    # --- Distancias Geodésicas TOPOLÓGICAS para múltiples k ---
    # FIX H1/H2: Sin pesos CAF. Grafo topológico puro (hop count).
    results = {
        'sample_id': sample_id,
        'n_spots': n_spots,
        'n_tumor': len(tumor_indices),
        'n_target': len(target_indices),
        'target_cell': target_cell,
        'euclidean_mean': np.nanmean(euc_distances[euc_distances < np.inf]),
        'euclidean_median': np.nanmedian(euc_distances[euc_distances < np.inf]),
    }
    
    geo_distances_by_k = {}
    
    for k in k_values:
        # FIX H1/H2: Grafo TOPOLÓGICO (sin pesos CAF)
        graph = build_spatial_graph_topological(coords, k_neighbors=k)
        
        geo_distances = compute_geodesic_distances(
            graph, tumor_indices, target_indices,
            max_distance=MAX_DISTANCE
        )
        
        # FIX H11: Detour SPOT-A-SPOT con filtrado de inf
        # Ambos arrays (geo y euc) tienen el mismo tamaño e indexación
        # (uno por cada tumor_index). Filtrar donde AMBOS son finitos.
        both_finite = (geo_distances < np.inf) & (euc_distances < np.inf)
        both_positive = both_finite & (geo_distances > 0) & (euc_distances > 1e-10)
        
        valid_geo = geo_distances[both_finite]
        
        if len(valid_geo) < 5:
            continue
        
        geo_mean = np.nanmean(valid_geo)
        geo_median = np.nanmedian(valid_geo)
        
        # FIX H11: Detour penalty SPOT-A-SPOT
        # Calcular ratio para cada spot individualmente, luego estadísticas
        if both_positive.sum() > 0:
            spot_detour = geo_distances[both_positive] / euc_distances[both_positive]
            detour_mean = np.nanmean(spot_detour)
            detour_median = np.nanmedian(spot_detour)
        else:
            detour_mean = np.nan
            detour_median = np.nan
        
        results[f'geodesic_k{k}_mean'] = geo_mean
        results[f'geodesic_k{k}_median'] = geo_median
        results[f'detour_k{k}_mean'] = detour_mean
        results[f'detour_k{k}_median'] = detour_median
        
        geo_distances_by_k[k] = geo_distances
    
    # --- Test pareado: ¿geodésica > euclidiana? ---
    if K_DEFAULT in geo_distances_by_k:
        geo_default = geo_distances_by_k[K_DEFAULT]
        # FIX H11: Usar máscara de ambos finitos
        valid = (geo_default < np.inf) & (euc_distances < np.inf)
        
        if valid.sum() > 10:
            stat, pval = stats.wilcoxon(
                geo_default[valid], euc_distances[valid],
                alternative='greater'  # H1: geodésica > euclidiana
            )
            results['wilcoxon_stat'] = stat
            results['wilcoxon_pval'] = pval
            results['wilcoxon_significant'] = pval < 0.05
            
            # Effect size (rank-biserial correlation)
            n = valid.sum()
            results['effect_size_r'] = 1 - (2 * stat) / (n * (n + 1) / 2)
    
    # --- Separar por fenotipo ---
    # FIX H11: Detour spot-a-spot por fenotipo
    for phenotype in ['Immune_Desert', 'Immune_Excluded']:
        pheno_mask = (phenotypes.values == phenotype) & tumor_mask.values
        pheno_indices = np.where(pheno_mask)[0]
        
        if len(pheno_indices) < 5:
            continue
        
        # Mapear a posiciones en tumor_indices
        pheno_in_tumor = np.isin(tumor_indices, pheno_indices)
        
        if pheno_in_tumor.sum() < 5:
            continue
        
        results[f'euc_{phenotype}_mean'] = np.nanmean(
            euc_distances[pheno_in_tumor]
        )
        
        if K_DEFAULT in geo_distances_by_k:
            geo_d = geo_distances_by_k[K_DEFAULT]
            
            # FIX H11: Máscara combinada para este fenotipo
            pheno_both_finite = pheno_in_tumor & (geo_d < np.inf) & (euc_distances < np.inf)
            pheno_geo_finite = pheno_in_tumor & (geo_d < np.inf)
            
            if pheno_geo_finite.sum() > 0:
                results[f'geo_{phenotype}_mean'] = np.nanmean(
                    geo_d[pheno_geo_finite]
                )
            
            # FIX H11: Detour spot-a-spot por fenotipo
            pheno_both_positive = (pheno_both_finite & 
                                   (geo_d > 0) & (euc_distances > 1e-10))
            if pheno_both_positive.sum() > 0:
                spot_detour_p = geo_d[pheno_both_positive] / euc_distances[pheno_both_positive]
                results[f'detour_{phenotype}_mean'] = np.nanmean(spot_detour_p)
                results[f'detour_{phenotype}_median'] = np.nanmedian(spot_detour_p)
                results[f'detour_{phenotype}_n'] = int(pheno_both_positive.sum())
    
    return results


def run_full_benchmark(adata, target_cells=None):
    """
    Ejecuta benchmark completo sobre todas las muestras.
    
    Parameters
    ----------
    adata : AnnData
        Objeto con deconvolución y fenotipos
    target_cells : dict, optional
        {nombre: columna_abundancia} para cada tipo celular objetivo
    
    Returns
    -------
    results_df : pd.DataFrame
        Resultados completos del benchmark
    """
    if target_cells is None:
        # Detectar columnas de abundancia en obsm 
        import pandas as pd
        abundance_cols = []
        for obsm_key in adata.obsm.keys():
            if 'abundance' in obsm_key:
                val = adata.obsm[obsm_key]
                if isinstance(val, pd.DataFrame):
                    abundance_cols.extend(list(val.columns))
        
        target_cells = {}
        for col in abundance_cols:
            name = col.lower()
            if any(x in name for x in ['cdc1', 'cdc', 'dendritic']):
                target_cells['cDC1'] = col
            elif any(x in name for x in ['cd8']):
                target_cells['CD8_T'] = col
        
        if not target_cells:
            print("⚠ No se detectaron cDC1/CD8 automáticamente")
            print(f"  Columnas disponibles: {abundance_cols[:10]}")
            if len(abundance_cols) >= 2:
                target_cells = {
                    'target_1': abundance_cols[0],
                    'target_2': abundance_cols[1]
                }
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK GEODÉSICA vs EUCLIDIANA")
    print(f"  Método: Grafo TOPOLÓGICO (hop count, sin pesos CAF)")
    print(f"  Correcciones: H1/H2 circularidad, H11 detour spot-a-spot")
    print(f"{'='*70}")
    print(f"  Tipos celulares objetivo: {list(target_cells.keys())}")
    print(f"  Valores de k: {K_VALUES}")
    print(f"  Muestras: {adata.obs['sample_id'].nunique()}")
    
    all_results = []
    samples = adata.obs['sample_id'].unique()
    
    for i, sample_id in enumerate(samples):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"\n  Procesando muestra {i+1}/{len(samples)}: {sample_id}")
        
        for cell_name, cell_col in target_cells.items():
            result = run_distance_benchmark_per_sample(
                adata=adata,
                sample_id=sample_id,
                cell_type_col=cell_col,
                target_cell=cell_name,
                k_values=K_VALUES,
                abundance_percentile=75
            )
            
            if result is not None:
                all_results.append(result)
        
        # Limpiar memoria
        if (i + 1) % 10 == 0:
            gc.collect()
    
    results_df = pd.DataFrame(all_results)
    
    # Guardar resultados
    output_file = BENCHMARK_DIR / "geodesic_vs_euclidean_comparison.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Resultados guardados: {output_file}")
    
    return results_df


# ============================================================================
# 3. ANÁLISIS DE SENSIBILIDAD A k
# ============================================================================

def analyze_k_sensitivity(results_df):
    """
    Analiza si el ratio geodésico/euclidiano es estable ante variación de k.
    
    Pregunta clave: ¿El aislamiento geodésico es robusto o depende
    del valor arbitrario k=6?
    """
    print(f"\n{'='*70}")
    print(f"ANÁLISIS DE SENSIBILIDAD A k")
    print(f"{'='*70}")
    
    k_summary = []
    
    for k in K_VALUES:
        geo_col = f'geodesic_k{k}_mean'
        detour_col = f'detour_k{k}_mean'
        
        if geo_col not in results_df.columns:
            continue
        
        valid = results_df[geo_col].notna()
        if valid.sum() == 0:
            continue
        
        geo_vals = results_df.loc[valid, geo_col]
        euc_vals = results_df.loc[valid, 'euclidean_mean']
        
        # Ratio geodésica/euclidiana promedio
        ratio = (geo_vals / euc_vals).mean()
        ratio_std = (geo_vals / euc_vals).std()
        
        # Detour penalty
        if detour_col in results_df.columns:
            detour = results_df.loc[valid, detour_col]
            detour_mean = detour.mean()
            detour_std = detour.std()
        else:
            detour_mean = np.nan
            detour_std = np.nan
        
        k_summary.append({
            'k': k,
            'n_samples': valid.sum(),
            'geo_euc_ratio_mean': ratio,
            'geo_euc_ratio_std': ratio_std,
            'detour_penalty_mean': detour_mean,
            'detour_penalty_std': detour_std,
            'geodesic_mean': geo_vals.mean(),
            'euclidean_mean': euc_vals.mean(),
        })
        
        print(f"\n  k={k}:")
        print(f"    Ratio Geo/Euc: {ratio:.3f} ± {ratio_std:.3f}")
        print(f"    Detour penalty: {detour_mean:.3f} ± {detour_std:.3f}")
    
    k_df = pd.DataFrame(k_summary)
    k_df.to_csv(BENCHMARK_DIR / "k_sensitivity_geodesic.csv", index=False)
    
    # Evaluar estabilidad
    if len(k_df) > 1:
        ratios = k_df['geo_euc_ratio_mean']
        cv = ratios.std() / ratios.mean() * 100
        print(f"\n  ESTABILIDAD:")
        print(f"    CV del ratio Geo/Euc: {cv:.1f}%")
        print(f"    {'✓ ESTABLE' if cv < 15 else '⚠ VARIABLE'} "
              f"(umbral CV < 15%)")
    
    return k_df


# ============================================================================
# 4. DETOUR PENALTY POR FENOTIPO
# ============================================================================

def analyze_detour_by_phenotype(results_df):
    """
    Compara detour penalty entre Desert y Excluded.
    
    Hipótesis: El detour penalty debe ser MAYOR en Excluded (CAFs bloquean),
    mientras que en Desert la ruta geodésica ~ euclidiana (sin barrera física,
    solo silenciamiento molecular).
    
    Esta es la prueba definitiva de que los mecanismos son diferentes.
    """
    print(f"\n{'='*70}")
    print(f"DETOUR PENALTY POR FENOTIPO")
    print(f"{'='*70}")
    
    phenotype_results = []
    
    for phenotype in ['Immune_Desert', 'Immune_Excluded']:
        detour_col = f'detour_{phenotype}_mean'
        geo_col = f'geo_{phenotype}_mean'
        euc_col = f'euc_{phenotype}_mean'
        
        if detour_col not in results_df.columns:
            continue
        
        valid = results_df[detour_col].notna()
        if valid.sum() < 3:
            continue
        
        detour_vals = results_df.loc[valid, detour_col]
        
        phenotype_results.append({
            'phenotype': phenotype,
            'n_samples': valid.sum(),
            'detour_mean': detour_vals.mean(),
            'detour_std': detour_vals.std(),
            'detour_median': detour_vals.median(),
            'geo_mean': results_df.loc[valid, geo_col].mean() if geo_col in results_df.columns else np.nan,
            'euc_mean': results_df.loc[valid, euc_col].mean() if euc_col in results_df.columns else np.nan,
        })
        
        print(f"\n  {phenotype}:")
        print(f"    Detour penalty: {detour_vals.mean():.3f} ± {detour_vals.std():.3f}")
        print(f"    N muestras: {valid.sum()}")
    
    # Test estadístico: Desert vs Excluded
    desert_col = 'detour_Immune_Desert_mean'
    excluded_col = 'detour_Immune_Excluded_mean'
    
    if desert_col in results_df.columns and excluded_col in results_df.columns:
        both_valid = results_df[desert_col].notna() & results_df[excluded_col].notna()
        
        if both_valid.sum() >= 5:
            desert_vals = results_df.loc[both_valid, desert_col]
            excluded_vals = results_df.loc[both_valid, excluded_col]
            
            stat, pval = stats.mannwhitneyu(
                excluded_vals, desert_vals,
                alternative='greater'  # H1: Excluded tiene mayor detour
            )
            
            # FIX FASE1-13: Cohen's d canónico (ddof=1, pooled ponderada)
            cohens_d = cohens_d_pooled(excluded_vals, desert_vals)
            
            print(f"\n  TEST ESTADÍSTICO (Excluded > Desert):")
            print(f"    Mann-Whitney U p-value: {pval:.2e}")
            print(f"    Cohen's d: {cohens_d:.3f}")
            print(f"    {'✓ SIGNIFICATIVO' if pval < 0.05 else '✗ NO significativo'}")
            
            phenotype_results.append({
                'phenotype': 'TEST_Excluded_vs_Desert',
                'n_samples': both_valid.sum(),
                'detour_mean': cohens_d,  # Store Cohen's d here
                'detour_std': pval,        # Store p-value here
                'detour_median': stat,
            })
    
    pheno_df = pd.DataFrame(phenotype_results)
    pheno_df.to_csv(BENCHMARK_DIR / "detour_penalty_by_phenotype.csv", index=False)
    
    return pheno_df


# ============================================================================
# 5. VISUALIZACIÓN PUBLICATION-QUALITY
# ============================================================================

def generate_benchmark_figures(results_df, k_sensitivity_df, phenotype_df):
    """
    Genera figura de 4 paneles para el paper.
    
    Panel A: Boxplot Geodésica vs Euclidiana por muestra
    Panel B: Detour penalty por fenotipo (Desert vs Excluded)
    Panel C: Sensibilidad a k (ratio estable)
    Panel D: Correlación distancia Euclidiana vs Detour Penalty
    """
    print(f"\n{'='*70}")
    print(f"GENERANDO FIGURAS")
    print(f"{'='*70}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # ---- Panel A: Geodésica vs Euclidiana ----
    ax = axes[0, 0]
    
    geo_col = f'geodesic_k{K_DEFAULT}_mean'
    if geo_col in results_df.columns:
        plot_data = results_df[['euclidean_mean', geo_col]].dropna()
        plot_data.columns = ['Euclidiana', 'Geodésica']
        
        # Melt para boxplot
        plot_melted = plot_data.melt(var_name='Métrica', value_name='Distancia')
        
        colors = ['#3498db', '#e74c3c']
        sns.boxplot(data=plot_melted, x='Métrica', y='Distancia',
                   palette=colors, ax=ax, width=0.5)
        
        # Líneas pareadas (conectar mismas muestras)
        for i in range(min(len(plot_data), 30)):  # Max 30 líneas para claridad
            ax.plot([0, 1],
                   [plot_data.iloc[i]['Euclidiana'], plot_data.iloc[i]['Geodésica']],
                   color='gray', alpha=0.2, linewidth=0.5)
        
        # Significancia
        if 'wilcoxon_pval' in results_df.columns:
            median_p = results_df['wilcoxon_pval'].median()
            sig_text = '***' if median_p < 0.001 else '**' if median_p < 0.01 else '*' if median_p < 0.05 else 'ns'
            y_max = plot_melted['Distancia'].max() * 1.05
            ax.plot([0, 1], [y_max, y_max], 'k-', linewidth=1)
            ax.text(0.5, y_max * 1.02, sig_text, ha='center', fontsize=14, fontweight='bold')
    
    ax.set_title('A. Geodésica vs Euclidiana (hop count)', fontweight='bold')
    ax.set_ylabel('Distancia media al cDC1 más cercano')
    ax.grid(axis='y', alpha=0.3)
    
    # ---- Panel B: Detour penalty por fenotipo ----
    ax = axes[0, 1]
    
    desert_col = 'detour_Immune_Desert_mean'
    excluded_col = 'detour_Immune_Excluded_mean'
    
    if desert_col in results_df.columns and excluded_col in results_df.columns:
        desert_vals = results_df[desert_col].dropna()
        excluded_vals = results_df[excluded_col].dropna()
        
        bp_data = pd.DataFrame({
            'Fenotipo': ['Desert'] * len(desert_vals) + ['Excluded'] * len(excluded_vals),
            'Detour Penalty': list(desert_vals) + list(excluded_vals)
        })
        
        colors_pheno = ['#f39c12', '#9b59b6']
        sns.boxplot(data=bp_data, x='Fenotipo', y='Detour Penalty',
                   palette=colors_pheno, ax=ax, width=0.5)
        sns.stripplot(data=bp_data, x='Fenotipo', y='Detour Penalty',
                     color='black', alpha=0.4, size=3, ax=ax)
        
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, 
                   label='Sin detour (ratio=1)')
        ax.legend(fontsize=9)
    
    ax.set_title('B. Detour Penalty por Fenotipo', fontweight='bold')
    ax.set_ylabel('Detour Penalty (Geodésica / Euclidiana)')
    ax.grid(axis='y', alpha=0.3)
    
    # ---- Panel C: Sensibilidad a k ----
    ax = axes[1, 0]
    
    if len(k_sensitivity_df) > 0:
        ax.errorbar(
            k_sensitivity_df['k'],
            k_sensitivity_df['geo_euc_ratio_mean'],
            yerr=k_sensitivity_df['geo_euc_ratio_std'],
            marker='o', color='#2ecc71', linewidth=2, markersize=8,
            capsize=5, capthick=2
        )
        
        # Banda de referencia ±10%
        mean_ratio = k_sensitivity_df['geo_euc_ratio_mean'].mean()
        ax.axhspan(mean_ratio * 0.9, mean_ratio * 1.1,
                  alpha=0.15, color='green', label='±10% banda')
        ax.axhline(y=mean_ratio, color='green', linestyle=':', alpha=0.5)
        ax.legend(fontsize=9)
    
    ax.set_title('C. Estabilidad del Ratio ante k', fontweight='bold')
    ax.set_xlabel('k (vecinos en grafo espacial)')
    ax.set_ylabel('Ratio Geodésica / Euclidiana')
    ax.set_xticks(K_VALUES)
    ax.grid(alpha=0.3)
    
    # ---- Panel D: Correlación Detour vs distancia euclidiana ----
    ax = axes[1, 1]
    
    detour_col_d = f'detour_k{K_DEFAULT}_mean'
    if detour_col_d in results_df.columns:
        valid = results_df[detour_col_d].notna()
        if valid.sum() > 5:
            x = results_df.loc[valid, 'euclidean_mean']
            y = results_df.loc[valid, detour_col_d]
            
            ax.scatter(x, y, c='#e74c3c', alpha=0.6, s=40, edgecolors='white')
            
            # Línea de tendencia
            if len(x) > 3:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), '--', color='black', alpha=0.5)
                
                r, p_val = stats.spearmanr(x, y)
                ax.text(0.05, 0.95, f'ρ = {r:.3f}\np = {p_val:.2e}',
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_title('D. Distancia Euclidiana vs Detour', fontweight='bold')
    ax.set_xlabel('Distancia Euclidiana media')
    ax.set_ylabel('Detour Penalty media')
    ax.grid(alpha=0.3)
    
    plt.suptitle('Benchmark: Distancia Geodésica vs Euclidiana en TNBC\n'
                 'Grafo topológico puro (hop count) — sin ponderación por identidad celular',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    fig_path = BENCHMARK_DIR / "figures" / "Fig_geodesic_benchmark.pdf"
    fig.savefig(fig_path, format='pdf')
    fig.savefig(str(fig_path).replace('.pdf', '.png'), format='png')
    plt.close(fig)
    
    print(f"✓ Figura guardada: {fig_path}")
    
    return fig_path


# ============================================================================
# 6. RESUMEN EJECUTIVO
# ============================================================================

def print_executive_summary(results_df, k_sensitivity_df, phenotype_df):
    """Imprime resumen con interpretación para el manuscrito."""
    
    print(f"\n{'='*70}")
    print(f"RESUMEN EJECUTIVO - GEODESIC BENCHMARK")
    print(f"{'='*70}")
    
    geo_col = f'geodesic_k{K_DEFAULT}_mean'
    
    if geo_col in results_df.columns:
        valid = results_df[geo_col].notna() & results_df['euclidean_mean'].notna()
        ratio = (results_df.loc[valid, geo_col] / results_df.loc[valid, 'euclidean_mean'])
        
        print(f"\n  1. DISTANCIAS ABSOLUTAS:")
        print(f"     Euclidiana media: {results_df.loc[valid, 'euclidean_mean'].mean():.2f}")
        print(f"     Geodésica media:  {results_df.loc[valid, geo_col].mean():.2f} (hops)")
        print(f"     Ratio Geo/Euc:    {ratio.mean():.2f}× ± {ratio.std():.2f}")
    
    if 'wilcoxon_pval' in results_df.columns:
        sig_pct = (results_df['wilcoxon_significant'] == True).mean() * 100
        print(f"\n  2. SIGNIFICANCIA ESTADÍSTICA:")
        print(f"     Muestras con Geo > Euc (p<0.05): {sig_pct:.0f}%")
        print(f"     p-valor mediano: {results_df['wilcoxon_pval'].median():.2e}")
    
    if len(k_sensitivity_df) > 1:
        cv = k_sensitivity_df['geo_euc_ratio_mean'].std() / \
             k_sensitivity_df['geo_euc_ratio_mean'].mean() * 100
        print(f"\n  3. ROBUSTEZ ANTE k:")
        print(f"     CV del ratio: {cv:.1f}%")
        print(f"     {'✓ k-INSENSITIVO' if cv < 15 else '⚠ k-DEPENDIENTE'}")
    
    desert_col = 'detour_Immune_Desert_mean'
    excluded_col = 'detour_Immune_Excluded_mean'
    if desert_col in results_df.columns and excluded_col in results_df.columns:
        d_mean = results_df[desert_col].mean()
        e_mean = results_df[excluded_col].mean()
        print(f"\n  4. DETOUR POR FENOTIPO:")
        print(f"     Desert:   {d_mean:.3f}")
        print(f"     Excluded: {e_mean:.3f}")
        if e_mean > d_mean:
            print(f"     → Excluded tiene {(e_mean/d_mean - 1)*100:.0f}% más detour")
            print(f"     → CONFIRMA barrera física SOLO en Excluded")
    


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Punto de entrada principal del benchmark."""
    
    start_time = time.time()
    
    print("="*70)
    print("GEODESIC BENCHMARK - Comparación Formal Geo vs Euc")
    print("  Método: TOPOLÓGICO (hop count, sin pesos CAF)")
    print("  Correcciones: H1 circularidad, H2 consistencia, H11 detour")
    print("="*70)
    print(f"  Directorio de salida: {BENCHMARK_DIR}")
    
    # --- Cargar datos ---
    print("\n1. Cargando datos...")
    
    # Intentar múltiples paths (el pipeline genera diferentes nombres)
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
            print(f"   ✓ Spots: {adata.n_obs:,}, Genes: {adata.n_vars:,}")
            print(f"   ✓ Muestras: {adata.obs['sample_id'].nunique()}")
            break
    
    if adata is None:
        print("❌ ERROR: No se encontró archivo de datos.")
        print(f"   Buscados: {[str(p) for p in possible_paths]}")
        sys.exit(1)
    
    # Verificar que tiene fenotipos
    if 'Phenotype' not in adata.obs.columns:
        # Intentar normalizar si existe con otro nombre
        if 'phenotype' in adata.obs.columns:
            adata.obs['Phenotype'] = adata.obs['phenotype']
        else:
            print("❌ ERROR: Sin columna 'Phenotype'. Ejecuta phenotype_classifier.py primero.")
            sys.exit(1)
    
    # Verificar coordenadas espaciales
    if 'spatial' not in adata.obsm:
        print("⚠ Reconstruyendo coordenadas espaciales por muestra...")
        spatial_keys = [k for k in adata.obsm.keys() if 'spatial' in k.lower()]
        if spatial_keys:
            adata.obsm['spatial'] = adata.obsm[spatial_keys[0]]
            print(f"  ✓ Usando {spatial_keys[0]}")
        else:
            print("❌ ERROR: Sin coordenadas espaciales en obsm")
            sys.exit(1)
    
    # FIX H13 (parcial): Verificar escala de coordenadas
    coords_sample = adata.obsm['spatial']
    coord_range = np.ptp(coords_sample, axis=0)  # range por eje
    print(f"\n   Rango coordenadas: X={coord_range[0]:.0f}, Y={coord_range[1]:.0f}")
    if coord_range.max() > 10000:
        print(f"  Coordenadas en alta resolución (píxeles). "
              f"max_distance={MAX_DISTANCE} hops cubre ampliamente la muestra.")
    elif coord_range.max() < 10:
        print(f"   Coordenadas normalizadas. Verificar que escala es correcta.")
    
    print(f"\n   Fenotipos disponibles:")
    for pheno, count in adata.obs['Phenotype'].value_counts().items():
        pct = count / len(adata) * 100
        print(f"     {pheno:25s}: {count:>6,} ({pct:.1f}%)")
    
    # --- Ejecutar benchmark ---
    print("\n2. Ejecutando benchmark...")
    results_df = run_full_benchmark(adata)
    
    if len(results_df) == 0:
        print("ERROR: Sin resultados. Verifica columnas de abundancia.")
        sys.exit(1)
    
    # --- Sensibilidad a k ---
    print("\n3. Analizando sensibilidad a k...")
    k_df = analyze_k_sensitivity(results_df)
    
    # --- Detour por fenotipo ---
    print("\n4. Analizando detour por fenotipo...")
    pheno_df = analyze_detour_by_phenotype(results_df)
    
    # --- Figuras ---
    print("\n5. Generando figuras...")
    generate_benchmark_figures(results_df, k_df, pheno_df)
    
    # --- Resumen ---
    print_executive_summary(results_df, k_df, pheno_df)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f" BENCHMARK COMPLETADO en {elapsed/60:.1f} minutos")
    print(f"{'='*70}")
    print(f"  Archivos generados en: {BENCHMARK_DIR}")
    print(f"  → geodesic_vs_euclidean_comparison.csv")
    print(f"  → k_sensitivity_geodesic.csv")
    print(f"  → detour_penalty_by_phenotype.csv")
    print(f"  → figures/Fig_geodesic_benchmark.pdf")


if __name__ == '__main__':
    main()
