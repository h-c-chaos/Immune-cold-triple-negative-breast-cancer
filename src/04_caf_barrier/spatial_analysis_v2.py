"""
================================================================================
ANALISIS ESPACIAL AVANZADO
================================================================================
FIXES APLICADOS:
1. Filtrado estadístico de valores Infinitos (spots desconectados en grafo).
2. Prevención de crash en Mann-Whitney y cálculos de media.
3. Uso de búsqueda robusta de columnas (Fuzzy matching).
4. Sanity check de threshold: evita que p90 muy bajo seleccione todos los spots.

================================================================================
"""

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import anndata as ad
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu
from typing import Optional, Tuple
import warnings

from config import PATHS

# Importar helpers de mechanism_validation
from mechanism_validation import find_cell_abundance_column, get_abundance_values

warnings.filterwarnings('ignore')

# ============================================================================
# FUNCIONES DE GRAFO Y DISTANCIA
# ============================================================================

def build_spatial_graph(adata: ad.AnnData, n_neighs: int = 6) -> ad.AnnData:
    """
    Construye grafo espacial usando Squidpy.
    
    Usa sample_id como library_key para evitar conexiones entre muestras.
    """
    print("\nConstruyendo grafo espacial...")
    
    if 'spatial' not in adata.obsm:
        raise ValueError("No hay coordenadas espaciales en .obsm['spatial']")
    
    # Verificar sample_id
    if 'sample_id' not in adata.obs.columns:
        print("[WARN] No hay sample_id, construyendo grafo sin separación por muestra")
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighs)
    else:
        n_samples = adata.obs['sample_id'].nunique()
        print(f"  Separando por {n_samples} muestras")
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighs, library_key='sample_id')
    
    n_edges = adata.obsp['spatial_connectivities'].nnz
    print(f"  Grafo construido: {adata.n_obs} nodos, {n_edges} edges")
    
    return adata


def calculate_geodesic_distance_to_cell_type(
    adata: ad.AnnData, 
    target_cell: str, 
    quantile: str = 'q50',
    min_threshold: float = 0.1,
    source_percentile: float = 90
) -> ad.AnnData:
    """
    Calcula distancia geodésica desde cada spot a los hotspots del tipo celular.
    
    Sanity check de threshold para evitar que valores muy bajos
    seleccionen todos los spots como fuentes.
    
    Parameters
    ----------
    adata : AnnData
        Datos con grafo espacial
    target_cell : str
        Tipo celular objetivo (ej. 'cDC1', 'CD8_T')
    quantile : str
        Cuantil de abundancia a usar
    min_threshold : float
        Threshold mínimo para considerar "presencia" (evita p90=0)
    source_percentile : float
        Percentil para definir hotspots (default: 90)
    """
    print(f"\nCalculando distancia geodésica a {target_cell}...")
    
    # 1. Asegurar grafo
    if 'spatial_connectivities' not in adata.obsp:
        adata = build_spatial_graph(adata)
        
    # 2. Identificar fuentes (spots con alta abundancia)
    col = find_cell_abundance_column(adata, target_cell, quantile)
    if not col:
        print(f"[WARN] No se encontró abundancia para {target_cell}")
        adata.obs[f'dist_to_{target_cell}'] = np.inf
        return adata
    
    values = adata.obs[col].values
    
    # Sanity check de threshold
    thresh = np.percentile(values, source_percentile)
    
    if thresh < min_threshold:
        print(f"  [WARN] Threshold p{source_percentile} muy bajo ({thresh:.4f})")
        print(f"         Usando mínimo biológico: {min_threshold}")
        thresh = min_threshold
    else:
        print(f"  Threshold p{source_percentile}: {thresh:.3f}")
    
    # Definir Hotspots
    sources = np.where(values > thresh)[0]
    
    if len(sources) == 0:
        print(f"  [WARN] No hay spots con {target_cell} > {thresh:.3f}")
        adata.obs[f'dist_to_{target_cell}'] = np.inf
        return adata
    
    pct_sources = (len(sources) / adata.n_obs) * 100
    print(f"  Fuentes identificadas: {len(sources)} spots ({pct_sources:.1f}%)")
    
    # Sanity check: no deberían ser demasiados
    if pct_sources > 50:
        print(f"  [WARN] >50% de spots son fuentes - threshold puede ser muy bajo")
        
    # 3. Dijkstra (Distancia Geodésica)
    graph = adata.obsp['spatial_connectivities']
    
    # Convertir a float para dijkstra
    if graph.dtype != np.float64:
        graph = graph.astype(np.float64)
    
    print(f"  Calculando distancias (puede tomar tiempo)...")
    dists_matrix = dijkstra(graph, indices=sources, directed=False)
    
    # Mínima distancia a cualquier fuente
    min_dists = np.min(dists_matrix, axis=0)
    
    # Estadísticas
    n_inf = np.isinf(min_dists).sum()
    if n_inf > 0:
        pct_inf = (n_inf / len(min_dists)) * 100
        print(f"  [INFO] {n_inf} spots desconectados ({pct_inf:.1f}%) - dist=Inf")
    
    finite_dists = min_dists[np.isfinite(min_dists)]
    if len(finite_dists) > 0:
        print(f"  Distancias: mediana={np.median(finite_dists):.1f}, max={finite_dists.max():.1f}")
    
    # Guardar
    adata.obs[f'dist_to_{target_cell}'] = min_dists
    return adata


# ============================================================================
# COMPARACION ESTADISTICA (BLINDADA)
# ============================================================================

def safe_filter_finite(values: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Filtra valores infinitos y NaN de forma segura.
    
    Returns
    -------
    clean_values : np.ndarray
        Valores finitos
    n_dropped : int
        Número de valores eliminados
    """
    mask = np.isfinite(values)
    n_dropped = (~mask).sum()
    return values[mask], n_dropped


def compare_distances_by_phenotype(
    adata: ad.AnnData, 
    dist_col: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compara distancias geodésicas entre fenotipos.
    
    Filtra infinitos antes de calcular estadísticas.
    """
    if verbose:
        print(f"\nComparando distancias ({dist_col}) por fenotipo...")
    
    if dist_col not in adata.obs.columns:
        print(f"[WARN] Columna {dist_col} no encontrada")
        return pd.DataFrame()
    
    if 'Phenotype' not in adata.obs.columns:
        print("[WARN] No hay columna Phenotype")
        return pd.DataFrame()
        
    results = []
    
    for pheno in adata.obs['Phenotype'].unique():
        # Extraer distancias
        mask = adata.obs['Phenotype'] == pheno
        dists = adata.obs.loc[mask, dist_col].values
        
        # FIX v2.4: Filtrar Infinitos y NaNs
        dists_clean, n_dropped = safe_filter_finite(dists)
        
        if n_dropped > 0 and len(dists) > 0:
            pct = (n_dropped / len(dists)) * 100
            if pct > 5 and verbose:
                print(f"  [INFO] {pheno}: {n_dropped} spots desconectados ({pct:.1f}%)")
        
        if len(dists_clean) > 0:
            results.append({
                'Phenotype': pheno,
                'Median_Dist': np.median(dists_clean),
                'Mean_Dist': np.mean(dists_clean),
                'Std_Dist': np.std(dists_clean),
                'Q25_Dist': np.percentile(dists_clean, 25),
                'Q75_Dist': np.percentile(dists_clean, 75),
                'N_spots': len(dists_clean),
                'N_disconnected': n_dropped
            })
            
    return pd.DataFrame(results)


def statistical_test_distances(
    adata: ad.AnnData,
    dist_col: str,
    group1: str = 'Immune_Desert',
    group2: str = 'Immune_Excluded'
) -> dict:
    """
    Test Mann-Whitney para comparar distancias entre dos grupos.
    
    FIX v2.4: Manejo robusto de infinitos y n pequeño.
    """
    print(f"\nTest estadístico: {group1} vs {group2} ({dist_col})")
    
    if dist_col not in adata.obs.columns:
        return {'error': f'Columna {dist_col} no encontrada'}
    
    # Extraer y filtrar
    mask1 = adata.obs['Phenotype'] == group1
    mask2 = adata.obs['Phenotype'] == group2
    
    dists1, dropped1 = safe_filter_finite(adata.obs.loc[mask1, dist_col].values)
    dists2, dropped2 = safe_filter_finite(adata.obs.loc[mask2, dist_col].values)
    
    print(f"  {group1}: n={len(dists1)} (dropped {dropped1})")
    print(f"  {group2}: n={len(dists2)} (dropped {dropped2})")
    
    # Verificar n mínimo
    if len(dists1) < 10 or len(dists2) < 10:
        print("[WARN] n muy pequeño para test robusto")
        return {
            'test': 'Mann-Whitney U',
            'n1': len(dists1),
            'n2': len(dists2),
            'error': 'n < 10'
        }
    
    # Test
    stat, pval = mannwhitneyu(dists1, dists2, alternative='two-sided')
    
    # Effect size (rank-biserial correlation)
    n1, n2 = len(dists1), len(dists2)
    r = 1 - (2 * stat) / (n1 * n2)  # Rank-biserial
    
    result = {
        'test': 'Mann-Whitney U',
        'group1': group1,
        'group2': group2,
        'n1': n1,
        'n2': n2,
        'median1': np.median(dists1),
        'median2': np.median(dists2),
        'U_statistic': stat,
        'p_value': pval,
        'rank_biserial_r': r,
        'significant': pval < 0.05
    }
    
    print(f"  p-value: {pval:.2e}")
    print(f"  Effect size (r): {r:.3f}")
    
    return result


# ============================================================================
# ANALISIS DE VECINDARIO
# ============================================================================

def analyze_neighborhood_composition(
    adata: ad.AnnData,
    cell_types: list = ['CD8_T', 'cDC1', 'CAF', 'Macrophage']
) -> pd.DataFrame:
    """
    Analiza composición del vecindario por fenotipo.
    """
    print("\nAnalizando composición de vecindario...")
    
    if 'Phenotype' not in adata.obs.columns:
        return pd.DataFrame()
    
    results = []
    
    for pheno in ['Immune_Desert', 'Immune_Excluded', 'Inflamed']:
        mask = adata.obs['Phenotype'] == pheno
        n_spots = mask.sum()
        
        if n_spots == 0:
            continue
            
        row = {'Phenotype': pheno, 'N_spots': n_spots}
        
        for ct in cell_types:
            values = get_abundance_values(adata[mask], ct, 'q50')
            if len(values) > 0:
                row[f'{ct}_median'] = np.median(values)
                row[f'{ct}_mean'] = np.mean(values)
                
        results.append(row)
        
    return pd.DataFrame(results)


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def run_spatial_analysis_v2(adata: ad.AnnData) -> ad.AnnData:
    """Pipeline completo de análisis espacial geodésico."""
    print("\n" + "="*80)
    print("ANALISIS ESPACIAL GEODESICO v2.4 (Q1)")
    print("="*80)
    
    # 1. Construir grafo si no existe
    if 'spatial_connectivities' not in adata.obsp:
        adata = build_spatial_graph(adata)
    
    # 2. Calcular distancias a tipos celulares clave
    for ct in ['cDC1', 'CD8_T']:
        adata = calculate_geodesic_distance_to_cell_type(adata, ct)
    
    # 3. Estadísticas por fenotipo
    results_tables = {}
    
    for ct in ['cDC1', 'CD8_T']:
        dist_col = f'dist_to_{ct}'
        df_stats = compare_distances_by_phenotype(adata, dist_col)
        if not df_stats.empty:
            results_tables[f'distances_{ct}'] = df_stats
            
    # 4. Tests estadísticos
    tests_results = []
    for ct in ['cDC1', 'CD8_T']:
        dist_col = f'dist_to_{ct}'
        test = statistical_test_distances(adata, dist_col)
        test['cell_type'] = ct
        tests_results.append(test)
    
    df_tests = pd.DataFrame(tests_results)
    
    # 5. Composición de vecindario
    df_composition = analyze_neighborhood_composition(adata)
    
    # Guardar
    PATHS.create_directories()
    
    for name, df in results_tables.items():
        df.to_csv(PATHS.TABLES_DIR / f'spatial_{name}.csv', index=False)
    
    if not df_tests.empty:
        df_tests.to_csv(PATHS.TABLES_DIR / 'spatial_distance_tests.csv', index=False)
        
    if not df_composition.empty:
        df_composition.to_csv(PATHS.TABLES_DIR / 'neighborhood_composition.csv', index=False)
    
    # Guardar adata actualizado
    out_path = PATHS.PROCESSED_DIR / 'adata_with_spatial.h5ad'
    adata.write_h5ad(out_path)
    print(f"\n[OK] Datos guardados: {out_path}")
    
    print("\n[OK] Análisis espacial completado.")
    
    # Mostrar resumen
    if results_tables:
        print("\n--- Resumen de Distancias ---")
        for name, df in results_tables.items():
            print(f"\n{name}:")
            print(df.to_string(index=False))
    
    return adata


if __name__ == '__main__':
    print("Cargando datos...")
    try:
        # FIX-04b AUDIT v3: Priorizar adata_with_mechanism.h5ad (tiene ratio CXCL9:SPP1)
        mechanism_path = PATHS.PROCESSED_DIR / 'adata_with_mechanism.h5ad'
        pheno_path = PATHS.PROCESSED_DIR / 'adata_with_phenotypes.h5ad'
        deconv_path = PATHS.PROCESSED_DIR / 'adata_with_deconvolution.h5ad'
        
        if mechanism_path.exists():
            print(f"[INFO] Usando: {mechanism_path}")
            adata = sc.read_h5ad(mechanism_path)
        elif pheno_path.exists():
            print(f"[INFO] Usando: {pheno_path}")
            adata = sc.read_h5ad(pheno_path)
        elif deconv_path.exists():
            print(f"[INFO] Usando: {deconv_path}")
            adata = sc.read_h5ad(deconv_path)
        else:
            raise FileNotFoundError("No se encontraron datos procesados")
            
        run_spatial_analysis_v2(adata)
        
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("        Ejecuta primero phenotype_classifier.py")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
