"""
================================================================================
ADICIONES A spatial_analysis_v2.py — v2.4 → v2.5
================================================================================
PREREQUISITO: geodesic_benchmark.py importa estas funciones.
Sin estas adiciones, el benchmark NO puede ejecutarse.

CAMBIOS:
  1. build_spatial_graph() → parametrizar n_neighbors (antes hardcoded k=6)
  2. NUEVA: calculate_euclidean_distances() — baseline para benchmark
  3. NUEVA: calculate_detour_penalty() — ratio geodésica/euclidiana
  4. NUEVA: run_multi_k_geodesic() — geodésica con k=[4,6,8,10]
  5. REFACTOR: run_spatial_analysis_v2() → acepta k_neighbors como parámetro

CORRECCIONES APLICADAS:
  1. KDTree k=2 en calculate_euclidean_distances() para evitar
       autocorrelación cuando un spot contiene fuente + objetivo (dist=0).
  2. Assert parcial en run_spatial_analysis_v2() para verificar que
       la escala de coordenadas espaciales es razonable.
================================================================================
"""

# ============================================================================
# IMPORTS NUEVOS (añadir a la sección de imports existente)
# ============================================================================
# Estos imports son ADICIONALES a los que ya tiene spatial_analysis_v2.py

from scipy.spatial import KDTree
from collections import defaultdict
import logging
import json

logger = logging.getLogger(__name__)


# ============================================================================
# CAMBIO 1: Modificar build_spatial_graph()
# ============================================================================
# ANTES (v2.4): n_neighbors=6 hardcoded
# AHORA (v2.5): n_neighbors es parámetro, default=6 para backward compatibility

def build_spatial_graph(
    adata: 'ad.AnnData',
    n_neighbors: int = 6,
    coord_type: str = 'generic',
) -> 'ad.AnnData':
    """
    Construye el grafo de vecindad espacial usando squidpy.
    
    CAMBIO v2.5: n_neighbors es ahora parametrizable (antes hardcoded k=6).
    Esto permite a geodesic_benchmark.py testear k=[4,6,8,10].
    
    Parameters
    ----------
    adata : AnnData
        Datos espaciales con coordenadas en .obsm['spatial']
    n_neighbors : int
        Número de vecinos en el grafo k-NN. Default=6 (hexagonal Visium).
        geodesic_benchmark.py llamará con k=4,6,8,10.
    coord_type : str
        Tipo de coordenadas para squidpy.
        
    Returns
    -------
    adata : AnnData con .obsp['connectivities'] y .obsp['distances']
    """
    import squidpy as sq
    
    # Si el adata tiene múltiples muestras, construir grafo por muestra
    # para evitar "teleportación" entre muestras
    if 'sample_id' in adata.obs.columns:
        samples = adata.obs['sample_id'].unique()
        
        if len(samples) > 1:
            print(f"  Construyendo grafos por muestra (k={n_neighbors})...")
            _build_per_sample_graphs(adata, n_neighbors, coord_type)
            return adata
    
    # Muestra única: construir grafo directo
    sq.gr.spatial_neighbors(
        adata,
        n_neighs=n_neighbors,
        coord_type=coord_type,
    )
    
    print(f"  Grafo espacial construido: k={n_neighbors}, "
          f"spots={adata.n_obs:,}")
    
    return adata


def _build_per_sample_graphs(
    adata: 'ad.AnnData',
    n_neighbors: int = 6,
    coord_type: str = 'generic',
):
    """
    Construye grafos espaciales POR MUESTRA para evitar teleportación.
    
    CRÍTICO: Sin library_key, squidpy conecta spots de muestras distintas
    que están numéricamente cerca pero físicamente en tejidos diferentes.
    
    Parameters
    ----------
    adata : AnnData con columna 'sample_id'
    n_neighbors : int — k para k-NN
    coord_type : str — tipo de coordenadas
    """
    import squidpy as sq
    from scipy import sparse
    import numpy as np
    
    samples = adata.obs['sample_id'].unique()
    n_total = adata.n_obs
    
    # Matrices globales vacías
    # Acumuladores SEPARADOS para conn y dist
    rows, cols, data_conn = [], [], []
    dist_rows, dist_cols, data_dist = [], [], []
    
    # Mapeo de índice local → global
    global_indices = np.arange(n_total)
    
    for sample_id in samples:
        mask = adata.obs['sample_id'] == sample_id
        local_adata = adata[mask].copy()
        
        if local_adata.n_obs < n_neighbors + 1:
            continue
        
        # Construir grafo local
        try:
            sq.gr.spatial_neighbors(
                local_adata,
                n_neighs=n_neighbors,
                coord_type=coord_type,
            )
        except Exception as e:
            print(f"  ⚠ {sample_id}: Error construyendo grafo (k={n_neighbors}): {e}")
            continue
        
        # Obtener matrices locales
        conn_local = local_adata.obsp['connectivities']
        dist_local = local_adata.obsp['distances']
        
        # Mapear índices locales a globales
        global_idx = global_indices[mask.values]
        
        # Convertir a COO para iterar eficientemente
        conn_coo = sparse.coo_matrix(conn_local)
        dist_coo = sparse.coo_matrix(dist_local)
        
        for r, c, v in zip(conn_coo.row, conn_coo.col, conn_coo.data):
            rows.append(global_idx[r])
            cols.append(global_idx[c])
            data_conn.append(v)
        
        # Acumular en dist_rows/dist_cols GLOBALES
        # ANTES: dist_coo_rows/cols se sobreescribían cada iteración del loop.
        #        Solo el último sample tenía índices correctos, pero data_dist
        #        contenía datos de TODOS los samples → CSR corrupta.
        for r, c, v in zip(dist_coo.row, dist_coo.col, dist_coo.data):
            dist_rows.append(global_idx[r])
            dist_cols.append(global_idx[c])
            data_dist.append(v)
    
    # Construir matrices globales sparse
    if len(rows) > 0:
        adata.obsp['connectivities'] = sparse.csr_matrix(
            (data_conn, (rows, cols)), shape=(n_total, n_total)
        )
        # Usar acumuladores globales dist_rows/dist_cols
        if len(data_dist) > 0:
            adata.obsp['distances'] = sparse.csr_matrix(
                (data_dist, (dist_rows, dist_cols)), 
                shape=(n_total, n_total)
            )
        else:
            adata.obsp['distances'] = adata.obsp['connectivities'].copy()
    
    print(f"  Grafos por muestra construidos: k={n_neighbors}, "
          f"{len(samples)} muestras, {n_total:,} spots")


# ============================================================================
# calculate_euclidean_distances()
# ============================================================================

def calculate_euclidean_distances(
    adata: 'ad.AnnData',
    source_cell_type: str = 'cDC1',
    target_cell_type: str = 'Tumor',
    abundance_percentile: float = 75.0,
    min_threshold: float = 0.1,
) -> dict:
    """
    Calcula distancias EUCLIDIANAS directas (baseline para benchmark).
    
    A diferencia de la geodésica, esta métrica ignora las barreras físicas
    del tejido y calcula la distancia en línea recta. Si la geodésica
    y la euclidiana difieren sustancialmente → la barrera de CAFs importa.
    
    Parameters
    ----------
    adata : AnnData
        Debe tener .obsm['spatial'] y columnas de abundancia celular
    source_cell_type : str
        Tipo celular origen (e.g., 'cDC1'). Buscará columna q05 o means.
    target_cell_type : str
        Tipo celular destino (e.g., 'Tumor')
    abundance_percentile : float
        Percentil para definir "alta abundancia"
    min_threshold : float
        Umbral mínimo de abundancia
        
    Returns
    -------
    dict : {phenotype: [distances]} — distancias euclidianas por fenotipo
    """
    import numpy as np
    from scipy.spatial import KDTree
    
    results = defaultdict(list)
    
    # Verificar coordenadas espaciales
    if 'spatial' not in adata.obsm:
        print("  No hay coordenadas espaciales (.obsm['spatial'])")
        return results
    
    coords = adata.obsm['spatial']
    
    # Encontrar columnas de abundancia
    source_col = _find_abundance_column(adata, source_cell_type)
    target_col = _find_abundance_column(adata, target_cell_type)
    
    if source_col is None or target_col is None:
        print(f"  No se encontraron columnas para {source_cell_type} o {target_cell_type}")
        return results
    
    # Verificar que existe Phenotype
    if 'Phenotype' not in adata.obs.columns:
        print("  Columna 'Phenotype' no encontrada")
        return results
    
    # Fenotipos a analizar
    phenotypes_of_interest = ['Immune_Desert', 'Immune_Excluded', 'Inflamed']
    
    # Procesar por muestra para evitar distancias inter-muestra
    sample_col = 'sample_id' if 'sample_id' in adata.obs.columns else None
    samples = adata.obs[sample_col].unique() if sample_col else ['all']
    
    for sample_id in samples:
        if sample_col and sample_id != 'all':
            mask = adata.obs[sample_col] == sample_id
            sample_coords = coords[mask.values]
            sample_obs = adata.obs[mask]
        else:
            sample_coords = coords
            sample_obs = adata.obs
        
        # Obtener abundancias
        source_vals = sample_obs[source_col].values.astype(float)
        target_vals = sample_obs[target_col].values.astype(float)
        
        # Umbrales adaptativos
        source_thresh = max(np.percentile(source_vals, abundance_percentile), min_threshold)
        target_thresh = max(np.percentile(target_vals, abundance_percentile), min_threshold)
        
        # Spots con alta abundancia
        high_source = source_vals >= source_thresh
        high_target = target_vals >= target_thresh
        
        if high_source.sum() < 3 or high_target.sum() < 3:
            continue
        
        # Coordenadas de spots fuente y target
        source_coords = sample_coords[high_source]
        target_coords = sample_coords[high_target]
        
        # KDTree para búsqueda eficiente de vecino más cercano
        target_tree = KDTree(target_coords)
        
        # Usar k=2 para evitar autocorrelación (distancia=0)
        # Un spot puede contener tanto el tipo celular fuente como objetivo.
        # Si k=1, KDTree lo empareja consigo mismo → distancia 0 → deflacta media.
        k_query = min(2, len(target_coords))
        
        # Para cada fenotipo, calcular distancias
        for phenotype in phenotypes_of_interest:
            pheno_mask_local = sample_obs['Phenotype'].values == phenotype
            
            # Spots que son del fenotipo Y tienen alta abundancia de source
            combined_mask = pheno_mask_local & high_source
            
            if combined_mask.sum() < 3:
                continue
            
            pheno_source_coords = sample_coords[combined_mask]
            
            # Distancia euclidiana con protección contra dist=0
            distances_raw, _ = target_tree.query(pheno_source_coords, k=k_query)
            
            if k_query == 1:
                distances = distances_raw.ravel()
            else:
                # Para cada spot, tomar segundo vecino si el primero es sí mismo
                distances = np.empty(len(pheno_source_coords))
                for idx in range(len(pheno_source_coords)):
                    if distances_raw[idx, 0] > 1e-10:
                        distances[idx] = distances_raw[idx, 0]
                    else:
                        distances[idx] = distances_raw[idx, 1]
            
            results[phenotype].extend(distances.tolist())
    
    # Log
    for pheno, dists in results.items():
        if len(dists) > 0:
            print(f"  Euclidiana {pheno}: n={len(dists)}, "
                  f"mean={np.mean(dists):.2f} ± {np.std(dists):.2f}")
    
    return results


# ============================================================================
# calculate_detour_penalty()
# ============================================================================

def calculate_detour_penalty(
    geodesic_distances: dict,
    euclidean_distances: dict,
) -> dict:
    """
    Calcula el "detour penalty" = geodésica / euclidiana por fenotipo.
    
    INTERPRETACIÓN CLAVE:
    - detour ≈ 1.0 → no hay barrera (distancia geodésica ≈ euclidiana)
    - detour >> 1.0 → hay barrera física que obliga a rodear
    
    HIPÓTESIS:
    - Excluded: detour ALTO (la barrera CAF obliga a rodear)
    - Desert: detour BAJO (sin barrera, solo silenciamiento)
    
    Parameters
    ----------
    geodesic_distances : dict
        {phenotype: [distances]} de distancias geodésicas
    euclidean_distances : dict
        {phenotype: [distances]} de distancias euclidianas
        
    Returns
    -------
    dict : {
        'per_phenotype': {phenotype: {'mean_detour', 'std', 'n'}},
        'test_result': {stat, pval, effect_size},
        'interpretation': str
    }
    """
    import numpy as np
    from scipy.stats import mannwhitneyu
    
    results = {
        'per_phenotype': {},
        'test_result': {},
        'interpretation': ''
    }
    
    for phenotype in geodesic_distances:
        geo = np.array(geodesic_distances.get(phenotype, []))
        euc = np.array(euclidean_distances.get(phenotype, []))
        
        if len(geo) == 0 or len(euc) == 0:
            continue
        
        # Emparejar por tamaño mínimo (los arrays pueden diferir en n)
        n_min = min(len(geo), len(euc))
        
        # No podemos emparejar spot-a-spot directamente porque se calculan
        # en marcos diferentes. Usamos estadísticas de distribución.
        
        # Detour como ratio de medias
        mean_geo = np.mean(geo)
        mean_euc = np.mean(euc)
        
        if mean_euc > 0:
            mean_detour = mean_geo / mean_euc
        else:
            mean_detour = np.nan
        
        results['per_phenotype'][phenotype] = {
            'mean_geodesic': float(mean_geo),
            'mean_euclidean': float(mean_euc),
            'mean_detour_ratio': float(mean_detour),
            'n_geodesic': len(geo),
            'n_euclidean': len(euc),
        }
        
        print(f"  Detour {phenotype}: geo={mean_geo:.2f}, euc={mean_euc:.2f}, "
              f"ratio={mean_detour:.3f}")
    
    # Test estadístico: ¿detour de Excluded > detour de Desert?
    desert_data = results['per_phenotype'].get('Immune_Desert', {})
    excluded_data = results['per_phenotype'].get('Immune_Excluded', {})
    
    if desert_data and excluded_data:
        desert_detour = desert_data.get('mean_detour_ratio', np.nan)
        excluded_detour = excluded_data.get('mean_detour_ratio', np.nan)
        
        # Test con distribuciones brutas
        geo_desert = np.array(geodesic_distances.get('Immune_Desert', []))
        euc_desert = np.array(euclidean_distances.get('Immune_Desert', []))
        geo_excluded = np.array(geodesic_distances.get('Immune_Excluded', []))
        euc_excluded = np.array(euclidean_distances.get('Immune_Excluded', []))
        
        if len(geo_desert) > 10 and len(geo_excluded) > 10:
            # Mann-Whitney sobre distancias geodésicas
            stat_geo, pval_geo = mannwhitneyu(
                geo_excluded, geo_desert, alternative='greater'
            )
            # Mann-Whitney sobre distancias euclidianas
            stat_euc, pval_euc = mannwhitneyu(
                euc_excluded, euc_desert, alternative='greater'
            )
            
            results['test_result'] = {
                'geodesic_MW_stat': float(stat_geo),
                'geodesic_MW_pval': float(pval_geo),
                'euclidean_MW_stat': float(stat_euc),
                'euclidean_MW_pval': float(pval_euc),
                'desert_detour_ratio': float(desert_detour),
                'excluded_detour_ratio': float(excluded_detour),
            }
            
            # Interpretación
            if pval_geo < 0.05 and pval_euc >= 0.05:
                results['interpretation'] = (
                    "STRONG EVIDENCE: Geodesic distances differ significantly "
                    "between phenotypes (p<0.05) while Euclidean do not. "
                    "This proves the CAF barrier creates detour, not just "
                    "spatial separation."
                )
            elif pval_geo < 0.05 and pval_euc < 0.05:
                # Ambas significativas — comparar effect sizes
                if excluded_detour > desert_detour * 1.1:
                    results['interpretation'] = (
                        "MODERATE EVIDENCE: Both metrics significant, but "
                        f"Excluded detour ({excluded_detour:.2f}) > "
                        f"Desert detour ({desert_detour:.2f}), indicating "
                        "barrier-mediated isolation beyond simple spatial "
                        "separation."
                    )
                else:
                    results['interpretation'] = (
                        "WEAK EVIDENCE: Both metrics detect differences. "
                        "Geodesic adds marginal information over Euclidean."
                    )
            else:
                results['interpretation'] = (
                    "INCONCLUSIVE: Geodesic does not significantly "
                    "outperform Euclidean for this analysis."
                )
            
            print(f"\n  === DETOUR PENALTY TEST ===")
            print(f"  Geodesic p={pval_geo:.4e} | Euclidean p={pval_euc:.4e}")
            print(f"  Desert ratio={desert_detour:.3f} | Excluded ratio={excluded_detour:.3f}")
            print(f"  → {results['interpretation'][:80]}...")
    
    return results


# ============================================================================
# run_multi_k_geodesic()
# ============================================================================

def run_multi_k_geodesic(
    adata: 'ad.AnnData',
    k_values: list = [4, 6, 8, 10],
    source_cell_type: str = 'cDC1',
    target_cell_type: str = 'Tumor',
    abundance_percentile: float = 75.0,
    min_threshold: float = 0.1,
) -> dict:
    """
    Ejecuta análisis geodésico con múltiples valores de k.
    
    Demuestra que el ratio Excluded/Desert es INSENSITIVO al parámetro k,
    es decir, no es un artefacto de elegir k=6.
    
    Parameters
    ----------
    adata : AnnData
        Datos con deconvolución y fenotipos
    k_values : list
        Valores de k a testear
    source_cell_type, target_cell_type : str
        Tipos celulares para la distancia
    abundance_percentile, min_threshold : float
        Parámetros de umbral
        
    Returns
    -------
    dict : {
        k: {
            'ratio_excluded_desert': float,
            'pval': float,
            'desert_mean': float,
            'excluded_mean': float
        }
    }
    """
    import numpy as np
    import squidpy as sq
    from scipy.sparse.csgraph import shortest_path
    from scipy.stats import mannwhitneyu
    
    results = {}
    
    print(f"\n  === MULTI-K GEODESIC ANALYSIS ===")
    print(f"  Testing k = {k_values}")
    
    for k in k_values:
        print(f"\n  --- k = {k} ---")
        
        # Hacer copia para no sobreescribir el grafo original
        adata_k = adata.copy()
        
        # Construir grafo con este k
        build_spatial_graph(adata_k, n_neighbors=k)
        
        # Calcular geodésicas por fenotipo
        geodesic_results = _calculate_geodesic_per_phenotype(
            adata_k,
            source_cell_type=source_cell_type,
            target_cell_type=target_cell_type,
            abundance_percentile=abundance_percentile,
            min_threshold=min_threshold,
        )
        
        # Extraer Desert vs Excluded
        desert_dists = np.array(geodesic_results.get('Immune_Desert', []))
        excluded_dists = np.array(geodesic_results.get('Immune_Excluded', []))
        
        if len(desert_dists) > 10 and len(excluded_dists) > 10:
            ratio = np.mean(excluded_dists) / np.mean(desert_dists) if np.mean(desert_dists) > 0 else np.nan
            stat, pval = mannwhitneyu(excluded_dists, desert_dists, alternative='greater')
            
            results[k] = {
                'ratio_excluded_desert': float(ratio),
                'pval': float(pval),
                'desert_mean': float(np.mean(desert_dists)),
                'desert_std': float(np.std(desert_dists)),
                'excluded_mean': float(np.mean(excluded_dists)),
                'excluded_std': float(np.std(excluded_dists)),
                'n_desert': len(desert_dists),
                'n_excluded': len(excluded_dists),
            }
            
            sig = "✅" if pval < 0.05 else "⚠️"
            print(f"  k={k}: ratio={ratio:.2f}x, p={pval:.2e} {sig}")
        else:
            results[k] = {
                'ratio_excluded_desert': np.nan,
                'pval': np.nan,
                'error': f'Insufficient data (Desert={len(desert_dists)}, Excluded={len(excluded_dists)})'
            }
            print(f"  k={k}: Datos insuficientes")
        
        # Liberar memoria de la copia
        del adata_k
    
    # Resumen
    print(f"\n  === MULTI-K SUMMARY ===")
    ratios = [v['ratio_excluded_desert'] for v in results.values() 
              if not np.isnan(v.get('ratio_excluded_desert', np.nan))]
    if ratios:
        print(f"  Ratios: {[f'{r:.2f}' for r in ratios]}")
        print(f"  Range: {min(ratios):.2f} - {max(ratios):.2f}")
        print(f"  CV: {np.std(ratios)/np.mean(ratios)*100:.1f}%")
        
        if np.std(ratios)/np.mean(ratios) < 0.15:
            print(f"  → INSENSITIVO a k (CV < 15%)")
        else:
            print(f"  → SENSITIVO a k (CV ≥ 15%)")
    
    return results


# ============================================================================
# HELPER: _calculate_geodesic_per_phenotype()
# ============================================================================

def _calculate_geodesic_per_phenotype(
    adata: 'ad.AnnData',
    source_cell_type: str = 'cDC1',
    target_cell_type: str = 'Tumor',
    abundance_percentile: float = 75.0,
    min_threshold: float = 0.1,
) -> dict:
    """
    Calcula distancias geodésicas por fenotipo (lógica extraída del 
    código existente de spatial_analysis_v2.py para reutilización).
    
    Returns: {phenotype: [distances]}
    """
    import numpy as np
    from scipy.sparse.csgraph import shortest_path
    
    results = defaultdict(list)
    
    # Encontrar columnas de abundancia
    source_col = _find_abundance_column(adata, source_cell_type)
    target_col = _find_abundance_column(adata, target_cell_type)
    
    if source_col is None or target_col is None:
        return results
    
    if 'Phenotype' not in adata.obs.columns:
        return results
    
    phenotypes = ['Immune_Desert', 'Immune_Excluded', 'Inflamed']
    sample_col = 'sample_id' if 'sample_id' in adata.obs.columns else None
    samples = adata.obs[sample_col].unique() if sample_col else ['all']
    
    for sample_id in samples:
        if sample_col and sample_id != 'all':
            mask = adata.obs[sample_col] == sample_id
            sample_indices = np.where(mask.values)[0]
        else:
            sample_indices = np.arange(adata.n_obs)
        
        if len(sample_indices) < 10:
            continue
        
        # Obtener submatriz de distancias para esta muestra
        if 'connectivities' not in adata.obsp:
            continue
            
        # Usar la submatriz de la muestra
        conn_sub = adata.obsp['connectivities'][np.ix_(sample_indices, sample_indices)]
        
        # Preparar matriz para Dijkstra
        dist_matrix = conn_sub.copy()
        dist_matrix.data = np.ones_like(dist_matrix.data)  # Unweighted (hop count)
        
        # Abundancias locales
        source_vals = adata.obs.iloc[sample_indices][source_col].values.astype(float)
        target_vals = adata.obs.iloc[sample_indices][target_col].values.astype(float)
        phenotype_vals = adata.obs.iloc[sample_indices]['Phenotype'].values
        
        # Umbrales adaptativos
        source_thresh = max(np.percentile(source_vals, abundance_percentile), min_threshold)
        target_thresh = max(np.percentile(target_vals, abundance_percentile), min_threshold)
        
        high_source = source_vals >= source_thresh
        high_target = target_vals >= target_thresh
        
        if high_source.sum() < 3 or high_target.sum() < 3:
            continue
        
        # Índices locales (dentro de la submatriz de la muestra)
        target_local = np.where(high_target)[0]
        
        for phenotype in phenotypes:
            pheno_mask = (phenotype_vals == phenotype) & high_source
            
            if pheno_mask.sum() < 3:
                continue
            
            source_local = np.where(pheno_mask)[0]
            
            try:
                paths = shortest_path(
                    dist_matrix,
                    method='D',
                    directed=False,
                    indices=source_local
                )
                
                for i in range(len(source_local)):
                    dists_to_targets = paths[i, target_local]
                    finite = dists_to_targets[np.isfinite(dists_to_targets)]
                    if len(finite) > 0:
                        results[phenotype].append(float(np.min(finite)))
                        
            except Exception:
                continue
    
    return results


# ============================================================================
# HELPER: _find_abundance_column()
# ============================================================================

def _find_abundance_column(adata: 'ad.AnnData', cell_type: str) -> str:
    """
    Busca columna de abundancia celular en adata.obs.
    Intenta: q05_{cell_type}, {cell_type}_q05, means_{cell_type}, etc.
    Compatible con las convenciones de Cell2Location.
    
    Returns: nombre de columna o None
    """
    candidates = [
        # Formato (meanscell_/q05cell_ sin underscore)
        f'q05cell_abundance_w_sf_{cell_type}',
        f'q05_cell_abundance_w_sf_{cell_type}',
        f'q05_{cell_type}',
        f'{cell_type}_q05',
        f'meanscell_abundance_w_sf_{cell_type}',
        f'means_cell_abundance_w_sf_{cell_type}',
        f'means_{cell_type}',
        f'{cell_type}_means',
        f'{cell_type}_abundance',
        cell_type,
    ]
    
    for col in candidates:
        if col in adata.obs.columns:
            return col
    
    # Búsqueda fuzzy: cualquier columna que contenga el cell_type
    for col in adata.obs.columns:
        if cell_type.lower() in col.lower():
            return col
    
    return None


# ============================================================================
# FUNCIÓN ACTUALIZADA: run_spatial_analysis_v2()
# ============================================================================
# Añadir k_neighbors como parámetro al wrapper principal.
# El resto de la lógica del wrapper existente se mantiene igual.

def run_spatial_analysis_v2(
    adata: 'ad.AnnData',
    k_neighbors: int = 6,
    abundance_percentile: float = 75.0,
    min_threshold: float = 0.1,
    run_euclidean: bool = False,
    run_multi_k: bool = False,
    k_values: list = None,
) -> 'ad.AnnData':
    """
    Wrapper principal actualizado.
    
    CAMBIOS vs v2.4:
    - k_neighbors es parametrizable (antes hardcoded k=6)
    - Opción run_euclidean para calcular baseline
    - Opción run_multi_k para test de insensibilidad
    
    Parameters
    ----------
    adata : AnnData con deconvolución y fenotipos
    k_neighbors : int — k para grafo espacial (default 6)
    abundance_percentile : float — percentil para alta abundancia
    min_threshold : float — umbral mínimo
    run_euclidean : bool — si True, calcula distancias euclidianas también
    run_multi_k : bool — si True, testea con k=[4,6,8,10]
    k_values : list — override de valores k para multi-k
    
    Returns
    -------
    adata : AnnData con resultados en .uns
    """
    import numpy as np
    from pathlib import Path
    
    print("\n" + "=" * 80)
    print(f"SPATIAL ANALYSIS v2.5 (k={k_neighbors})")
    print("  Correcciones: H9 autocorrelación KDTree, H13 escala coordenadas")
    print("=" * 80)
    
    # FIX H13 (parcial): Verificar escala de coordenadas espaciales
    if 'spatial' in adata.obsm:
        coord_range = np.ptp(adata.obsm['spatial'], axis=0)
        print(f"\n  Rango coordenadas: X={coord_range[0]:.0f}, Y={coord_range[1]:.0f}")
        if coord_range.max() < 1:
            print(" ALERTA: Coordenadas muy pequeñas (posiblemente normalizadas).")
            print("    Verificar que la escala es correcta para el cálculo de distancias.")
        elif coord_range.max() > 50000:
            print(" Coordenadas en alta resolución (píxeles). Esto es normal para Visium.")
    
    # 1. Construir grafo espacial
    print(f"\n[1/4] Construyendo grafo espacial (k={k_neighbors})...")
    build_spatial_graph(adata, n_neighbors=k_neighbors)
    
    # 2. Distancias geodésicas (análisis principal existente)
    print(f"\n[2/4] Calculando distancias geodésicas...")
    geodesic_results = _calculate_geodesic_per_phenotype(
        adata,
        source_cell_type='cDC1',
        target_cell_type='Tumor',
        abundance_percentile=abundance_percentile,
        min_threshold=min_threshold,
    )
    
    # Guardar en .uns
    adata.uns['geodesic_distances'] = {
        k: v for k, v in geodesic_results.items()
    }
    adata.uns['spatial_params'] = {
        'k_neighbors': k_neighbors,
        'abundance_percentile': abundance_percentile,
        'min_threshold': min_threshold,
    }
    
    # Log resultados principales
    desert_dists = np.array(geodesic_results.get('Immune_Desert', []))
    excluded_dists = np.array(geodesic_results.get('Immune_Excluded', []))
    
    if len(desert_dists) > 0 and len(excluded_dists) > 0:
        ratio = np.mean(excluded_dists) / np.mean(desert_dists) if np.mean(desert_dists) > 0 else float('inf')
        from scipy.stats import mannwhitneyu
        stat, pval = mannwhitneyu(excluded_dists, desert_dists, alternative='greater')
        
        print(f"\n  === RESULTADO PRINCIPAL ===")
        print(f"  Desert: {np.mean(desert_dists):.2f} ± {np.std(desert_dists):.2f} hops (n={len(desert_dists)})")
        print(f"  Excluded: {np.mean(excluded_dists):.2f} ± {np.std(excluded_dists):.2f} hops (n={len(excluded_dists)})")
        print(f"  Ratio: {ratio:.2f}x | p-value: {pval:.2e}")
        
        if ratio > 1.2 and pval < 0.05:
            print(f"  VALIDATED: cDC1 geodésicamente más lejos en Excluded")
        
        adata.uns['geodesic_main_result'] = {
            'ratio': float(ratio),
            'pval': float(pval),
            'desert_mean': float(np.mean(desert_dists)),
            'excluded_mean': float(np.mean(excluded_dists)),
        }
    
    # 3. Distancias euclidianas (opcional, para benchmark)
    if run_euclidean:
        print(f"\n[3/4] Calculando distancias euclidianas (baseline)...")
        euclidean_results = calculate_euclidean_distances(
            adata,
            source_cell_type='cDC1',
            target_cell_type='Tumor',
            abundance_percentile=abundance_percentile,
            min_threshold=min_threshold,
        )
        
        adata.uns['euclidean_distances'] = {
            k: v for k, v in euclidean_results.items()
        }
        
        # Calcular detour penalty
        print(f"\n  Calculando detour penalty...")
        detour_results = calculate_detour_penalty(
            geodesic_results, euclidean_results
        )
        adata.uns['detour_penalty'] = detour_results
    else:
        print(f"\n[3/4] Euclidiana: SKIP (run_euclidean=False)")
    
    # 4. Multi-k test (opcional, para benchmark)
    if run_multi_k:
        k_list = k_values or [4, 6, 8, 10]
        print(f"\n[4/4] Multi-k geodesic test (k={k_list})...")
        multi_k_results = run_multi_k_geodesic(
            adata, k_values=k_list,
            abundance_percentile=abundance_percentile,
            min_threshold=min_threshold,
        )
        adata.uns['multi_k_results'] = {
            str(k): v for k, v in multi_k_results.items()
        }
    else:
        print(f"\n[4/4] Multi-k: SKIP (run_multi_k=False)")
    
    print(f"\n{'=' * 80}")
    print(f" SPATIAL ANALYSIS v2.5 COMPLETADO")
    print(f"{'=' * 80}")
    
    return adata

# ============================================================================
# MAIN (para testing independiente)
# ============================================================================

if __name__ == '__main__':
    import scanpy as sc
    import sys
    
    print("=" * 80)
    print("SPATIAL ANALYSIS v2.5 — TESTING INDEPENDIENTE")
    print("=" * 80)
    
    # Detectar paths
    from pathlib import Path
    possible_paths = [
        Path("/home/external/frjimenez/fabian/genoma/data/processed/adata_with_phenotypes.h5ad"),
        Path("/home/external/frjimenez/fabian/genoma/data/processed/adata_with_deconvolution.h5ad"),
        Path("data/processed/adata_with_phenotypes.h5ad"),
    ]
    
    adata_path = None
    for p in possible_paths:
        if p.exists():
            adata_path = p
            break
    
    if adata_path is None:
        print("No se encontró adata. Rutas buscadas:")
        for p in possible_paths:
            print(f"  {p}")
        sys.exit(1)
    
    print(f"\nCargando: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    print(f"  Spots: {adata.n_obs:,}")
    print(f"  Genes: {adata.n_vars:,}")
    
    if 'Phenotype' not in adata.obs.columns:
        print("No hay columna 'Phenotype'. Ejecuta phenotype_classifier.py primero.")
        sys.exit(1)
    
    # Ejecutar con benchmark completo
    adata = run_spatial_analysis_v2(
        adata,
        k_neighbors=6,
        run_euclidean=True,
        run_multi_k=True,
        k_values=[4, 6, 8, 10],
    )
    
    # Guardar resultados
    output_dir = Path("results/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    
    if 'multi_k_results' in adata.uns:
        with open(output_dir / 'multi_k_geodesic_results.json', 'w') as f:
            json.dump(adata.uns['multi_k_results'], f, indent=2, default=str)
        print(f"\n✓ Multi-k results: {output_dir / 'multi_k_geodesic_results.json'}")
    
    if 'detour_penalty' in adata.uns:
        with open(output_dir / 'detour_penalty_results.json', 'w') as f:
            json.dump(adata.uns['detour_penalty'], f, indent=2, default=str)
        print(f"✓ Detour penalty: {output_dir / 'detour_penalty_results.json'}")
    
    print(f"\n{'=' * 80}")
    print(f"TEST COMPLETADO")
    print(f"{'=' * 80}")
