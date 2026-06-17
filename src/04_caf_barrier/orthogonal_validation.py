"""
================================================================================
MÓDULO: ORTHOGONAL VALIDATION — Rompe Circularidad Estadística (P1)
================================================================================
Dos enfoques ortogonales:
  (a) Clustering no supervisado usando SOLO abundancias Cell2Location
      (sin gene scores). Testear si los clusters muestran los patrones predichos.
  (b) Clasificación por contexto espacial: spots en vecindarios uniformemente
      immune-cold (desert pattern) vs spots con vecinos immune-rich pero ellos
      immune-poor (exclusion pattern).


LEE:   adata_with_phenotypes.h5ad (Discovery)
       Requiere: .obsm['q05_cell_abundance_w_sf'] o ['means_cell_abundance_w_sf']
       Requiere: .obs['Phenotype'], .obs['sample_id']
       Requiere: .obsp['spatial_connectivities'] (o se construye)
ESCRIBE: results/orthogonal_validation/
         orthogonal_clustering_comparison.csv
         spatial_context_classification.csv
         orthogonal_summary.json
================================================================================
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.stats import spearmanr, mannwhitneyu, chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from collections import Counter
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from config import PATHS, SIGNATURES, PHENOTYPE_PARAMS, CANONICAL_SIGNATURES
    BASE_DIR = PATHS.BASE_DIR
except ImportError:
    BASE_DIR = Path("/home/external/frjimenez/fabian/genoma")

try:
    from utils_stats import cohens_d_pooled, apply_fdr, safe_mannwhitney
except ImportError:
    raise ImportError("utils_stats.py es REQUERIDO. No se puede ejecutar sin él.")

try:
    from mechanism_validation import find_cell_abundance_column
except ImportError:
    find_cell_abundance_column = None

RANDOM_SEED = 42
OUTPUT_DIR = (PATHS.RESULTS_DIR if 'PATHS' in dir() else BASE_DIR / "results") / "orthogonal_validation"


# ============================================================================
# ENFOQUE A: CLUSTERING CON ABUNDANCIAS CELL2LOCATION (SIN GENE SCORES)
# ============================================================================

def _extract_abundance_matrix(adata: ad.AnnData, quantile: str = 'means') -> Optional[pd.DataFrame]:
    """
    Extrae matriz de abundancias celulares de .obsm como DataFrame.
    
    Usa el quantile especificado. Columnas = tipos celulares.
    Formato real HPC: 'meanscell_abundance_w_sf_{ct}' (sin underscore).
    """
    key_map = {
        'means': 'means_cell_abundance_w_sf',
        'q05': 'q05_cell_abundance_w_sf',
        'q50': 'q50_cell_abundance_w_sf',
    }
    obsm_key = key_map.get(quantile, f'{quantile}_cell_abundance_w_sf')
    
    if obsm_key not in adata.obsm:
        print(f"  [WARN] obsm key '{obsm_key}' no encontrado")
        return None
    
    data = adata.obsm[obsm_key]
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data, index=adata.obs_names)
    
    # Limpiar nombres de columnas (strip prefijos)
    clean_cols = {}
    for c in df.columns:
        clean = c
        for prefix in ['meanscell_abundance_w_sf_', 'means_cell_abundance_w_sf_',
                       'q05cell_abundance_w_sf_', 'q05_cell_abundance_w_sf_',
                       'q50cell_abundance_w_sf_', 'q50_cell_abundance_w_sf_']:
            clean = clean.replace(prefix, '')
        clean_cols[c] = clean
    df = df.rename(columns=clean_cols)
    
    return df


def cluster_by_abundances(
    adata: ad.AnnData,
    n_clusters: int = 5,
    quantile: str = 'means',
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Enfoque A: Clustering K-means usando SOLO abundancias Cell2Location.
    
    INDEPENDIENTE de gene scores. Si los clusters resultantes muestran
    los mismos patrones que la clasificación por gene scores, la
    clasificación no es circular — los patrones existen en los datos.
    
    Returns: (labels, stats_df)
    """
    print("\n" + "=" * 70)
    print("ENFOQUE A: Clustering por Abundancias Cell2Location")
    print("=" * 70)
    
    ab_df = _extract_abundance_matrix(adata, quantile)
    if ab_df is None:
        return None, pd.DataFrame()

    # subsetear a spots frios antes del KMeans
    # Normal_Stroma = 60% de spots -> domina todos los clusters si se incluye
    cold_phenotypes = ['Immune_Desert', 'Immune_Excluded', 'Ambiguous_Cold']
    if 'Phenotype' in adata.obs.columns:
        cold_mask = adata.obs['Phenotype'].isin(cold_phenotypes).values
        n_cold = cold_mask.sum()
        print(f"  Subsetting a spots frios: {n_cold:,} / {adata.n_obs:,} ({100*n_cold/adata.n_obs:.1f}%)")
        if n_cold < 50:
            print("  [WARN] Muy pocos spots frios, usando todos los spots")
            cold_mask = np.ones(adata.n_obs, dtype=bool)
    else:
        cold_mask = np.ones(adata.n_obs, dtype=bool)

    ab_cold = ab_df.iloc[cold_mask]

    print(f"  Matriz de abundancias (frios): {ab_cold.shape[0]:,} spots x {ab_cold.shape[1]} tipos celulares")
    print(f"  Tipos: {', '.join(ab_cold.columns[:8])}...")

    # Estandarizar sobre spots frios
    scaler = StandardScaler()
    ab_scaled = scaler.fit_transform(ab_cold.values)

    # K-means con multiples k
    k_results = []
    for k in [3, 4, 5, 6]:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels_k = km.fit_predict(ab_scaled)
        inertia = km.inertia_
        k_results.append({'k': k, 'inertia': inertia, 'labels': labels_k})
        print(f"  k={k}: inertia={inertia:.0f}")

    # Usar k especificado
    km_final = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels_cold = km_final.fit_predict(ab_scaled)

    # Array full-length alineado con adata (no-frios = cluster -1)
    labels_full = np.full(adata.n_obs, -1, dtype=int)
    labels_full[cold_mask] = labels_cold

    # Estadisticas por cluster
    stats_rows = []
    for cl in range(n_clusters):
        mask = labels_full == cl
        n = mask.sum()
        row = {'cluster': cl, 'n_spots': int(n), 'pct': 100 * n / cold_mask.sum()}
        for ct in ab_cold.columns:
            vals = ab_df.iloc[mask][ct].values if n > 0 else np.array([np.nan])
            row[f'{ct}_median'] = float(np.nanmedian(vals))
        stats_rows.append(row)

    print(chr(10) + "  Clusters (k=" + str(n_clusters) + ", solo spots frios):")
    for r in stats_rows:
        print(f"    Cluster {r['cluster']}: {r['n_spots']:,} spots ({r['pct']:.1f}%)")

    return labels_full, pd.DataFrame(stats_rows)


def compare_clustering_vs_classification(
    adata: ad.AnnData,
    cluster_labels: np.ndarray,
) -> Dict:
    """
    Compara clustering por abundancias vs clasificación por gene scores.
    
    Métricas: ARI (Adjusted Rand Index), NMI (Normalized Mutual Information),
    contingency table, y test chi-cuadrado.
    
    INTERPRETACIÓN:
    - ARI > 0.3: acuerdo moderado → los patrones son reales (no circulares)
    - ARI < 0.1: sin acuerdo → clasificación por gene scores captura algo
      diferente de las abundancias (puede ser circular O complementario)
    - ARI = 1.0: acuerdo perfecto → redundante (sospechoso)
    """
    print("\n" + "-" * 50)
    print("COMPARACIÓN: Clustering vs Clasificación")
    print("-" * 50)
    
    if 'Phenotype' not in adata.obs.columns:
        return {'error': 'No Phenotype column'}
    
    gene_labels = adata.obs['Phenotype'].values
    
    # ARI y NMI
    ari = adjusted_rand_score(gene_labels, cluster_labels)
    nmi = normalized_mutual_info_score(gene_labels, cluster_labels)
    
    print(f"  ARI = {ari:.4f}")
    print(f"  NMI = {nmi:.4f}")
    
    if ari > 0.3:
        interp = "MODERATE agreement — patterns exist independently of gene scores"
    elif ari > 0.1:
        interp = "WEAK agreement — some overlap but classifications capture different aspects"
    else:
        interp = "POOR agreement — clustering by abundances differs from gene-score classification"
    print(f"  Interpretación: {interp}")
    
    # Contingency table
    pheno_cats = sorted(set(gene_labels))
    cluster_cats = sorted(set(cluster_labels))
    
    contingency = pd.crosstab(
        pd.Series(gene_labels, name='Phenotype'),
        pd.Series(cluster_labels, name='Cluster')
    )
    print(f"\n  Contingency table:")
    print(contingency.to_string())
    
    # Chi-squared
    try:
        chi2, p_chi2, dof, expected = chi2_contingency(contingency.values)
        print(f"\n  Chi-squared: χ²={chi2:.1f}, p={p_chi2:.2e}, dof={dof}")
    except Exception:
        chi2, p_chi2 = np.nan, np.nan
    
    # Para cada cluster, encontrar el fenotipo dominante
    cluster_phenotype_mapping = {}
    for cl in cluster_cats:
        cl_mask = cluster_labels == cl
        pheno_in_cluster = gene_labels[cl_mask]
        most_common = Counter(pheno_in_cluster).most_common(1)[0]
        purity = most_common[1] / cl_mask.sum()
        cluster_phenotype_mapping[int(cl)] = {
            'dominant_phenotype': most_common[0],
            'purity': float(purity),
            'n_spots': int(cl_mask.sum()),
        }
        print(f"  Cluster {cl} → {most_common[0]} (purity={purity:.2f})")
    
    # Testear CAF en clusters equivalentes a Desert/Excluded
    results = {
        'ARI': float(ari),
        'NMI': float(nmi),
        'chi2': float(chi2) if np.isfinite(chi2) else None,
        'p_chi2': float(p_chi2) if np.isfinite(p_chi2) else None,
        'interpretation': interp,
        'cluster_mapping': cluster_phenotype_mapping,
    }
    
    return results


def test_caf_in_clusters(
    adata: ad.AnnData,
    cluster_labels: np.ndarray,
    cluster_mapping: Dict,
) -> Dict:
    """
    Test ortogonal CLAVE: ¿Los clusters equivalentes a Desert/Excluded
    muestran la diferencia de CAF predicha?
    """
    print("\n" + "-" * 50)
    print("TEST ORTOGONAL: CAF en clusters análogos a Desert/Excluded")
    print("-" * 50)
    
    # Encontrar clusters que mapean a Desert y Excluded
    desert_clusters = [k for k, v in cluster_mapping.items() 
                       if v['dominant_phenotype'] == 'Immune_Desert']
    excluded_clusters = [k for k, v in cluster_mapping.items() 
                        if v['dominant_phenotype'] == 'Immune_Excluded']
    
    if not desert_clusters or not excluded_clusters:
        print("  [WARN] No se encontraron clusters análogos a Desert Y Excluded")
        return {'tested': False, 'reason': 'no_analogous_clusters'}
    
    # Obtener CAF abundance
    ab_df = _extract_abundance_matrix(adata, 'q05')
    if ab_df is None:
        ab_df = _extract_abundance_matrix(adata, 'means')
    if ab_df is None:
        return {'tested': False, 'reason': 'no_abundance_data'}
    
    caf_col = None
    for c in ab_df.columns:
        if 'CAF' in c.upper():
            caf_col = c
            break
    if caf_col is None:
        return {'tested': False, 'reason': 'no_CAF_column'}
    
    desert_mask = np.isin(cluster_labels, desert_clusters)
    excluded_mask = np.isin(cluster_labels, excluded_clusters)
    
    caf_desert = ab_df.loc[desert_mask, caf_col].values
    caf_excluded = ab_df.loc[excluded_mask, caf_col].values
    
    caf_desert = caf_desert[np.isfinite(caf_desert)]
    caf_excluded = caf_excluded[np.isfinite(caf_excluded)]
    
    print(f"  Desert-like clusters: {desert_clusters} (n={len(caf_desert):,})")
    print(f"  Excluded-like clusters: {excluded_clusters} (n={len(caf_excluded):,})")
    
    if len(caf_desert) < 10 or len(caf_excluded) < 10:
        return {'tested': False, 'reason': 'insufficient_n'}
    
    d = cohens_d_pooled(caf_desert, caf_excluded)
    stat, pval = safe_mannwhitney(caf_desert, caf_excluded)
    
    print(f"  CAF Desert-like: median={np.median(caf_desert):.3f}")
    print(f"  CAF Excluded-like: median={np.median(caf_excluded):.3f}")
    print(f"  Cohen's d = {d:.3f}, p = {pval:.2e}")
    
    # Hipótesis: CAF MENOR en Desert-like, MAYOR en Excluded-like → d < 0
    if d < -0.3 and pval < 0.05:
        verdict = "CONFIRMED — CAF difference replicates in orthogonal clustering"
    elif d < 0:
        verdict = "WEAK — CAF trend present but effect size small or not significant"
    else:
        verdict = "NOT CONFIRMED — CAF difference may be artefact of gene-score classification"
    
    print(f"  VERDICT: {verdict}")
    
    return {
        'tested': True,
        'cohens_d': float(d),
        'p_value': float(pval) if np.isfinite(pval) else None,
        'desert_like_n': len(caf_desert),
        'excluded_like_n': len(caf_excluded),
        'desert_like_median': float(np.median(caf_desert)),
        'excluded_like_median': float(np.median(caf_excluded)),
        'verdict': verdict,
    }


# ============================================================================
# ENFOQUE B: CLASIFICACIÓN POR CONTEXTO ESPACIAL
# ============================================================================

def classify_by_spatial_context(
    adata: ad.AnnData,
    cd8_quantile: str = 'means',
    min_neighbors: int = 3,
) -> np.ndarray:
    """
    Enfoque B: Clasificación basada en el CONTEXTO ESPACIAL del spot.
    
    LÓGICA:
    - Un spot tumoral con vecinos immune-poor Y él mismo immune-poor
      → patrón DESERT (silenciamiento uniforme)
    - Un spot tumoral immune-poor pero con vecinos immune-rich
      → patrón EXCLUDED (barrera impidiendo penetración)
    - Un spot tumoral immune-rich
      → patrón INFLAMED
    
    Esta clasificación NO usa gene scores de MYC/STING/barrera.
    Usa SOLO presencia de CD8 T cells (de Cell2Location).
    """
    print("\n" + "=" * 70)
    print("ENFOQUE B: Clasificación por Contexto Espacial")
    print("=" * 70)
    
    # Construir grafo si no existe
    if 'spatial_connectivities' not in adata.obsp:
        try:
            import squidpy as sq
            if 'sample_id' in adata.obs.columns:
                sq.gr.spatial_neighbors(adata, n_neighs=6, library_key='sample_id')
            else:
                sq.gr.spatial_neighbors(adata, n_neighs=6)
        except Exception as e:
            print(f"  [ERROR] No se pudo construir grafo: {e}")
            return None
    
    # Obtener CD8 abundance
    ab_df = _extract_abundance_matrix(adata, cd8_quantile)
    if ab_df is None:
        return None
    
    cd8_col = None
    for c in ab_df.columns:
        if 'CD8' in c.upper():
            cd8_col = c
            break
    if cd8_col is None:
        print("  [WARN] No se encontró columna CD8")
        return None
    
    cd8_vals = ab_df[cd8_col].values
    
    # Tumor threshold (mismo criterio que phenotype_classifier)
    tumor_col = None
    for c in ab_df.columns:
        if 'Tumor' in c:
            tumor_col = c
            break
    
    # Definir "immune-rich" como CD8 > mediana
    cd8_thresh = np.percentile(cd8_vals, 50)
    immune_rich = cd8_vals > cd8_thresh
    
    print(f"  CD8 threshold (p50): {cd8_thresh:.3f}")
    print(f"  Immune-rich spots: {immune_rich.sum():,} ({100*immune_rich.mean():.1f}%)")
    
    # Para cada spot, calcular fracción de vecinos immune-rich
    conn = adata.obsp['spatial_connectivities']
    n_spots = adata.n_obs
    spatial_labels = np.full(n_spots, 'Unclassified', dtype=object)
    
    for i in range(n_spots):
        # Encontrar vecinos
        neighbors = conn[i].nonzero()[1]
        if len(neighbors) < min_neighbors:
            continue
        
        frac_immune_neighbors = immune_rich[neighbors].mean()
        spot_is_immune = immune_rich[i]
        
        if spot_is_immune:
            spatial_labels[i] = 'Spatial_Inflamed'
        elif frac_immune_neighbors > 0.4:
            # Spot frío pero vecinos calientes → barrera
            spatial_labels[i] = 'Spatial_Excluded'
        else:
            # Spot frío y vecinos fríos → desierto
            spatial_labels[i] = 'Spatial_Desert'
    
    # Resumen
    for label in ['Spatial_Desert', 'Spatial_Excluded', 'Spatial_Inflamed', 'Unclassified']:
        n = (spatial_labels == label).sum()
        print(f"  {label}: {n:,} ({100*n/n_spots:.1f}%)")
    
    return spatial_labels


def compare_spatial_vs_genescore(
    adata: ad.AnnData,
    spatial_labels: np.ndarray,
) -> Dict:
    """
    Compara clasificación espacial vs gene-score.
    
    Mapeo: Spatial_Desert↔Immune_Desert, Spatial_Excluded↔Immune_Excluded,
    Spatial_Inflamed↔Inflamed.
    """
    print("\n" + "-" * 50)
    print("COMPARACIÓN: Spatial Context vs Gene-Score Classification")
    print("-" * 50)
    
    if 'Phenotype' not in adata.obs.columns:
        return {'error': 'No Phenotype column'}
    
    gene_labels = adata.obs['Phenotype'].values
    
    # Mapear para ARI (excluir Unclassified y Normal_Stroma)
    valid_spatial = ~np.isin(spatial_labels, ['Unclassified'])
    valid_gene = ~np.isin(gene_labels, ['Normal_Stroma', 'Unclassified', 'Ambiguous_Cold'])
    valid = valid_spatial & valid_gene
    
    if valid.sum() < 100:
        return {'error': f'insufficient valid spots: {valid.sum()}'}
    
    ari = adjusted_rand_score(gene_labels[valid], spatial_labels[valid])
    nmi = normalized_mutual_info_score(gene_labels[valid], spatial_labels[valid])
    
    print(f"  ARI = {ari:.4f} (valid spots: {valid.sum():,})")
    print(f"  NMI = {nmi:.4f}")
    
    # Concordancia por pares
    mapping = {
        'Spatial_Desert': 'Immune_Desert',
        'Spatial_Excluded': 'Immune_Excluded',
        'Spatial_Inflamed': 'Inflamed',
    }
    
    concordance = {}
    for sp_label, gs_label in mapping.items():
        sp_mask = spatial_labels == sp_label
        gs_mask = gene_labels == gs_label
        both = sp_mask & gs_mask
        if sp_mask.sum() > 0:
            concordance[sp_label] = {
                'spatial_n': int(sp_mask.sum()),
                'genescore_n': int(gs_mask.sum()),
                'overlap_n': int(both.sum()),
                'concordance': float(both.sum() / sp_mask.sum()) if sp_mask.sum() > 0 else 0,
            }
            print(f"  {sp_label} ↔ {gs_label}: "
                  f"overlap={both.sum():,}/{sp_mask.sum():,} "
                  f"({100*both.sum()/sp_mask.sum():.1f}%)")
    
    return {
        'ARI': float(ari),
        'NMI': float(nmi),
        'n_valid': int(valid.sum()),
        'concordance': concordance,
    }


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def run_orthogonal_validation(
    adata: ad.AnnData,
    save_dir: Path = None,
) -> Dict:
    """Ejecuta ambos enfoques ortogonales y genera reporte."""
    
    if save_dir is None:
        save_dir = OUTPUT_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("ORTHOGONAL VALIDATION — Rompe Circularidad")
    print("=" * 80)
    
    all_results = {}
    
    # --- ENFOQUE A: Clustering ---
    cluster_labels, cluster_stats = cluster_by_abundances(adata)
    
    if cluster_labels is not None:
        comparison_a = compare_clustering_vs_classification(adata, cluster_labels)
        all_results['clustering'] = comparison_a
        
        # Test CAF ortogonal
        if 'cluster_mapping' in comparison_a:
            caf_test = test_caf_in_clusters(adata, cluster_labels, comparison_a['cluster_mapping'])
            all_results['caf_orthogonal'] = caf_test
        
        cluster_stats.to_csv(save_dir / 'cluster_stats.csv', index=False)
        adata.obs['Orthogonal_Cluster'] = cluster_labels
    
    # --- ENFOQUE B: Contexto espacial ---
    spatial_labels = classify_by_spatial_context(adata)
    
    if spatial_labels is not None:
        comparison_b = compare_spatial_vs_genescore(adata, spatial_labels)
        all_results['spatial_context'] = comparison_b
        adata.obs['Spatial_Context_Label'] = spatial_labels
        
        # Test CAF en clasificación espacial
        print("\n" + "-" * 50)
        print("TEST CAF en clasificación espacial:")
        sp_desert = spatial_labels == 'Spatial_Desert'
        sp_excluded = spatial_labels == 'Spatial_Excluded'
        
        ab_df = _extract_abundance_matrix(adata, 'q05')
        if ab_df is None:
            ab_df = _extract_abundance_matrix(adata, 'means')
        
        if ab_df is not None:
            caf_col = [c for c in ab_df.columns if 'CAF' in c.upper()]
            if caf_col:
                caf_sp_desert = ab_df.loc[sp_desert, caf_col[0]].values
                caf_sp_excluded = ab_df.loc[sp_excluded, caf_col[0]].values
                caf_sp_desert = caf_sp_desert[np.isfinite(caf_sp_desert)]
                caf_sp_excluded = caf_sp_excluded[np.isfinite(caf_sp_excluded)]
                
                if len(caf_sp_desert) >= 10 and len(caf_sp_excluded) >= 10:
                    d_sp = cohens_d_pooled(caf_sp_desert, caf_sp_excluded)
                    _, p_sp = safe_mannwhitney(caf_sp_desert, caf_sp_excluded)
                    print(f"  Spatial_Desert CAF median: {np.median(caf_sp_desert):.3f}")
                    print(f"  Spatial_Excluded CAF median: {np.median(caf_sp_excluded):.3f}")
                    print(f"  Cohen's d = {d_sp:.3f}, p = {p_sp:.2e}")
                    
                    all_results['caf_spatial'] = {
                        'cohens_d': float(d_sp),
                        'p_value': float(p_sp) if np.isfinite(p_sp) else None,
                    }
    
    # --- RESUMEN ---
    print("\n" + "=" * 80)
    print("RESUMEN ORTHOGONAL VALIDATION")
    print("=" * 80)
    
    if 'clustering' in all_results:
        print(f"  Enfoque A (Clustering): ARI={all_results['clustering'].get('ARI', 'N/A'):.4f}")
    if 'caf_orthogonal' in all_results and all_results['caf_orthogonal'].get('tested'):
        print(f"  CAF ortogonal: d={all_results['caf_orthogonal']['cohens_d']:.3f}, "
              f"verdict={all_results['caf_orthogonal']['verdict']}")
    if 'spatial_context' in all_results:
        print(f"  Enfoque B (Spatial): ARI={all_results['spatial_context'].get('ARI', 'N/A'):.4f}")
    if 'caf_spatial' in all_results:
        print(f"  CAF spatial: d={all_results['caf_spatial']['cohens_d']:.3f}")
    
    # Guardar
    with open(save_dir / 'orthogonal_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Resultados guardados: {save_dir}")
    
    return all_results


if __name__ == '__main__':
    from config import PATHS
    
    print("Cargando datos...")
    adata_path = PATHS.PROCESSED_DIR / 'adata_with_phenotypes.h5ad'
    if not adata_path.exists():
        print(f"[ERROR] No encontrado: {adata_path}")
        import sys; sys.exit(1)
    
    adata = sc.read_h5ad(adata_path)
    run_orthogonal_validation(adata)
