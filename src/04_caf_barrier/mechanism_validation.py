"""
================================================================================
VALIDACION DEL MECANISMO
================================================================================
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.stats import spearmanr, mannwhitneyu
from scipy.sparse import issparse
from typing import Optional, List, Union
from config import PATHS, SIGNATURES, CELL_PRESENCE_PARAMS

from utils_stats import cohens_d_pooled, apply_fdr, safe_mannwhitney

# ============================================================================
# HELPER DE EXTRACCION DE GENES
# ============================================================================

def get_gene_expression(
    adata: ad.AnnData, 
    gene: str, 
    use_raw: bool = True,
    aliases: Optional[List[str]] = None
) -> np.ndarray:
    """
    Extrae vector de expresión de forma segura.
    
    - Prioriza .raw si existe (log-norm)
    - Maneja sparse matrices (csr) y arrays densos (numpy)
    - Soporta aliases de genes (ej. TMEM173/STING1)
    - Retorna array 1D plano
    
    Parameters
    ----------
    adata : AnnData
        Objeto de datos
    gene : str
        Nombre del gen principal a buscar
    use_raw : bool
        Si usar .raw (recomendado para expresión)
    aliases : list, optional
        Lista de nombres alternativos del gen
        
    Returns
    -------
    np.ndarray
        Vector de expresión (n_obs,)
    """
    # 1. Seleccionar fuente
    source = adata.raw if (use_raw and adata.raw is not None) else adata
    
    # 2. Buscar gen (incluyendo aliases)
    genes_to_try = [gene]
    if aliases:
        genes_to_try.extend(aliases)
    
    found_gene = None
    for g in genes_to_try:
        if g in source.var_names:
            found_gene = g
            break
    
    if found_gene is None:
        print(f"[WARN] Gen {gene} no encontrado (aliases probados: {aliases})")
        return np.zeros(adata.n_obs)
    
    if found_gene != gene:
        print(f"  [INFO] Usando alias '{found_gene}' para '{gene}'")
        
    # 3. Extraer matriz (puede ser objeto sparse o numpy array)
    data = source[:, found_gene].X
    
    # 4. Convertir a array 1D plano de forma segura
    if issparse(data):
        return data.toarray().flatten()
    elif hasattr(data, 'A'):  # Alternativa para matrices sparse antiguas
        return data.A.flatten()
    else:
        return np.asarray(data).flatten()


# ============================================================================
# HELPER DE COLUMNAS
# ============================================================================

def find_cell_abundance_column(
    adata: ad.AnnData, 
    cell_type: str, 
    quantile: str = 'q05'
) -> Optional[str]:
    """
    Busca columnas de abundancia celular con Fuzzy Matching agresivo.
    
    Soluciona el problema de que Cell2Location a veces cambia prefijos.
    
    Parameters
    ----------
    adata : AnnData
        Objeto de datos con abundancias en .obs o .obsm
    cell_type : str
        Tipo celular (ej. 'CD8_T', 'cDC1')
    quantile : str
        Cuantil ('q05', 'q50', 'q95', 'means')
        
    Returns
    -------
    str or None
        Nombre de la columna encontrada, o None si no existe
    """
    # 1. Candidatos exactos y esperados
    candidates = [
        f'{cell_type}_{quantile}',
        f'{cell_type}_{quantile.upper()}',
        f'{quantile}_{cell_type}',
    ]
    # Añadir patrones REALES de Cell2Location en HPC
    # Formato real: 'meanscell_abundance_w_sf_{ct}' (sin underscore entre prefijo y 'cell')
    if quantile == 'means':
        candidates.extend([
            f'meanscell_abundance_w_sf_{cell_type}',   # Formato REAL HPC
            f'means_cell_abundance_w_sf_{cell_type}',   # Formato esperado
        ])
    elif quantile in ('q05', 'q50', 'q95'):
        candidates.extend([
            f'{quantile}cell_abundance_w_sf_{cell_type}',   # Formato REAL HPC
            f'{quantile}_cell_abundance_w_sf_{cell_type}',  # Formato esperado
        ])
    
    for col in candidates:
        if col in adata.obs.columns:
            return col
            
    # 2. Búsqueda Fuzzy (Flexible)
    # Busca cualquier columna que contenga el tipo celular Y el cuantil
    clean_type = cell_type.replace('_', '').replace(' ', '').lower()
    
    for col in adata.obs.columns:
        clean_col = col.replace('_', '').lower()
        if clean_type in clean_col and quantile.lower() in col.lower():
            return col
            
    # 3. Último recurso: buscar en .obsm (si no se extrajo a .obs)
    possible_keys = [k for k in adata.obsm.keys() if quantile in k.lower() or 'abundance' in k.lower()]
    
    for key in possible_keys:
        try:
            obsm_data = adata.obsm[key]
            
            # Puede ser DataFrame o array
            if hasattr(obsm_data, 'columns'):
                cols = obsm_data.columns
            elif hasattr(obsm_data, 'dtype') and obsm_data.dtype.names:
                cols = obsm_data.dtype.names
            else:
                continue
                
            # Buscar el tipo celular en las columnas
            for col in cols:
                if cell_type in str(col):
                    print(f"  [INFO] Encontrado en .obsm['{key}']['{col}']")
                    
                    # Intentar guardar en .obs para uso futuro
                    new_col_name = f"{cell_type}_{quantile}"
                    try:
                        if hasattr(obsm_data, 'columns'):
                            adata.obs[new_col_name] = obsm_data[col].values
                        else:
                            idx = list(cols).index(col)
                            adata.obs[new_col_name] = obsm_data[:, idx]
                        return new_col_name
                    except Exception as e:
                        print(f"  [WARN] No se pudo copiar a .obs: {e}")
                        # Retornar el key de obsm para manejo manual
                        return f"obsm:{key}:{col}"
        except Exception:
            continue

    print(f"[WARN] No se encontró columna de abundancia para {cell_type} ({quantile})")
    return None


def get_abundance_values(
    adata: ad.AnnData, 
    cell_type: str, 
    quantile: str = 'q05'
) -> np.ndarray:
    """
    Obtiene valores de abundancia celular de forma robusta.
    
    Wrapper de find_cell_abundance_column que siempre retorna un array.
    """
    col = find_cell_abundance_column(adata, cell_type, quantile)
    
    if col is None:
        return np.zeros(adata.n_obs)
    
    # Caso especial: dato en obsm
    if col.startswith('obsm:'):
        parts = col.split(':')
        key, subcol = parts[1], parts[2]
        obsm_data = adata.obsm[key]
        if hasattr(obsm_data, 'columns'):
            return obsm_data[subcol].values
        else:
            idx = list(obsm_data.dtype.names).index(subcol)
            return obsm_data[:, idx]
    
    return adata.obs[col].values


# ============================================================================
# GENE ALIASES
# ============================================================================

# Diccionario de aliases conocidos para genes importantes
GENE_ALIASES = {
    'TMEM173': ['STING1', 'MITA', 'MPYS', 'ERIS'],  # STING
    'STING1': ['TMEM173', 'MITA', 'MPYS', 'ERIS'],
    'PDCD1': ['PD1', 'CD279'],  # PD-1
    'CD274': ['PDL1', 'B7H1'],  # PD-L1
    'IFNG': ['IFN-gamma', 'IFNG1'],
    'IL2': ['IL-2'],
}


# ============================================================================
# ANALISIS ESPECIFICOS
# ============================================================================

def calculate_cxcl9_spp1_ratio(adata: ad.AnnData) -> ad.AnnData:
    """
    Calcula ratio CXCL9:SPP1 (marcador de inflamación vs supresión).
    """
    print("\nCalculando Ratio CXCL9:SPP1 (FIX v2.5 — espacio lineal)...")
    
    # Extraer expresión (log1p-normalizada de .raw)
    cxcl9_log = get_gene_expression(adata, 'CXCL9')
    spp1_log = get_gene_expression(adata, 'SPP1')
    
    # FIX AUDIT v2.5: Revertir log1p → espacio lineal
    cxcl9_linear = np.expm1(cxcl9_log)  # expm1(x) = e^x - 1 (inversa de log1p)
    spp1_linear = np.expm1(spp1_log)
    
    # Pseudocount en espacio lineal (1.0, no 0.1)
    PSEUDOCOUNT = 1.0
    ratio_linear = (cxcl9_linear + PSEUDOCOUNT) / (spp1_linear + PSEUDOCOUNT)
    log2_ratio = np.log2(ratio_linear)
    
    # Guardar AMBOS ratios
    adata.obs['CXCL9_SPP1_ratio_linear'] = ratio_linear
    adata.obs['CXCL9_SPP1_log2ratio'] = log2_ratio
    
    # Diagnósticos
    print(f"  [DIAG] CXCL9 log-space: min={cxcl9_log.min():.2f}, "
          f"max={cxcl9_log.max():.2f}, median={np.median(cxcl9_log):.2f}")
    print(f"  [DIAG] CXCL9 linear:    min={cxcl9_linear.min():.2f}, "
          f"max={cxcl9_linear.max():.2f}, median={np.median(cxcl9_linear):.2f}")
    print(f"  [DIAG] SPP1  linear:    min={spp1_linear.min():.2f}, "
          f"max={spp1_linear.max():.2f}, median={np.median(spp1_linear):.2f}")
    print(f"  [DIAG] Ratio linear:    min={ratio_linear.min():.2f}, "
          f"max={ratio_linear.max():.2f}, median={np.median(ratio_linear):.2f}")
    print(f"  [DIAG] Log2 ratio:      min={log2_ratio.min():.2f}, "
          f"max={log2_ratio.max():.2f}, median={np.median(log2_ratio):.2f}")
    
    # Verificar rango razonable
    if log2_ratio.min() < -15 or log2_ratio.max() > 15:
        print(f"  [WARN] Log2 ratio fuera de rango esperado [-15, +15]")
    
    print(f"  [OK] Ratio calculado en espacio lineal. Mediana log2: {np.median(log2_ratio):.2f}")
    return adata


def validate_myc_sting_axis(adata: ad.AnnData) -> pd.DataFrame:
    """
    Valida el eje MYC-STING (represores epigenéticos vs respuesta inmune).
    
    """
    print("\nValidando Eje MYC-STING (Correlaciones single-gene — SUPLEMENTARIAS)...")
    print("  NOTA: La evidencia primaria es MYC Hallmark Score (Block 2)")
    print("  mRNA de MYC es proxy pobre de actividad MYC (Linstra 2025)")
    results = []
    
    # Subset de 4 represores con evidencia directa de represión epigenética
    # solo testeamos los 4 con evidencia funcional directa:
    # MYC (Lee 2022), EZH2 (PRC2 H3K27me3), DNMT1 (Wu 2021), STAT3 (Snoeren 2025)
    # SUZ12, CTNNB1, ATF3 son contribuyentes indirectos — no incluidos en test de correlación
    genes_repressors = ['MYC', 'EZH2', 'DNMT1', 'STAT3']
    
    # Buscar STING con aliases
    sting_aliases = ['TMEM173', 'STING1', 'MITA']
    target_expr = get_gene_expression(
        adata, 
        'TMEM173', 
        aliases=['STING1', 'MITA']
    )
    
    # Verificar que tenemos expresión de STING
    if np.all(target_expr == 0):
        print("[WARN] No se encontró expresión de STING (TMEM173/STING1)")
        return pd.DataFrame()
    
    gene_target = 'STING'  # Nombre display
    
    for repressor in genes_repressors:
        rep_expr = get_gene_expression(adata, repressor)
        
        # Detectar gen ausente (todos zeros) y reportar como faltante
        if np.all(rep_expr == 0):
            print(f"  [SKIP] {repressor}: expresión es todo ceros (gen ausente o no expresado)")
            continue
        
        # Solo calcular si hay variación
        if np.std(rep_expr) < 1e-10 or np.std(target_expr) < 1e-10:
            print(f"  [SKIP] {repressor}: varianza insuficiente")
            continue
            
        corr, pval = spearmanr(rep_expr, target_expr)
        
        # Report n per test
        n_nonzero = int(np.sum((rep_expr > 0) & (target_expr > 0)))
        results.append({
            'Pair': f'{repressor} vs {gene_target}',
            'Spearman_rho': corr,
            'p_value': pval,
            'n_spots': int(adata.n_obs),
            'n_both_expressed': n_nonzero,
            'Significant': pval < 0.05,
            'Direction': 'Negative' if corr < 0 else 'Positive'
        })
        print(f"  {repressor} vs {gene_target}: rho={corr:.3f}, p={pval:.2e} (n={adata.n_obs:,})")
        
    return pd.DataFrame(results)


def validate_chemokine_correlations(adata: ad.AnnData) -> pd.DataFrame:
    """Valida correlaciones entre quimioquinas y células inmunes."""
    print("\nValidando Correlaciones Quimioquina-Células...")
    results = []
    
    chemokines = ['CXCL9', 'CXCL10', 'CCL5']
    cell_types = ['CD8_T', 'cDC1']
    
    for chemo in chemokines:
        chemo_expr = get_gene_expression(adata, chemo)
        
        for ct in cell_types:
            # FIX-19 AUDIT v3: q05 para consistencia con statistical_tests_desert_vs_excluded
            ct_abundance = get_abundance_values(adata, ct, 'q05')
            
            if np.std(chemo_expr) < 1e-10 or np.std(ct_abundance) < 1e-10:
                continue
                
            corr, pval = spearmanr(chemo_expr, ct_abundance)
            
            results.append({
                'Chemokine': chemo,
                'Cell_Type': ct,
                'Spearman_rho': corr,
                'p_value': pval,
                'Significant': pval < 0.05
            })
            
    return pd.DataFrame(results)


def statistical_tests_desert_vs_excluded(adata: ad.AnnData) -> pd.DataFrame:
    """
    Compara abundancias celulares entre Desert vs Excluded.
    
    """
    print("\nTests Estadísticos: Desert vs Excluded...")
    
    if 'Phenotype' not in adata.obs.columns:
        print("[WARN] No hay fenotipos clasificados.")
        return pd.DataFrame()
        
    results = []
    cell_types = ['CD8_T', 'cDC1', 'CAF', 'Macrophage']
    
    for ct in cell_types:
        col = find_cell_abundance_column(adata, ct, 'q05')
        if not col:
            continue
        
        # Obtener valores por fenotipo
        desert_mask = adata.obs['Phenotype'] == 'Immune_Desert'
        excluded_mask = adata.obs['Phenotype'] == 'Immune_Excluded'
        
        desert = adata.obs.loc[desert_mask, col].values
        excluded = adata.obs.loc[excluded_mask, col].values
        
        # Filtrar NaN/Inf
        desert = desert[np.isfinite(desert)]
        excluded = excluded[np.isfinite(excluded)]
        
        if len(desert) < 10 or len(excluded) < 10:
            print(f"  [SKIP] {ct}: n insuficiente (Desert={len(desert)}, Excluded={len(excluded)})")
            continue
        
        stat, pval = safe_mannwhitney(desert, excluded)
        
        d = cohens_d_pooled(desert, excluded)
        
        results.append({
            'Cell_Type': ct,
            'Desert_mean': desert.mean(),
            'Excluded_mean': excluded.mean(),
            'Desert_n': len(desert),
            'Excluded_n': len(excluded),
            'p_value': pval,
            'Cohens_d': d,
            'Effect_Size': 'Large' if abs(d) > 0.8 else ('Medium' if abs(d) > 0.5 else 'Small'),
            'Significant': pval < 0.05 if not np.isnan(pval) else False
        })
        print(f"  {ct}: p={pval:.2e}, Cohen's d={d:.2f} (pooled, ddof=1)")
            
    return pd.DataFrame(results)


# ============================================================================
# FDR CORRECTION 
# ============================================================================

def _collect_all_pvalues(results: dict) -> pd.DataFrame:
    """
    Recopila TODOS los p-values generados por el pipeline de validación.
    
    Recoge de:
    - myc_sting: correlaciones represores vs STING
    - chemokines: correlaciones quimioquina-célula
    - differential_tests: tests Desert vs Excluded
    
    Returns
    -------
    pd.DataFrame
        Tabla con columnas: test_name, p_value, source
    """
    all_pvals = []
    
    # 1. Correlaciones MYC-STING
    df_corr = results.get('myc_sting', pd.DataFrame())
    if not df_corr.empty and 'p_value' in df_corr.columns:
        for _, row in df_corr.iterrows():
            all_pvals.append({
                'test_name': row.get('Pair', 'unknown'),
                'p_value': row['p_value'],
                'source': 'myc_sting_correlation',
                'statistic': row.get('Spearman_rho', np.nan),
            })
    
    # 2. Correlaciones quimioquinas
    df_chemo = results.get('chemokines', pd.DataFrame())
    if not df_chemo.empty and 'p_value' in df_chemo.columns:
        for _, row in df_chemo.iterrows():
            all_pvals.append({
                'test_name': f"{row.get('Chemokine', '?')} vs {row.get('Cell_Type', '?')}",
                'p_value': row['p_value'],
                'source': 'chemokine_correlation',
                'statistic': row.get('Spearman_rho', np.nan),
            })
    
    # 3. Tests diferenciales
    df_tests = results.get('differential_tests', pd.DataFrame())
    if not df_tests.empty and 'p_value' in df_tests.columns:
        for _, row in df_tests.iterrows():
            all_pvals.append({
                'test_name': f"Desert_vs_Excluded_{row.get('Cell_Type', '?')}",
                'p_value': row['p_value'],
                'source': 'differential_test',
                'statistic': row.get('Cohens_d', np.nan),
            })
    
    return pd.DataFrame(all_pvals)


def apply_fdr_correction(results: dict) -> pd.DataFrame:
    """    
    Recopila todos los p-values del pipeline, aplica corrección FDR global,
    y genera tabla con q-values.
    
    Returns
    -------
    pd.DataFrame
        Tabla con p-values crudos y q-values corregidos.
    """
    print("\n" + "="*60)
    print("FDR CORRECTION (Benjamini-Hochberg)")
    print("="*60)
    
    df_pvals = _collect_all_pvalues(results)
    
    if df_pvals.empty:
        print("[WARN] No se encontraron p-values para corregir.")
        return pd.DataFrame()
    
    n_tests = len(df_pvals)
    print(f"  Tests recopilados: {n_tests}")
    
    # Aplicar FDR
    pvals = df_pvals['p_value'].values
    reject, q_values = apply_fdr(pvals, method='fdr_bh', alpha=0.05)
    
    df_pvals['q_value'] = q_values
    df_pvals['fdr_significant'] = reject
    
    # Resumen
    n_raw_sig = (pvals < 0.05).sum()
    n_fdr_sig = reject.sum()
    
    print(f"  Significativos (p < 0.05):   {n_raw_sig}/{n_tests}")
    print(f"  Sobreviven FDR (q < 0.05):   {n_fdr_sig}/{n_tests}")
    
    if n_fdr_sig < n_raw_sig:
        print(f"  [INFO] {n_raw_sig - n_fdr_sig} tests perdieron significancia tras FDR.")
    
    print("\n  Detalle:")
    for _, row in df_pvals.iterrows():
        status = "✓ FDR" if row['fdr_significant'] else "✗ FDR"
        print(f"    [{status}] {row['test_name']}: p={row['p_value']:.2e} → q={row['q_value']:.2e}")
    
    return df_pvals


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def validate_silencing_mechanism(adata: ad.AnnData) -> dict:
    """Pipeline completo de validación del mecanismo."""
    print("\n" + "="*80)
    print("PIPELINE DE VALIDACION DE MECANISMO")
    print("="*80)
    
    results = {}
    
    # 1. Ratios 
    adata = calculate_cxcl9_spp1_ratio(adata)
    
    # 2. Correlaciones MYC-STING
    df_corr = validate_myc_sting_axis(adata)
    results['myc_sting'] = df_corr
    
    # 3. Correlaciones Quimioquinas
    df_chemo = validate_chemokine_correlations(adata)
    results['chemokines'] = df_chemo
    
    # 4. Tests Diferenciales
    df_tests = statistical_tests_desert_vs_excluded(adata)
    results['differential_tests'] = df_tests
    
    # Guardar resultados individuales
    PATHS.create_directories()
    
    if not df_corr.empty:
        df_corr.to_csv(PATHS.TABLES_DIR / 'myc_sting_correlations.csv', index=False)
        print(f"\n[OK] Correlaciones guardadas")
        
    if not df_chemo.empty:
        df_chemo.to_csv(PATHS.TABLES_DIR / 'chemokine_correlations.csv', index=False)
        
    if not df_tests.empty:
        df_tests.to_csv(PATHS.TABLES_DIR / 'desert_vs_excluded_tests.csv', index=False)
        print(f"[OK] Tests estadísticos guardados")
    
    df_fdr = apply_fdr_correction(results)
    results['fdr_table'] = df_fdr
    
    if not df_fdr.empty:
        fdr_path = PATHS.TABLES_DIR / 'fdr_corrected_pvalues.csv'
        df_fdr.to_csv(fdr_path, index=False)
        print(f"\n[OK] FDR table guardada: {fdr_path}")
    
    # para que módulos downstream (visualization, spatial_analysis) puedan acceder a ellas
    updated_path = PATHS.PROCESSED_DIR / 'adata_with_mechanism.h5ad'
    try:
        adata.write_h5ad(updated_path)
        print(f"\n[OK] adata actualizado guardado: {updated_path}")
    except Exception as e:
        print(f"\n[WARN] No se pudo guardar adata actualizado: {e}")

    print("\n[OK] Validación completada.")
    return results


if __name__ == '__main__':
    print("Cargando datos clasificados...")
    try:
        adata = sc.read_h5ad(PATHS.PROCESSED_DIR / 'adata_with_phenotypes.h5ad')
        validate_silencing_mechanism(adata)
    except FileNotFoundError:
        print("[ERROR] No se encontró adata_with_phenotypes.h5ad")
        print("        Ejecuta primero phenotype_classifier.py")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
