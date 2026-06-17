"""
================================================================================
MODULO 3: CLASIFICACION DE FENOTIPOS TUMORALES
===============================================================================
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Dict, List, Tuple, Optional
from scipy.stats import zscore
from scipy.sparse import issparse

from config import (
    SIGNATURES, PHENOTYPE_PARAMS, PATHS
)

# ============================================================================
# CALCULO DE GENE SIGNATURES (BLINDADO)
# ============================================================================

def _sample_matrix_stats(X, n_samples: int = 1000) -> Tuple[float, float]:
    """
    Obtiene min/max de una muestra de la matriz de forma eficiente.
    Evita cargar toda la matriz en RAM.
    """
    n_rows = min(n_samples, X.shape[0])
    n_cols = min(n_samples, X.shape[1])
    
    if issparse(X):
        sample = X[:n_rows, :n_cols]
        if hasattr(sample, 'data') and len(sample.data) > 0:
            return sample.data.min(), sample.data.max()
        return 0.0, 0.0
    else:
        sample = X[:n_rows, :n_cols]
        return float(sample.min()), float(sample.max())


def calculate_gene_signature_score(
    adata: ad.AnnData,
    gene_list: List[str],
    score_name: str,
    use_raw: bool = True,
) -> ad.AnnData:
    """
    Calcula score de firma génica con validación automática de escala.
    
    FIX v2.4: Optimización de RAM - usa AnnData ligero en lugar de copia completa.
    """
    # 1. Determinar la mejor fuente de datos inicial
    if use_raw and adata.raw is not None:
        adata_work = adata.raw.to_adata()
        source_name = ".raw"
    else:
        adata_work = adata.copy()
        source_name = ".X"
        
    # 2. Diagnóstico de Escala (Detectar Z-Scores o Counts)
    data_min, data_max = _sample_matrix_stats(adata_work.X)
        
    # CASO A: Z-Scores (Negativos detectados) -> TÓXICO para score_genes
    if data_min < -0.1:
        print(f"  [AUTO-FIX] {score_name}: Detectados Z-scores (min={data_min:.2f}).")
        
        # Intentar recuperar counts limpios y re-normalizar
        if 'counts' in adata.layers:
            print("             Usando layer['counts'] + log1p temporal.")
            # FIX v2.4: AnnData ligero en lugar de copia completa
            adata_work = ad.AnnData(
                X=adata.layers['counts'].copy(),
                var=adata.var.copy(),
                obs=adata.obs[[]].copy()  # Solo índices, sin columnas
            )
            sc.pp.normalize_total(adata_work, target_sum=1e4)
            sc.pp.log1p(adata_work)
            
        elif adata.raw is not None:
            # Verificar si .raw también está escalado
            raw_min, raw_max = _sample_matrix_stats(adata.raw.X)
            if raw_min >= -0.1:
                print("             Forzando uso de .raw (Log-norm).")
                adata_work = adata.raw.to_adata()
            else:
                print("             [WARN] .raw también parece escalado. Scores pueden ser inexactos.")
        else:
            print("             [WARN] No se encontró fuente limpia (counts/raw).")

    # CASO B: Counts Crudos (Valores muy altos) -> TÓXICO para score_genes
    elif data_max > 50:
        print(f"  [AUTO-FIX] {score_name}: Detectados counts crudos (max={data_max:.1f}).")
        print("             Aplicando log1p temporal.")
        # FIX v2.4: AnnData ligero
        if issparse(adata_work.X):
            X_copy = adata_work.X.copy()
        else:
            X_copy = adata_work.X.copy()
        adata_work = ad.AnnData(
            X=X_copy,
            var=adata_work.var.copy(),
            obs=adata_work.obs[[]].copy()
        )
        sc.pp.normalize_total(adata_work, target_sum=1e4)
        sc.pp.log1p(adata_work)

    # 3. Validar genes
    available_genes = [g for g in gene_list if g in adata_work.var_names]
    if len(available_genes) == 0:
        print(f"  [WARN] Ningún gen de {score_name} encontrado.")
        adata.obs[score_name] = 0.0
        return adata
    
    n_missing = len(gene_list) - len(available_genes)
    if n_missing > 0:
        print(f"  [INFO] {score_name}: {len(available_genes)}/{len(gene_list)} genes encontrados")

    # 4. Calcular Score
    try:
        sc.tl.score_genes(
            adata_work,
            gene_list=available_genes,
            score_name=score_name,
            use_raw=False  # Ya gestionamos la fuente arriba
        )
        # Transferir resultado al adata original
        adata.obs[score_name] = adata_work.obs[score_name].values
    except Exception as e:
        print(f"  [ERROR] Fallo calculando {score_name}: {e}")
        adata.obs[score_name] = 0.0
        
    # Limpieza final de NaNs
    if adata.obs[score_name].isna().any():
        n_nans = adata.obs[score_name].isna().sum()
        print(f"  [WARN] {n_nans} NaNs en {score_name}, rellenando con 0")
        adata.obs[score_name].fillna(0, inplace=True)
    
    # Liberar memoria
    del adata_work
        
    return adata


def calculate_all_mechanism_scores(adata: ad.AnnData) -> ad.AnnData:
    """Calcula todos los scores mecanísticos."""
    print("\n" + "="*80)
    print("CALCULANDO SCORES DE FIRMAS GÉNICAS (v2.4)")
    print("="*80)
    
    signatures = {
        'Tumor_Score': SIGNATURES.TUMOR_MARKERS,
        'CD8_Score': SIGNATURES.CD8_T_CELLS,
        'Silencing_Score': SIGNATURES.SILENCING_REPRESSORS,
        'STING_Score': SIGNATURES.STING_PATHWAY,
        'Chemokine_Score': SIGNATURES.CHEMOKINE_SIGNALS,
        'Barrier_Score': SIGNATURES.PHYSICAL_BARRIER,
        'Desert_Stroma_Score': SIGNATURES.DESERT_STROMA,
    }
    
    for score_name, gene_list in signatures.items():
        print(f"\nProcesando {score_name}...")
        adata = calculate_gene_signature_score(adata, gene_list, score_name, use_raw=True)
        
    return adata


# ============================================================================
# NORMALIZACION Y CLASIFICACION
# ============================================================================

def normalize_scores_per_sample(adata: ad.AnnData) -> ad.AnnData:
    """Normaliza scores usando z-score por muestra."""
    if not PHENOTYPE_PARAMS.NORMALIZE_SCORES:
        return adata
    
    print("\nNormalizando scores por muestra (z-score)...")
    score_cols = [c for c in adata.obs.columns if c.endswith('_Score')]
    
    for score in score_cols:
        normalized_values = []
        for sample_id in adata.obs['sample_id'].unique():
            mask = adata.obs['sample_id'] == sample_id
            vals = adata.obs.loc[mask, score].values
            
            # Z-score seguro (evita div por 0)
            if len(vals) > 0 and np.std(vals) > 1e-10:
                z_vals = zscore(vals)
            else:
                z_vals = np.zeros_like(vals)
                
            # Manejo NaN
            z_vals = np.nan_to_num(z_vals, nan=0.0, posinf=0.0, neginf=0.0)
            normalized_values.append(pd.Series(z_vals, index=adata.obs[mask].index))
            
        adata.obs[f'{score}_norm'] = pd.concat(normalized_values)
        
    return adata


def classify_phenotype_mechanistic(adata: ad.AnnData, use_normalized: bool = True) -> ad.AnnData:
    """Clasifica fenotipos según lógica del paper."""
    print("\n" + "="*80)
    print("CLASIFICACION MECANISTICA")
    print("="*80)
    
    suffix = '_norm' if use_normalized else ''
    tumor_col = f'Tumor_Score{suffix}'
    cd8_col = f'CD8_Score{suffix}'
    silence_col = f'Silencing_Score{suffix}'
    barrier_col = f'Barrier_Score{suffix}'
    
    # Verificar columnas
    required_cols = [tumor_col, cd8_col, silence_col, barrier_col]
    missing = [c for c in required_cols if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    phenotypes = np.full(adata.n_obs, 'Unclassified', dtype=object)
    
    # Umbrales adaptativos
    t_thresh = np.percentile(adata.obs[tumor_col], PHENOTYPE_PARAMS.TUMOR_PERCENTILE)
    c_thresh = np.percentile(adata.obs[cd8_col], PHENOTYPE_PARAMS.CD8_PERCENTILE)
    
    print(f"  Tumor threshold (p{PHENOTYPE_PARAMS.TUMOR_PERCENTILE}): {t_thresh:.2f}")
    print(f"  CD8 threshold (p{PHENOTYPE_PARAMS.CD8_PERCENTILE}): {c_thresh:.2f}")

    # 1. Normal Stroma
    normal_mask = adata.obs[tumor_col] < t_thresh
    phenotypes[normal_mask] = 'Normal_Stroma'
    
    # 2. Inflamed
    tumor_mask = ~normal_mask
    inflamed_mask = tumor_mask & (adata.obs[cd8_col] > c_thresh)
    phenotypes[inflamed_mask] = 'Inflamed'
    
    # 3. Cold Tumors (Desert vs Excluded)
    cold_mask = tumor_mask & ~inflamed_mask
    if cold_mask.sum() > 0:
        diff = adata.obs.loc[cold_mask, silence_col].values - adata.obs.loc[cold_mask, barrier_col].values
        ambig = PHENOTYPE_PARAMS.COLD_AMBIGUITY_THRESHOLD
        
        idx = np.where(cold_mask)[0]
        phenotypes[idx[diff > ambig]] = 'Immune_Desert'
        phenotypes[idx[diff < -ambig]] = 'Immune_Excluded'
        phenotypes[idx[(diff >= -ambig) & (diff <= ambig)]] = 'Ambiguous_Cold'
        
    adata.obs['Phenotype'] = pd.Categorical(phenotypes)
    
    print("\nResultados de clasificación:")
    print(adata.obs['Phenotype'].value_counts())
    
    return adata


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def calculate_classification_confidence(adata: ad.AnnData) -> ad.AnnData:
    """Calcula métrica de confianza de la clasificación."""
    print("\nCalculando confianza...")
    suffix = '_norm' if PHENOTYPE_PARAMS.NORMALIZE_SCORES else ''
    silence_col = f'Silencing_Score{suffix}'
    barrier_col = f'Barrier_Score{suffix}'
    
    if silence_col in adata.obs.columns and barrier_col in adata.obs.columns:
        adata.obs['Mechanism_Diff'] = adata.obs[silence_col] - adata.obs[barrier_col]
        
    return adata


def summarize_phenotypes_by_sample(adata: ad.AnnData) -> pd.DataFrame:
    """Genera tabla resumen por muestra."""
    print("\nGenerando resumen por muestra...")
    summary_data = []
    
    for sample_id in adata.obs['sample_id'].unique():
        sample_mask = adata.obs['sample_id'] == sample_id
        counts = adata.obs.loc[sample_mask, 'Phenotype'].value_counts()
        total = counts.sum()
        
        row = {'sample_id': sample_id, 'total_spots': total}
        for pheno in counts.index:
            row[f'{pheno}_pct'] = (counts[pheno] / total) * 100 if total > 0 else 0
            
        # Ratio Desert/Excluded
        n_des = counts.get('Immune_Desert', 0)
        n_exc = counts.get('Immune_Excluded', 0)
        row['Desert_to_Excluded_Ratio'] = n_des / n_exc if n_exc > 0 else np.inf
        
        summary_data.append(row)
        
    return pd.DataFrame(summary_data)


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def classify_spatial_phenotypes(adata: ad.AnnData) -> Tuple[ad.AnnData, pd.DataFrame]:
    """Pipeline completo de clasificación de fenotipos."""
    print("\n" + "="*80)
    print("PIPELINE DE CLASIFICACION DE FENOTIPOS v2.4 (Q1)")
    print("="*80)
    
    adata = calculate_all_mechanism_scores(adata)
    adata = normalize_scores_per_sample(adata)
    adata = classify_phenotype_mechanistic(adata, use_normalized=PHENOTYPE_PARAMS.NORMALIZE_SCORES)
    adata = calculate_classification_confidence(adata)
    summary = summarize_phenotypes_by_sample(adata)
    
    # Guardar
    PATHS.create_directories()
    out_path = PATHS.PROCESSED_DIR / 'adata_with_phenotypes.h5ad'
    adata.write_h5ad(out_path)
    print(f"\n[OK] Datos guardados: {out_path}")
    
    summary_path = PATHS.TABLES_DIR / 'phenotype_summary.csv'
    summary.to_csv(summary_path, index=False)
    print(f"[OK] Resumen guardado: {summary_path}")
    
    return adata, summary


if __name__ == '__main__':
    print("Cargando datos...")
    # Prioridad: Cargar salida de deconvolución si existe
    deconv_path = PATHS.PROCESSED_DIR / 'adata_with_deconvolution.h5ad'
    preproc_path = PATHS.PROCESSED_DIR / 'adata_preprocessed.h5ad'
    
    if deconv_path.exists():
        print(f"[INFO] Usando output de deconvolución: {deconv_path}")
        adata = sc.read_h5ad(deconv_path)
    elif preproc_path.exists():
        print(f"[WARN] No se encontró output de deconvolución, usando preprocesado.")
        adata = sc.read_h5ad(preproc_path)
    else:
        raise FileNotFoundError("No se encontraron datos preprocesados")

    classify_spatial_phenotypes(adata)
    print("\n[OK] Script finalizado.")
