"""
================================================================================
MODULO 2: PREPROCESAMIENTO Y CARGA DE DATOS
================================================================================
Funciones para:
- Cargar datos Visium de GSE210616 y GSE213688
- Cargar referencia scRNA-seq de GSE176078
- Control de calidad (QC) con verificacion de NaN
- Normalizacion y filtrado
- Integracion con Harmony

================================================================================
"""

import os
import gc
import gzip
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from scipy.sparse import issparse, csr_matrix
import anndata as ad

from config import (
    PATHS, SIGNATURES, QC_PARAMS, HARMONY_PARAMS,
    ENVIRONMENT
)

warnings.filterwarnings('ignore')

# Configurar scanpy para HPC (headless)
sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=150, facecolor='white', frameon=False)
sc.settings.autosave = True
try:
    sc.settings.figdir = PATHS.FIGURES_DIR
except (OSError, AttributeError):
    pass


# ============================================================================
# VERIFICACION DE INTEGRIDAD DE DATOS
# ============================================================================

def validate_no_nan(adata: ad.AnnData, step_name: str) -> None:
    """
    Verificacion CRITICA: detecta NaN/Inf en la matriz de expresion
    
    Esta funcion previene el error historico de Cell2Location donde
    valores NaN en adata.X causaban que los parametros del modelo
    se volvieran NaN durante el entrenamiento.
    
    Args:
        adata: Objeto AnnData a verificar
        step_name: Nombre del paso actual (para logging)
    
    Raises:
        ValueError: Si se detectan valores invalidos
    """
    print(f"\n[{step_name}] Verificando integridad de datos...")
    
    # Obtener matriz (densa o sparse)
    if issparse(adata.X):
        data_array = adata.X.data
    else:
        data_array = adata.X.ravel()
    
    # Verificar NaN
    n_nan = np.sum(np.isnan(data_array))
    if n_nan > 0:
        raise ValueError(
            f"[ERROR] en [{step_name}]: "
            f"Se encontraron {n_nan} valores NaN en adata.X\n"
            f"Esto causaria fallo en Cell2Location. Pipeline detenido."
        )
    
    # Verificar Inf
    n_inf = np.sum(np.isinf(data_array))
    if n_inf > 0:
        raise ValueError(
            f"[ERROR] en [{step_name}]: "
            f"Se encontraron {n_inf} valores Inf en adata.X\n"
            f"Pipeline detenido."
        )
    
    # Verificar valores negativos (no deberian existir en counts)
    n_negative = np.sum(data_array < 0)
    if n_negative > 0:
        print(f"[WARN] Se encontraron {n_negative} valores negativos en adata.X")
    
    print(f"[OK] [{step_name}] Datos validos: sin NaN, sin Inf")
    print(f"     Rango de valores: [{data_array.min():.2f}, {data_array.max():.2f}]")
    if len(data_array) > 0:
        print(f"     Sparsity: {1 - np.count_nonzero(data_array) / len(data_array):.2%}")


def check_and_clean_nan(X: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Detecta y reemplaza NaN de manera segura
    
    Args:
        X: Array numpy o sparse
        fill_value: Valor para reemplazar NaN
    
    Returns:
        Array sin NaN
    """
    if issparse(X):
        X.data = np.nan_to_num(X.data, nan=fill_value, posinf=fill_value, neginf=fill_value)
        return X
    else:
        return np.nan_to_num(X, nan=fill_value, posinf=fill_value, neginf=fill_value)


# ============================================================================
# CARGA DE DATOS VISIUM
# ============================================================================

def load_visium_sample(
    sample_path: Path,
    sample_id: str,
    min_counts: int = 500,
    min_genes: int = 200,
    h5_file: Optional[Path] = None,
) -> Optional[ad.AnnData]:
    """
    Carga una muestra individual de Visium
    
    Soporta multiples formatos de nombres de archivos:
    - GSE210616: GSM6433585_092A_filtered_feature_bc_matrix.h5
    - GSE213688: GSM6592048_M1_filtered_feature_bc_matrix.h5
    - Estructura organizada: sample_dir/filtered_feature_bc_matrix.h5
    
    Args:
        sample_path: Ruta al directorio con los archivos de la muestra
        sample_id: ID de la muestra (092A, M1, etc.)
        min_counts: Minimo de counts por spot
        min_genes: Minimo de genes por spot
        h5_file: Ruta directa al archivo h5 (opcional)
    
    Returns:
        AnnData con la muestra o None si falla
    """
    print(f"\nCargando muestra: {sample_id}")
    
    try:
        # Si se proporciona h5_file directamente, usarlo
        if h5_file is not None and h5_file.exists():
            print(f"  Archivo (directo): {h5_file.name}")
        else:
            # Buscar archivo h5 en el directorio
            # Patron 1: estructura organizada (sample_dir/filtered_feature_bc_matrix.h5)
            h5_files = list(sample_path.glob("filtered_feature_bc_matrix.h5"))
            
            # Patron 2: archivos con prefijo GSM
            if not h5_files:
                h5_files = list(sample_path.glob(f"*{sample_id}*filtered*.h5"))
            
            # Patron 3: cualquier h5
            if not h5_files:
                h5_files = list(sample_path.glob("*filtered*.h5")) + list(sample_path.glob("*.h5"))
            
            if not h5_files:
                print(f"[WARN] No se encontro archivo h5 en {sample_path}")
                return None
            
            h5_file = h5_files[0]
            print(f"  Archivo: {h5_file.name}")
        
        # Cargar matriz de expresion
        adata = sc.read_10x_h5(h5_file)
        
        # Intentar cargar informacion espacial
        spatial_dir = sample_path
        spatial_loaded = False
        
        try:
            # Descomprimir archivos .gz si es necesario
            for file in spatial_dir.glob("*.gz"):
                uncompressed = file.with_suffix('')
                if not uncompressed.exists():
                    print(f"  Descomprimiendo {file.name}...")
                    with gzip.open(file, 'rb') as f_in:
                        with open(uncompressed, 'wb') as f_out:
                            f_out.write(f_in.read())
            
            # Verificar si existe tissue_positions
            tissue_files = list(spatial_dir.glob("*tissue_positions*"))
            if tissue_files:
                try:
                    adata_sq = sq.read.visium(str(spatial_dir))
                    
                    # Copiar coordenadas espaciales al adata original
                    if 'spatial' in adata_sq.obsm:
                        common_idx = adata.obs_names.intersection(adata_sq.obs_names)
                        if len(common_idx) > 0:
                            adata = adata[common_idx, :].copy()
                            adata.obsm['spatial'] = adata_sq[common_idx, :].obsm['spatial']
                            
                            if 'spatial' in adata_sq.uns:
                                adata.uns['spatial'] = adata_sq.uns['spatial']
                            
                            spatial_loaded = True
                            print(f"  [OK] Coordenadas espaciales cargadas: {adata.obsm['spatial'].shape}")
                    
                    del adata_sq
                    gc.collect()
                    
                except Exception as e:
                    print(f"  [WARN] squidpy fallo, intentando metodo alternativo: {e}")
            
            # Metodo alternativo: cargar tissue_positions manualmente
            if not spatial_loaded:
                for tp_file in tissue_files:
                    try:
                        tp_df = pd.read_csv(tp_file, header=None if 'csv' in str(tp_file) else 0)
                        if len(tp_df.columns) >= 6:
                            tp_df.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col'][:len(tp_df.columns)]
                            tp_df = tp_df.set_index('barcode')
                            
                            common_barcodes = adata.obs_names.intersection(tp_df.index)
                            if len(common_barcodes) > 0:
                                adata = adata[common_barcodes, :].copy()
                                spatial_coords = tp_df.loc[common_barcodes, ['pxl_col', 'pxl_row']].values
                                adata.obsm['spatial'] = spatial_coords
                                spatial_loaded = True
                                print(f"  [OK] Coordenadas cargadas manualmente: {spatial_coords.shape}")
                            break
                    except Exception as e:
                        continue
                        
        except Exception as e:
            print(f"  [WARN] No se pudieron cargar coordenadas espaciales: {e}")
        
        if not spatial_loaded:
            print(f"  [WARN] Muestra cargada sin coordenadas espaciales")
        
        # Agregar metadatos
        adata.obs['sample_id'] = sample_id
        adata.var_names_make_unique()
        
        # Filtrado basico
        initial_spots = adata.n_obs
        sc.pp.filter_cells(adata, min_counts=min_counts)
        sc.pp.filter_cells(adata, min_genes=min_genes)
        
        print(f"  Spots: {initial_spots} -> {adata.n_obs} (despues de filtrado)")
        print(f"  Genes: {adata.n_vars}")
        
        return adata
        
    except Exception as e:
        print(f"[ERROR] Error cargando muestra {sample_id}: {e}")
        return None


def load_visium_cohort(
    data_dir: Path,
    cohort_name: str,
    min_counts: int = 500,
    min_genes: int = 200,
) -> ad.AnnData:
    """
    Carga una cohorte completa de muestras Visium
    
    Soporta dos estructuras:
    1. Organizada: data_dir/sample_id/filtered_feature_bc_matrix.h5
    2. Dispersa: data_dir/GSMxxxx_sampleid_filtered_feature_bc_matrix.h5
    
    Args:
        data_dir: Directorio con las muestras
        cohort_name: Nombre de la cohorte (GSE210616 o GSE213688)
        min_counts: Minimo de counts por spot
        min_genes: Minimo de genes por spot
    
    Returns:
        AnnData concatenado con todas las muestras
    """
    print("\n" + "="*80)
    print(f"CARGANDO COHORTE: {cohort_name}")
    print("="*80)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {data_dir}")
    
    # Identificar muestras segun estructura
    sample_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    h5_files = list(data_dir.glob("*filtered_feature_bc_matrix.h5"))
    
    adatas = []
    
    # Caso 1: Estructura organizada (subdirectorios por muestra)
    if sample_dirs and not h5_files:
        print(f"Detectada estructura organizada: {len(sample_dirs)} subdirectorios")
        for sample_dir in sorted(sample_dirs):
            sample_id = sample_dir.name
            adata = load_visium_sample(sample_dir, sample_id, min_counts, min_genes)
            if adata is not None:
                adatas.append(adata)
    
    # Caso 2: Archivos h5 dispersos (patron GSE)
    elif h5_files:
        print(f"Detectada estructura dispersa: {len(h5_files)} archivos h5")
        
        # Extraer sample_id de los nombres de archivo
        # GSE210616: GSM6433585_092A_filtered -> 092A
        # GSE213688: GSM6592048_M1_filtered -> M1
        
        samples_found = {}
        
        for h5_file in h5_files:
            filename = h5_file.name
            
            # Patron GSE210616
            if filename.startswith('GSM6433'):
                import re
                match = re.match(r'GSM6433\d+_(\w+)_filtered', filename)
                if match:
                    sample_id = match.group(1)
                    samples_found[sample_id] = h5_file
            
            # Patron GSE213688
            elif filename.startswith('GSM6592'):
                import re
                match = re.match(r'GSM6592\d+_(M\d+)_filtered', filename)
                if match:
                    sample_id = match.group(1)
                    samples_found[sample_id] = h5_file
            
            # Patron generico
            else:
                sample_id = h5_file.stem.replace('_filtered_feature_bc_matrix', '')
                samples_found[sample_id] = h5_file
        
        print(f"  Muestras identificadas: {len(samples_found)}")
        
        for sample_id, h5_file in sorted(samples_found.items()):
            adata = load_visium_sample(
                data_dir, sample_id, min_counts, min_genes, h5_file=h5_file
            )
            if adata is not None:
                adatas.append(adata)
    
    else:
        raise ValueError(f"No se encontraron muestras en {data_dir}")
    
    if not adatas:
        raise ValueError(f"No se pudieron cargar muestras de {data_dir}")
    
    # Concatenar muestras
    print(f"\nConcatenando {len(adatas)} muestras...")
    adata_combined = ad.concat(adatas, join='outer', label='sample_id', index_unique='-')
    
    # Agregar metadatos de cohorte
    adata_combined.obs['cohort'] = cohort_name
    adata_combined.obs['batch'] = adata_combined.obs['sample_id']
    
    # Verificar integridad
    validate_no_nan(adata_combined, f"Carga de {cohort_name}")
    
    print(f"\n[OK] Cohorte {cohort_name} cargada:")
    print(f"     Total spots: {adata_combined.n_obs}")
    print(f"     Total genes: {adata_combined.n_vars}")
    print(f"     Muestras: {adata_combined.obs['sample_id'].nunique()}")
    
    # Liberar memoria
    for adata in adatas:
        del adata
    gc.collect()
    
    return adata_combined


# ============================================================================
# CONTROL DE CALIDAD
# ============================================================================

def calculate_qc_metrics(adata: ad.AnnData) -> ad.AnnData:
    """
    Calcula metricas de QC para cada spot
    
    Args:
        adata: Objeto AnnData
    
    Returns:
        AnnData con metricas de QC en .obs
    """
    print("\nCalculando metricas de QC...")
    
    # Identificar genes mitocondriales
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    
    # Identificar genes ribosomales
    adata.var['ribo'] = adata.var_names.str.startswith(('RPS', 'RPL'))
    
    # Calcular metricas
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=['mt', 'ribo'],
        percent_top=None,
        log1p=False,
        inplace=True
    )
    
    print(f"  Mediana counts/spot: {adata.obs['total_counts'].median():.0f}")
    print(f"  Mediana genes/spot: {adata.obs['n_genes_by_counts'].median():.0f}")
    print(f"  Mediana % MT: {adata.obs['pct_counts_mt'].median():.1f}%")
    
    return adata


def filter_cells_and_genes(adata: ad.AnnData) -> ad.AnnData:
    """
    Filtra spots y genes segun parametros de QC
    
    Args:
        adata: AnnData con metricas de QC
    
    Returns:
        AnnData filtrado
    """
    print("\nFiltrando spots y genes...")
    
    initial_spots = adata.n_obs
    initial_genes = adata.n_vars
    
    # Filtrar por counts
    adata = adata[adata.obs['total_counts'] >= QC_PARAMS.MIN_COUNTS_PER_SPOT, :].copy()
    adata = adata[adata.obs['total_counts'] <= QC_PARAMS.MAX_COUNTS_PER_SPOT, :].copy()
    
    # Filtrar por genes
    adata = adata[adata.obs['n_genes_by_counts'] >= QC_PARAMS.MIN_GENES_PER_SPOT, :].copy()
    
    # Filtrar por % mitocondrial
    adata = adata[adata.obs['pct_counts_mt'] <= QC_PARAMS.MAX_MT_PERCENT, :].copy()
    
    # Filtrar genes
    sc.pp.filter_genes(adata, min_cells=QC_PARAMS.MIN_SPOTS_PER_GENE)
    
    print(f"  Spots: {initial_spots} -> {adata.n_obs} ({adata.n_obs/initial_spots*100:.1f}%)")
    print(f"  Genes: {initial_genes} -> {adata.n_vars} ({adata.n_vars/initial_genes*100:.1f}%)")
    
    validate_no_nan(adata, "Despues de filtrado QC")
    
    return adata


# ============================================================================
# NORMALIZACION
# ============================================================================

def normalize_data(adata: ad.AnnData) -> ad.AnnData:
    """
    Normaliza los datos de expresion
    
    Args:
        adata: AnnData filtrado
    
    Returns:
        AnnData normalizado con .raw guardado
    """
    print("\nNormalizando datos...")
    
    # Guardar counts crudos
    adata.layers['counts'] = adata.X.copy()
    
    # Normalizar a target_sum counts por spot
    sc.pp.normalize_total(adata, target_sum=QC_PARAMS.TARGET_SUM)
    validate_no_nan(adata, "Despues de normalize_total")
    
    # Guardar datos normalizados (antes de log)
    adata.layers['normalized'] = adata.X.copy()
    
    # log1p
    sc.pp.log1p(adata)
    validate_no_nan(adata, "Despues de log1p")
    
    # Guardar en .raw
    adata.raw = adata.copy()
    
    print("[OK] Normalizacion completada")
    
    return adata


def select_highly_variable_genes(adata: ad.AnnData, n_top_genes: int = 3000) -> ad.AnnData:
    """
    Identifica genes altamente variables para reduccion dimensional
    
    Args:
        adata: Objeto AnnData normalizado
        n_top_genes: Numero de genes altamente variables a seleccionar
    
    Returns:
        AnnData con genes marcados en .var['highly_variable']
    """
    print(f"\nSeleccionando top {n_top_genes} genes altamente variables...")
    
    # Verificar existencia de columna batch antes de usarla
    batch_key = None
    if 'batch' in adata.obs.columns and adata.obs['batch'].nunique() > 1:
        batch_key = 'batch'
        print(f"  Usando batch_key='{batch_key}' ({adata.obs['batch'].nunique()} batches)")
    else:
        print("  Sin correccion de batch (un solo batch o columna no encontrada)")
    
    try:
        if batch_key:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=n_top_genes,
                subset=False,
                flavor='seurat_v3',
                batch_key=batch_key,
            )
        else:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=n_top_genes,
                subset=False,
                flavor='seurat_v3',
            )
    except Exception as e:
        print(f"[WARN] Error con seurat_v3, intentando flavor='cell_ranger': {e}")
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            subset=False,
            flavor='cell_ranger',
        )
    
    n_hvg = adata.var['highly_variable'].sum()
    print(f"[OK] Genes altamente variables identificados: {n_hvg}")
    
    return adata


# ============================================================================
# INTEGRACION CON HARMONY
# ============================================================================

def integrate_with_harmony(adata: ad.AnnData) -> ad.AnnData:
    """
    Integra datos de multiples batches usando Harmony
    
    Harmony corrige el efecto lote entre GSE210616 y GSE213688
    sin perder la variabilidad biologica real.
    
    Args:
        adata: Objeto AnnData con multiples batches
    
    Returns:
        AnnData con embeddings integrados en .obsm['X_pca_harmony']
    """
    print("\n" + "="*80)
    print("INTEGRACION CON HARMONY")
    print("="*80)
    
    # Verificar que existe informacion de batch
    batch_key = HARMONY_PARAMS.BATCH_KEY
    if batch_key not in adata.obs.columns:
        print(f"[WARN] No se encontro columna '{batch_key}', saltando integracion")
        return adata
    
    if adata.obs[batch_key].nunique() <= 1:
        print("[WARN] Solo hay un batch, saltando integracion")
        return adata
    
    # PCA sobre genes altamente variables
    print("\nCalculando PCA...")
    sc.pp.scale(adata, max_value=10)
    validate_no_nan(adata, "Despues de escalado")
    
    sc.tl.pca(adata, n_comps=HARMONY_PARAMS.N_PCS, svd_solver='arpack')
    
    # Verificar que PCA no tiene NaN
    if np.any(np.isnan(adata.obsm['X_pca'])):
        print("[WARN] PCA contiene NaN, aplicando limpieza")
        adata.obsm['X_pca'] = np.nan_to_num(adata.obsm['X_pca'], nan=0.0)
    
    # Aplicar Harmony
    print("\nAplicando Harmony...")
    try:
        import harmonypy as hm
        
        ho = hm.run_harmony(
            adata.obsm['X_pca'],
            adata.obs,
            batch_key,
            theta=HARMONY_PARAMS.THETA,
            lamb=HARMONY_PARAMS.LAMBDA_VALUE,
            max_iter_harmony=HARMONY_PARAMS.MAX_ITER,
            verbose=False,
        )
        
        adata.obsm['X_pca_harmony'] = ho.Z_corr.T
        
        # Verificar integridad
        if np.any(np.isnan(adata.obsm['X_pca_harmony'])):
            raise ValueError("Harmony produjo valores NaN")
        
        print("[OK] Harmony completado")
        
    except ImportError:
        print("[WARN] harmonypy no instalado, usando PCA sin correccion")
        adata.obsm['X_pca_harmony'] = adata.obsm['X_pca'].copy()
    
    except Exception as e:
        print(f"[WARN] Error en Harmony: {e}")
        print("  Usando PCA sin correccion")
        adata.obsm['X_pca_harmony'] = adata.obsm['X_pca'].copy()
    
    return adata


# ============================================================================
# PIPELINE COMPLETO DE PREPROCESAMIENTO
# ============================================================================

def preprocess_spatial_data(
    adata: ad.AnnData,
    do_integration: bool = True,
) -> ad.AnnData:
    """
    Pipeline completo de preprocesamiento
    
    Args:
        adata: AnnData crudo concatenado
        do_integration: Si aplicar Harmony
    
    Returns:
        AnnData preprocesado listo para deconvolucion
    """
    print("\n" + "="*80)
    print("PIPELINE DE PREPROCESAMIENTO")
    print("="*80)
    
    # QC
    adata = calculate_qc_metrics(adata)
    adata = filter_cells_and_genes(adata)
    
    # Normalizacion
    adata = normalize_data(adata)
    
    # HVG
    adata = select_highly_variable_genes(adata)
    
    # Integracion
    if do_integration:
        adata = integrate_with_harmony(adata)
    
    # Resumen final
    print("\n" + "="*80)
    print("PREPROCESAMIENTO COMPLETADO")
    print("="*80)
    print(f"  Spots finales: {adata.n_obs}")
    print(f"  Genes finales: {adata.n_vars}")
    print(f"  Genes HVG: {adata.var['highly_variable'].sum()}")
    if 'batch' in adata.obs.columns:
        print(f"  Batches: {adata.obs['batch'].nunique()}")
    
    return adata


# ============================================================================
# FUNCION PRINCIPAL
# ============================================================================

if __name__ == '__main__':
    """
    Ejemplo de uso del modulo de preprocesamiento
    """
    print("="*80)
    print("PREPROCESAMIENTO DE DATOS TNBC VISIUM")
    print("="*80)
    
    # Cargar GSE210616 (discovery)
    adata_discovery = load_visium_cohort(
        data_dir=PATHS.VISIUM_GSE210616,
        cohort_name='GSE210616',
    )
    
    # Cargar GSE213688 (validation)
    adata_validation = load_visium_cohort(
        data_dir=PATHS.VISIUM_GSE213688,
        cohort_name='GSE213688',
    )
    
    # Combinar cohortes
    print("\nCombinando cohortes...")
    adata_combined = ad.concat(
        [adata_discovery, adata_validation],
        join='inner',
        label='cohort',
        keys=['discovery', 'validation'],
    )
    
    # Preprocesar
    adata_processed = preprocess_spatial_data(adata_combined, do_integration=True)
    
    # Guardar
    output_path = PATHS.PROCESSED_DIR / 'combined_preprocessed.h5ad'
    PATHS.create_directories()
    adata_processed.write_h5ad(output_path)
    print(f"\n[OK] Datos guardados: {output_path}")
