"""
================================================================================
DECONVOLUCION CELULAR CON CELL2LOCATION 
================================================================================
Implementa deconvolucion celular espacial para estimar abundancias de:
- CD8+ T cells, CD4+ T cells, cDC1, Macrofagos, CAFs, Tumor

================================================================================
"""

import os
import gc
import shutil
import hashlib
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
import json
from scipy.sparse import issparse

# =============================================================================
# SEMILLAS DE DETERMINISMO
# =============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =============================================================================
# IMPORTACIONES Y OPTIMIZACIONES GPU
# =============================================================================

try:
    import torch
    
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        _gpu_name = torch.cuda.get_device_name(0)
        _gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print("=" * 70)
        print("CONFIGURACION GPU + REPRODUCIBILIDAD")
        print("=" * 70)
        print(f"  GPU: {_gpu_name} ({_gpu_mem:.1f} GB)")
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  Semilla: {RANDOM_SEED}")
        print("=" * 70 + "\n")
    
    import cell2location
    from cell2location.models import RegressionModel, Cell2location
    import scvi
    
except ImportError as e:
    print(f"[ERROR FATAL] {e}")
    raise

from config import CELL2LOC_PARAMS, CELL_PRESENCE_PARAMS, PATHS, SIGNATURES
from preprocessing import validate_no_nan

warnings.filterwarnings('ignore')


# =============================================================================
# FUNCIONES DE LIMPIEZA
# =============================================================================

def delete_corrupted_reference_model() -> None:
    """Elimina modelo de referencia corrupto."""
    print("\n[CLEANUP] Eliminando modelo de referencia...")
    for path in [PATHS.MODELS_DIR / 'reference_model', 
                 PATHS.MODELS_DIR / 'reference_adata.h5ad',
                 PATHS.MODELS_DIR / 'reference_metadata.json']:
        if path.exists():
            shutil.rmtree(path) if path.is_dir() else path.unlink()
            print(f"  [DEL] {path}")


def delete_corrupted_spatial_model() -> None:
    """Elimina modelo espacial corrupto."""
    print("\n[CLEANUP] Eliminando modelo espacial...")
    for path in [PATHS.MODELS_DIR / 'spatial_model',
                 PATHS.MODELS_DIR / 'spatial_metadata.json']:
        if path.exists():
            shutil.rmtree(path) if path.is_dir() else path.unlink()
            print(f"  [DEL] {path}")


# =============================================================================
# FUNCIONES DE HASH
# =============================================================================

def compute_reference_hash(factor_names: List[str], n_genes: int) -> str:
    content = f"factors:{','.join(sorted(factor_names))}|genes:{n_genes}"
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


def validate_spatial_model_consistency(current_ref_hash: str, metadata_path: Path) -> bool:
    if not metadata_path.exists():
        return False
    try:
        with open(metadata_path, 'r') as f:
            stored_hash = json.load(f).get('reference_hash')
        if stored_hash != current_ref_hash:
            print(f"\n[WARN] Hash mismatch - forzando reentrenamiento")
            return False
        return True
    except Exception:
        return False


# =============================================================================
# VERIFICACION DE ENTORNO
# =============================================================================

def setup_cell2location_environment() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("[ERROR] GPU NO DISPONIBLE")
    
    scvi.settings.seed = RANDOM_SEED
    scvi.settings.dl_pin_memory_gpu_training = True
    scvi.settings.verbosity = 20
    
    print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
    return torch.device('cuda')


# =============================================================================
# GUARDADO/CARGA DE MODELOS
# =============================================================================

def save_reference_results(adata_ref: ad.AnnData, mod_ref: RegressionModel, 
                           factor_names: List[str]) -> str:
    PATHS.create_directories()
    ref_hash = compute_reference_hash(factor_names, adata_ref.n_vars)
    
    mod_ref.save(str(PATHS.MODELS_DIR / 'reference_model'), overwrite=True)
    adata_ref.write_h5ad(PATHS.MODELS_DIR / 'reference_adata.h5ad')
    
    with open(PATHS.MODELS_DIR / 'reference_metadata.json', 'w') as f:
        json.dump({
            'factor_names': factor_names,
            'n_cells': int(adata_ref.n_obs),
            'n_genes': int(adata_ref.n_vars),
            'reference_hash': ref_hash,
            'random_seed': RANDOM_SEED,
        }, f, indent=2)
    
    print(f"[SAVE] Modelo referencia (hash={ref_hash})")
    return ref_hash


def load_reference_results() -> Tuple[ad.AnnData, Any, List[str], str]:
    print("\n[LOAD] Cargando modelo de referencia...")
    
    with open(PATHS.MODELS_DIR / 'reference_metadata.json', 'r') as f:
        metadata = json.load(f)
    factor_names = metadata['factor_names']
    ref_hash = metadata.get('reference_hash', '')
    
    adata_ref = sc.read_h5ad(PATHS.MODELS_DIR / 'reference_adata.h5ad')
    if 'means_per_cluster_mu_fg' not in adata_ref.varm:
        raise ValueError("adata no tiene means_per_cluster_mu_fg")
    
    mod_ref = RegressionModel.load(str(PATHS.MODELS_DIR / 'reference_model'))
    print(f"  [OK] {len(factor_names)} tipos, hash={ref_hash}")
    
    return adata_ref, mod_ref, factor_names, ref_hash


def check_reference_model_exists() -> bool:
    paths = [PATHS.MODELS_DIR / 'reference_model' / 'model.pt',
             PATHS.MODELS_DIR / 'reference_adata.h5ad',
             PATHS.MODELS_DIR / 'reference_metadata.json']
    exists = all(p.exists() for p in paths)
    if exists:
        print("\n[INFO] Modelo de referencia existente detectado")
    return exists


def save_spatial_results(adata_spatial: ad.AnnData, mod_spatial: Cell2location,
                         factor_names: List[str], reference_hash: str) -> None:
    PATHS.create_directories()
    mod_spatial.save(str(PATHS.MODELS_DIR / 'spatial_model'), overwrite=True)
    
    with open(PATHS.MODELS_DIR / 'spatial_metadata.json', 'w') as f:
        json.dump({
            'factor_names': factor_names,
            'n_spots': int(adata_spatial.n_obs),
            'n_genes': int(adata_spatial.n_vars),
            'training_complete': True,
            'reference_hash': reference_hash,
        }, f, indent=2)
    print(f"[SAVE] Modelo espacial")


def check_spatial_model_exists(reference_hash: str) -> bool:
    model_path = PATHS.MODELS_DIR / 'spatial_model' / 'model.pt'
    metadata_path = PATHS.MODELS_DIR / 'spatial_metadata.json'
    
    if not (model_path.exists() and metadata_path.exists()):
        return False
    if not validate_spatial_model_consistency(reference_hash, metadata_path):
        return False
    
    try:
        with open(metadata_path, 'r') as f:
            if not json.load(f).get('training_complete', False):
                return False
    except Exception:
        return False
    
    print("\n[INFO] Modelo espacial consistente detectado")
    return True


def load_spatial_model(adata_spatial: ad.AnnData) -> Tuple[Cell2location, List[str]]:
    print("\n[LOAD] Cargando modelo espacial...")
    with open(PATHS.MODELS_DIR / 'spatial_metadata.json', 'r') as f:
        factor_names = json.load(f)['factor_names']
    mod_spatial = Cell2location.load(str(PATHS.MODELS_DIR / 'spatial_model'), adata_spatial)
    print("  [OK] Modelo cargado")
    return mod_spatial, factor_names


# =============================================================================
# PREPARACION DE DATOS
# =============================================================================

def load_scrna_reference(reference_path: Path, min_counts: int = 500, 
                         min_genes: int = 200) -> ad.AnnData:
    """Carga referencia scRNA-seq con QC."""
    print("\n" + "=" * 80)
    print("CARGANDO REFERENCIA scRNA-seq")
    print("=" * 80)
    
    if reference_path.suffix == '.h5ad':
        adata_ref = sc.read_h5ad(reference_path)
    elif reference_path.suffix == '.h5':
        adata_ref = sc.read_10x_h5(reference_path)
    else:
        raise ValueError(f"Formato no soportado: {reference_path.suffix}")
    
    print(f"Datos crudos: {adata_ref.n_obs} células, {adata_ref.n_vars} genes")
    
    # QC
    sc.pp.filter_cells(adata_ref, min_counts=min_counts)
    sc.pp.filter_cells(adata_ref, min_genes=min_genes)
    sc.pp.filter_genes(adata_ref, min_cells=10)
    
    # MT filtering
    adata_ref.var['mt'] = adata_ref.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata_ref, qc_vars=['mt'], percent_top=None, 
                                log1p=False, inplace=True)
    adata_ref = adata_ref[adata_ref.obs['pct_counts_mt'] < 20, :].copy()
    
    print(f"Después de QC: {adata_ref.n_obs} células, {adata_ref.n_vars} genes")
    
    # CRITICO: Guardar counts crudos
    adata_ref.layers['counts'] = adata_ref.X.copy()
    print(f"[OK] Counts guardados: max={adata_ref.X.max()}, dtype={adata_ref.X.dtype}")
    
    # Cell type
    if 'cell_type' not in adata_ref.obs.columns:
        for col in ['celltype', 'CellType', 'cell_types', 'annotation', 'cluster']:
            if col in adata_ref.obs.columns:
                adata_ref.obs['cell_type'] = adata_ref.obs[col]
                break
        else:
            raise ValueError("No se encontró columna de tipo celular")
    
    print(f"Tipos celulares: {adata_ref.obs['cell_type'].nunique()}")
    
    return adata_ref


def prepare_reference_for_cell2location(
    adata_ref: ad.AnnData,
    batch_key: Optional[str] = None,
) -> ad.AnnData:
    """
    Prepara la referencia para Cell2Location (VERSION v2.9).
    
    FIX v2.9: subset=False + subset manual para evitar corrupción de layer
    
    FLUJO:
    1. Guardar counts en layer
    2. Calcular HVG con subset=False (solo marca, no filtra)
    3. Subset manual de genes
    4. Restaurar counts crudos en .X
    """
    print("\n[PREP] Preparando referencia para Cell2Location (v2.9)...")
    
    # 1. Asegurar integridad de counts crudos
    if 'counts' not in adata_ref.layers:
        print("  [INFO] Creando backup de counts en layers['counts']")
        adata_ref.layers['counts'] = adata_ref.X.copy()
    
    # 2. Identificar HVG SIN hacer subset (evita bug de corrupción)
    print("  [1/3] Identificando genes variables (seurat_v3, subset=False)...")
    try:
        sc.pp.highly_variable_genes(
            adata_ref,
            layer='counts',
            n_top_genes=12000,
            flavor='seurat_v3',
            batch_key=batch_key if (batch_key and batch_key in adata_ref.obs.columns) else None,
            subset=False,  # <-- FIX: No hacer subset aquí
        )
    except Exception as e:
        print(f"  [WARN] Fallo seurat_v3 ({e}), usando cell_ranger...")
        adata_norm = adata_ref.copy()
        sc.pp.normalize_total(adata_norm, target_sum=1e4)
        sc.pp.log1p(adata_norm)
        sc.pp.highly_variable_genes(adata_norm, n_top_genes=12000, flavor='cell_ranger', subset=False)
        adata_ref.var['highly_variable'] = adata_norm.var['highly_variable']
        del adata_norm
    
    # 3. Subset MANUAL de genes HVG
    print("  [2/3] Filtrando a genes HVG...")
    n_hvg = adata_ref.var['highly_variable'].sum()
    print(f"       {n_hvg} genes marcados como HVG")
    adata_ref = adata_ref[:, adata_ref.var['highly_variable']].copy()
    
    # 4. Restaurar counts crudos en .X (el subset puede haber copiado el layer)
    print("  [3/3] Restaurando counts crudos en .X...")
    adata_ref.X = adata_ref.layers['counts'].copy()
    
    # Asegurar dtype int64
    if issparse(adata_ref.X):
        adata_ref.X = adata_ref.X.astype(np.int64)
    else:
        adata_ref.X = np.asarray(adata_ref.X, dtype=np.int64)

    print(f"[OK] Referencia lista: {adata_ref.n_obs} células, {adata_ref.n_vars} genes")
    print(f"     X stats: Max={adata_ref.X.max()}, Dtype={adata_ref.X.dtype}")
    
    return adata_ref


# =============================================================================
# ENTRENAMIENTO MODELO DE REFERENCIA
# =============================================================================

def train_reference_model(adata_ref: ad.AnnData, 
                          cell_type_col: str = 'cell_type') -> Tuple[ad.AnnData, RegressionModel, List[str], str]:
    """Entrena modelo de referencia con counts crudos."""
    print("\n" + "=" * 80)
    print("ENTRENANDO MODELO DE REFERENCIA")
    print("=" * 80)
    
    # Verificar datos
    x_max = adata_ref.X.max()
    print(f"[CHECK] X max={x_max}, dtype={adata_ref.X.dtype}")
    
    if x_max < 100:
        print("[WARN] Max muy bajo - verificar datos")
    
    # Factor names
    if hasattr(adata_ref.obs[cell_type_col], 'cat'):
        factor_names = adata_ref.obs[cell_type_col].cat.categories.tolist()
    else:
        factor_names = sorted(adata_ref.obs[cell_type_col].unique().tolist())
    
    print(f"Tipos celulares ({len(factor_names)}): {factor_names[:5]}...")
    
    # Setup y entrenar
    RegressionModel.setup_anndata(adata_ref, labels_key=cell_type_col)
    mod_ref = RegressionModel(adata_ref)
    
    print(f"\nEntrenando ({CELL2LOC_PARAMS.REF_MAX_EPOCHS} épocas)...")
    mod_ref.train(
        max_epochs=CELL2LOC_PARAMS.REF_MAX_EPOCHS,
        batch_size=CELL2LOC_PARAMS.REF_BATCH_SIZE,
        train_size=CELL2LOC_PARAMS.REF_TRAIN_SIZE,
        lr=0.002,
        accelerator="gpu",
    )
    
    print("\nExportando expresión por tipo celular...")
    adata_ref = mod_ref.export_posterior(
        adata_ref,
        sample_kwargs={'num_samples': 1000, 'batch_size': 2500},
    )
    
    if 'means_per_cluster_mu_fg' not in adata_ref.varm:
        raise ValueError("No se generó means_per_cluster_mu_fg")
    
    ref_hash = save_reference_results(adata_ref, mod_ref, factor_names)
    
    return adata_ref, mod_ref, factor_names, ref_hash


# =============================================================================
# DECONVOLUCION ESPACIAL
# =============================================================================

def prepare_spatial_for_cell2location(adata_spatial: ad.AnnData, adata_ref: ad.AnnData,
                                       factor_names: List[str]) -> Tuple[ad.AnnData, pd.DataFrame]:
    """
    Prepara datos espaciales (VERSION v2.13 - NUMPY FORCE).
    
    FIX DEFINITIVO: Extrae .values para evitar que la alineación
    de índices de Pandas genere NaNs silenciosos.
    """
    print("\n[PREP] Preparando datos espaciales (v2.13 Numpy Force)...")
    
    # 1. Identificar genes comunes
    common_genes = adata_spatial.var_names.intersection(adata_ref.var_names)
    print(f"  Genes comunes detectados: {len(common_genes)}")
    
    if len(common_genes) == 0:
        raise RuntimeError("[ERROR] No hay genes comunes. Revisar nombres de genes.")

    # 2. ALINEACIÓN POSICIONAL (Bypass de Pandas Index)
    print("  [FIX] Extrayendo firmas por posición numérica...")
    
    # Mapa: Gen -> Posición en Referencia
    gene_to_pos = {gene: i for i, gene in enumerate(adata_ref.var_names)}
    
    # Índices enteros para los genes comunes
    ref_indices = [gene_to_pos[g] for g in common_genes]
    
    if 'means_per_cluster_mu_fg' not in adata_ref.varm:
        raise ValueError("Falta 'means_per_cluster_mu_fg' en reference.varm")
    
    # EXTRACCIÓN SEGURA DE DATOS (Numpy puro)
    raw_signatures = adata_ref.varm['means_per_cluster_mu_fg']
    
    # Si es DataFrame/Series, sacamos los valores puros
    if hasattr(raw_signatures, 'values'): 
        raw_signatures = raw_signatures.values
    elif hasattr(raw_signatures, 'to_numpy'):
        raw_signatures = raw_signatures.to_numpy()
    elif hasattr(raw_signatures, 'toarray'):
        raw_signatures = raw_signatures.toarray()
    
    # Slicing numérico directo
    signatures_subset = raw_signatures[ref_indices, :]
    
    # Construir DataFrame limpio y alineado
    inf_aver = pd.DataFrame(
        signatures_subset, 
        index=common_genes, 
        columns=factor_names
    )
    
    # Verificar integridad
    if inf_aver.isna().all().all():
        raise RuntimeError("[ERROR] La extracción generó todo NaNs")
    
    # 3. SAFETY FLOOR (Fix del error rate=0)
    min_val = inf_aver.min().min()
    print(f"  [INFO] Mínimo firma original: {min_val:.2e}")
    
    # Aplicar clip para seguridad numérica en GPU
    inf_aver = inf_aver.clip(lower=1e-5)
    print(f"  [FIX] Aplicado suelo de seguridad (clip 1e-5)")

    # 4. Recortar objeto espacial a genes comunes
    adata_spatial = adata_spatial[:, common_genes].copy()
    
    # 5. Preparar counts espaciales
    if 'counts' not in adata_spatial.layers:
        raise ValueError("[ERROR] Layer 'counts' no encontrado en spatial")
    
    counts_data = adata_spatial.layers['counts'].copy()
    if issparse(counts_data):
        counts_data = counts_data.toarray()
    
    counts_data = np.round(counts_data).astype(np.int32)
    counts_data = np.maximum(counts_data, 0)
    adata_spatial.X = counts_data
    
    print(f"  [OK] Counts: max={adata_spatial.X.max()}, dtype={adata_spatial.X.dtype}")
    print(f"  [OK] inf_aver: shape={inf_aver.shape}, min={inf_aver.min().min():.2e}")
    
    return adata_spatial, inf_aver


def run_cell2location_spatial(adata_spatial: ad.AnnData, inf_aver: pd.DataFrame,
                               factor_names: List[str], reference_hash: str) -> Tuple[ad.AnnData, Cell2location]:
    """Entrena modelo espacial."""
    print("\n" + "=" * 80)
    print("DECONVOLUCION ESPACIAL")
    print("=" * 80)
    print(f"  Spots: {adata_spatial.n_obs}, Genes: {adata_spatial.n_vars}")
    
    Cell2location.setup_anndata(adata_spatial)
    
    mod_spatial = Cell2location(
        adata_spatial,
        cell_state_df=inf_aver,
        N_cells_per_location=CELL2LOC_PARAMS.N_CELLS_PER_LOCATION,
        detection_alpha=CELL2LOC_PARAMS.DETECTION_ALPHA,
    )
    
    print(f"\nEntrenando ({CELL2LOC_PARAMS.SPATIAL_MAX_EPOCHS} épocas)...")
    mod_spatial.train(
        max_epochs=CELL2LOC_PARAMS.SPATIAL_MAX_EPOCHS,
        batch_size=CELL2LOC_PARAMS.SPATIAL_BATCH_SIZE,
        train_size=1.0,
        lr=0.002,
        accelerator="gpu",
    )
    
    print("\n[OK] Entrenamiento completado")
    
    # FIX v2.15: GUARDAR MODELO INMEDIATAMENTE después del entrenamiento
    # Esto asegura que si export_posterior falla, el modelo ya está en disco
    print("\n[SAVE] Guardando modelo espacial (seguridad pre-export)...")
    try:
        mod_spatial.save(str(PATHS.MODELS_DIR / 'spatial_model'), overwrite=True)
        print("[OK] Modelo guardado exitosamente")
    except Exception as e:
        print(f"[WARN] No se pudo guardar modelo: {e}")
    
    print("\nExportando abundancias...")
    
    # FIX v2.14: El argumento 'quantiles' ya no existe en versiones recientes de cell2location
    # La exportación simple calcula means y stds automáticamente
    try:
        adata_spatial = mod_spatial.export_posterior(
            adata_spatial,
            sample_kwargs={
                'num_samples': 1000,
                'batch_size': min(2500, mod_spatial.adata.n_obs),
                'use_gpu': CELL2LOC_PARAMS.USE_GPU,
            }
        )
    except TypeError as e:
        # Fallback si hay problemas con argumentos
        print(f"[WARN] export_posterior con kwargs falló: {e}")
        print("       Intentando exportación simple...")
        adata_spatial = mod_spatial.export_posterior(adata_spatial)
    
    print("[OK] Abundancias exportadas")
    
    adata_spatial.uns['mod'] = {'factor_names': factor_names}
    
    for key in adata_spatial.obsm.keys():
        if 'cell_abundance' in key or 'means' in key:
            arr = adata_spatial.obsm[key]
            if np.any(np.isnan(arr)):
                arr[np.isnan(arr)] = 0.0
                adata_spatial.obsm[key] = arr
    
    save_spatial_results(adata_spatial, mod_spatial, factor_names, reference_hash)
    
    return adata_spatial, mod_spatial


# =============================================================================
# POST-PROCESAMIENTO
# =============================================================================

def extract_key_cell_abundances(adata_spatial: ad.AnnData, 
                                 factor_names: List[str]) -> ad.AnnData:
    """Extrae abundancias de tipos celulares clave."""
    print("\nExtrayendo abundancias clave...")
    
    cell_type_mappings = {
        'CD8_T': ['CD8_T', 'CD8+ T', 'CD8 T', 'T cells CD8+', 'CD8+ T cells', 'Cytotoxic T'],
        'CD4_T': ['CD4_T', 'CD4+ T', 'CD4 T', 'T cells CD4+', 'CD4+ T cells', 'Helper T'],
        'cDC1': ['cDC1', 'DC1', 'cDC', 'DC', 'Dendritic', 'Dendritic cells'],
        'CAF': ['CAF', 'Fibroblast', 'myCAF', 'iCAF', 'Fibroblasts', 'Stromal'],
        'Macrophage': ['Macrophage', 'Macro', 'TAM', 'M1', 'M2', 'Macrophages', 'Myeloid'],
        'Tumor': ['Tumor', 'Cancer', 'Epithelial', 'Malignant', 'Neoplastic'],
    }
    
    for quantile in ['q05', 'q50', 'q95', 'means']:
        obsm_key = f'{quantile}_cell_abundance_w_sf' if quantile != 'means' else 'means_cell_abundance_w_sf'
        
        if obsm_key not in adata_spatial.obsm:
            continue
        
        abundance_df = pd.DataFrame(
            adata_spatial.obsm[obsm_key],
            index=adata_spatial.obs_names,
            columns=factor_names,
        )
        
        for key, possible_names in cell_type_mappings.items():
            found_col = None
            for name in possible_names:
                if name in abundance_df.columns:
                    found_col = name
                    break
                for col in abundance_df.columns:
                    if name.lower() in col.lower():
                        found_col = col
                        break
                if found_col:
                    break
            
            if found_col:
                adata_spatial.obs[f'{key}_{quantile}'] = abundance_df[found_col].values
                if quantile == 'q50':
                    print(f"  [OK] {key}_q50 ← '{found_col}'")
    
    return adata_spatial


def assess_cell_presence(adata: ad.AnnData, cell_type: str) -> pd.DataFrame:
    """Evalúa presencia celular."""
    q05_col = f'{cell_type}_q05'
    if q05_col not in adata.obs.columns:
        return pd.DataFrame()
    
    results = pd.DataFrame({
        'spot_id': adata.obs_names,
        'q05': adata.obs[f'{cell_type}_q05'].values,
        'q50': adata.obs[f'{cell_type}_q50'].values,
        'q95': adata.obs[f'{cell_type}_q95'].values,
    })
    results['nUMI_est'] = results['q50'] * 10
    
    def classify(row):
        if row['q05'] > CELL_PRESENCE_PARAMS.Q05_ABUNDANCE_THRESHOLD:
            return 'substantial'
        elif row['q95'] > CELL_PRESENCE_PARAMS.Q95_UPPER_THRESHOLD:
            return 'low_confidence'
        elif row['nUMI_est'] < CELL_PRESENCE_PARAMS.NUMI_THRESHOLD:
            return 'limited_evidence'
        return 'uncertain'
    
    results['presence_category'] = results.apply(classify, axis=1)
    
    summary = results['presence_category'].value_counts()
    print(f"\n  {cell_type}:")
    for cat, count in summary.items():
        print(f"    {cat:20s}: {count:6d} ({count/len(results)*100:5.1f}%)")
    
    return results


def assess_all_cell_types_presence(adata: ad.AnnData) -> Dict[str, pd.DataFrame]:
    """Evalúa presencia para todos los tipos celulares."""
    print("\n" + "=" * 80)
    print("EVALUACION DE PRESENCIA CELULAR")
    print("=" * 80)
    
    results = {}
    for ct in ['CD8_T', 'CD4_T', 'cDC1', 'CAF', 'Macrophage']:
        df = assess_cell_presence(adata, ct)
        if len(df) > 0:
            results[ct] = df
    
    if results:
        summary_data = [{
            'cell_type': ct,
            'n_substantial': df['presence_category'].value_counts().get('substantial', 0),
            'pct_substantial': df['presence_category'].value_counts().get('substantial', 0) / len(df) * 100,
        } for ct, df in results.items()]
        
        PATHS.create_directories()
        pd.DataFrame(summary_data).to_csv(
            PATHS.TABLES_DIR / 'cell_presence_assessment_summary.csv', index=False)
    
    return results


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_complete_deconvolution(adata_spatial: ad.AnnData, reference_path: Path) -> ad.AnnData:
    """Pipeline completo de deconvolución v2.10."""
    print("\n" + "=" * 80)
    print("PIPELINE DE DECONVOLUCION CELULAR v2.15")
    print("=" * 80)
    print("  [FIX] Extracción Numpy pura + clip de seguridad")
    print("=" * 80)
    
    setup_cell2location_environment()
    
    # FASE 1: MODELO DE REFERENCIA
    adata_ref = None
    ref_hash = None
    
    if check_reference_model_exists():
        try:
            adata_ref, mod_ref, factor_names, ref_hash = load_reference_results()
        except Exception as e:
            print(f"\n[ERROR] Carga fallida: {e}")
            delete_corrupted_reference_model()
    
    if adata_ref is None:
        print("\n[TRAIN] Entrenando modelo de referencia...")
        adata_ref = load_scrna_reference(reference_path)
        adata_ref = prepare_reference_for_cell2location(adata_ref)
        adata_ref, mod_ref, factor_names, ref_hash = train_reference_model(adata_ref)
    
    print(f"\n[INFO] Hash: {ref_hash}")
    
    # FASE 2: MODELO ESPACIAL
    adata_spatial_prep, inf_aver = prepare_spatial_for_cell2location(
        adata_spatial, adata_ref, factor_names
    )
    
    spatial_loaded = False
    if check_spatial_model_exists(ref_hash):
        try:
            mod_spatial, factor_names_loaded = load_spatial_model(adata_spatial_prep)
            
            print("\nExportando abundancias...")
            adata_spatial_prep = mod_spatial.export_posterior(
                adata_spatial_prep,
                sample_kwargs={'num_samples': 1000, 'batch_size': mod_spatial.adata.n_obs,
                               'return_numpy_array': False},
                export_slot='quantiles',
                quantiles=list(CELL2LOC_PARAMS.QUANTILES),
            )
            adata_spatial_prep = mod_spatial.export_posterior(
                adata_spatial_prep,
                sample_kwargs={'num_samples': 1000, 'batch_size': mod_spatial.adata.n_obs},
                add_to_obsm=['means', 'stds'],
            )
            adata_spatial_prep.uns['mod'] = {'factor_names': factor_names_loaded}
            factor_names = factor_names_loaded
            spatial_loaded = True
        except Exception as e:
            print(f"\n[ERROR] Carga fallida: {e}")
            delete_corrupted_spatial_model()
    
    if not spatial_loaded:
        print("\n[TRAIN] Entrenando modelo espacial...")
        adata_spatial_prep, mod_spatial = run_cell2location_spatial(
            adata_spatial_prep, inf_aver, factor_names, ref_hash
        )
    
    # FASE 3: POST-PROCESAMIENTO
    adata_spatial_prep = extract_key_cell_abundances(adata_spatial_prep, factor_names)
    assess_all_cell_types_presence(adata_spatial_prep)
    
    PATHS.create_directories()
    output_path = PATHS.PROCESSED_DIR / 'adata_with_deconvolution.h5ad'
    adata_spatial_prep.write_h5ad(output_path)
    print(f"\n[SAVE] {output_path}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print("\n" + "=" * 80)
    print("[OK] DECONVOLUCION COMPLETADA")
    print("=" * 80)
    
    return adata_spatial_prep


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    adata_spatial = sc.read_h5ad(PATHS.PROCESSED_DIR / 'adata_preprocessed.h5ad')
    adata_deconv = run_complete_deconvolution(
        adata_spatial,
        reference_path=PATHS.SCRNA_GSE176078 / 'reference.h5ad',
    )
