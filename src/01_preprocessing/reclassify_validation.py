"""
================================================================================
RECLASSIFY VALIDATION — Re-clasifica GSE213688 con phenotype_classifier.py
================================================================================
Re-clasificar usando EXACTAMENTE la misma lógica que Discovery:
  1. calculate_all_mechanism_scores() → calcula 7 scores (Tumor, CD8, Silencing, etc.)
  2. normalize_scores_per_sample() → z-score por muestra
  3. classify_phenotype_mechanistic() → clasificación jerárquica con mismos params

NOTA SOBRE .raw:
  GSE213688 NO tiene .raw.
  
================================================================================
"""

import sys
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path

# ============================================================================
# IMPORTS DEL PIPELINE — NO reimplementar lógica
# ============================================================================
try:
    from config import PATHS, PHENOTYPE_PARAMS, SIGNATURES
    PROCESSED_DIR = PATHS.PROCESSED_DIR
    BASE_DIR = PATHS.BASE_DIR
except ImportError:
    BASE_DIR = Path("/home/external/frjimenez/fabian/genoma")
    PROCESSED_DIR = BASE_DIR / "data" / "processed"

from phenotype_classifier import (
    calculate_all_mechanism_scores,
    normalize_scores_per_sample,
    classify_phenotype_mechanistic,
    calculate_classification_confidence,
    summarize_phenotypes_by_sample,
)


def find_validation_h5ad() -> Path:
    """Busca el h5ad de validación deconvolucionado."""
    candidates = [
        # Deconvolucionado (preferido: tiene obsm con Cell2Location)
        BASE_DIR / "results/validation_gse213688/adata_gse213688_deconvolved.h5ad",
        BASE_DIR / "data/processed/adata_gse213688_deconvolved.h5ad",
        # Clasificado v2 (tiene obsm + scores viejos que sobreescribiremos)
        BASE_DIR / "results/validation_gse213688/adata_gse213688_classified_v2.h5ad",
        BASE_DIR / "results/validation_gse213688/adata_gse213688_classified.h5ad",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def ensure_sample_id(adata):
    """
    Asegura que adata.obs tiene columna 'sample_id' para z-normalización por muestra.
    
    Si no existe, intenta derivarla de:
    1. 'library_id' (estándar Visium)
    2. 'batch'
    3. Prefijo del barcode (antes del primer '-')
    """
    if 'sample_id' in adata.obs.columns:
        n = adata.obs['sample_id'].nunique()
        print(f"  [OK] sample_id presente: {n} muestras")
        return adata
    
    # Intentar derivar
    for alt in ['library_id', 'batch', 'sample', 'patient_id']:
        if alt in adata.obs.columns:
            adata.obs['sample_id'] = adata.obs[alt].astype(str)
            n = adata.obs['sample_id'].nunique()
            print(f"  [INFO] sample_id derivado de '{alt}': {n} muestras")
            return adata
    
    # Último recurso: extraer del barcode (formato ACGT-1_samplename o similar)
    barcodes = adata.obs_names.astype(str)
    if '-' in barcodes[0]:
        # Formato Visium: BARCODE-N o BARCODE-N_SAMPLEID
        parts = barcodes.str.split('-', n=1)
        suffixes = parts.str[1] if parts.str.len().min() >= 2 else None
        if suffixes is not None:
            # Si hay formato BARCODE-1_GSM12345, extraer GSM12345
            if '_' in str(suffixes.iloc[0]):
                sample_ids = suffixes.str.split('_', n=1).str[1]
            else:
                sample_ids = suffixes
            adata.obs['sample_id'] = sample_ids.values
            n = adata.obs['sample_id'].nunique()
            print(f"  [INFO] sample_id extraído de barcode suffixes: {n} muestras")
            return adata
    
    # Si todo falla, asignar un solo sample_id (normalización será global)
    print("  [WARN] No se pudo derivar sample_id. Usando ID único para todo el dataset.")
    print("         La z-normalización será GLOBAL, no por muestra.")
    adata.obs['sample_id'] = 'GSE213688_all'
    return adata


def remove_old_scores(adata):
    """Elimina scores y fenotipos de la clasificación anterior."""
    cols_to_remove = []
    for col in adata.obs.columns:
        if any(col.endswith(s) for s in ['_Score', '_Score_norm', '_Diff']):
            cols_to_remove.append(col)
    for col in ['Phenotype', 'phenotype', 'phenotype_v2', 'Immune_Score']:
        if col in adata.obs.columns:
            cols_to_remove.append(col)
    
    cols_to_remove = list(set(cols_to_remove) & set(adata.obs.columns))
    if cols_to_remove:
        print(f"  [INFO] Eliminando {len(cols_to_remove)} columnas de clasificación anterior:")
        for c in sorted(cols_to_remove):
            print(f"         - {c}")
        adata.obs = adata.obs.drop(columns=cols_to_remove)
    return adata


def main():
    print("=" * 80)
    print("RECLASSIFY VALIDATION — GSE213688")
    print("Usando EXACTAMENTE phenotype_classifier.py de Discovery")
    print("=" * 80)
    
    # 1. Encontrar datos
    h5ad_path = find_validation_h5ad()
    if h5ad_path is None:
        print("[ERROR] No se encontró h5ad de validación GSE213688")
        print("  Rutas buscadas:")
        print("    - results/validation_gse213688/adata_gse213688_deconvolved.h5ad")
        print("    - results/validation_gse213688/adata_gse213688_classified_v2.h5ad")
        sys.exit(1)
    
    print(f"\n[1/6] Cargando: {h5ad_path.name}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  Spots: {adata.n_obs:,} | Genes: {adata.n_vars:,}")
    
    # Diagnóstico de datos
    has_raw = adata.raw is not None
    x_min = float(adata.X.min()) if hasattr(adata.X, 'min') else 0
    x_max = float(adata.X.max()) if hasattr(adata.X, 'max') else 0
    print(f"  .raw existe: {has_raw}")
    print(f"  .X rango: [{x_min:.2f}, {x_max:.2f}]")
    if 'counts' in adata.layers:
        print(f"  .layers['counts']: presente")
    
    # Distribución anterior (si existe)
    for col in ['Phenotype', 'phenotype', 'phenotype_v2']:
        if col in adata.obs.columns:
            print(f"\n  Distribución ANTERIOR ({col}):")
            for p, n in adata.obs[col].value_counts().items():
                print(f"    {p}: {n:,} ({100*n/adata.n_obs:.1f}%)")
    
    # 2. Preparar: sample_id + eliminar scores viejos
    print(f"\n[2/6] Preparando datos...")
    adata = ensure_sample_id(adata)
    adata = remove_old_scores(adata)
    
    # 3. Calcular scores (MISMA función que Discovery)
    print(f"\n[3/6] Calculando scores de firmas génicas...")
    print(f"  Parámetros: TUMOR_PCT={PHENOTYPE_PARAMS.TUMOR_PERCENTILE}, "
          f"CD8_PCT={PHENOTYPE_PARAMS.CD8_PERCENTILE}, "
          f"AMBIGUITY={PHENOTYPE_PARAMS.COLD_AMBIGUITY_THRESHOLD}")
    
    # Reportar genes disponibles por firma
    for name, genes in [
        ('SILENCING_REPRESSORS', SIGNATURES.SILENCING_REPRESSORS),
        ('STING_PATHWAY', SIGNATURES.STING_PATHWAY),
        ('PHYSICAL_BARRIER', SIGNATURES.PHYSICAL_BARRIER),
        ('CD8_T_CELLS', SIGNATURES.CD8_T_CELLS),
        ('CHEMOKINE_SIGNALS', SIGNATURES.CHEMOKINE_SIGNALS),
    ]:
        available = [g for g in genes if g in adata.var_names]
        missing = [g for g in genes if g not in adata.var_names]
        print(f"  {name}: {len(available)}/{len(genes)} presentes", end='')
        if missing:
            print(f"  (faltan: {', '.join(missing)})")
        else:
            print()
    
    adata = calculate_all_mechanism_scores(adata)
    
    # 4. Normalizar por muestra (z-score)
    print(f"\n[4/6] Normalizando scores por muestra (z-score)...")
    adata = normalize_scores_per_sample(adata)
    
    # Diagnóstico post-normalización
    for col in ['Silencing_Score_norm', 'Barrier_Score_norm']:
        if col in adata.obs.columns:
            vals = adata.obs[col].values
            print(f"  {col}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
                  f"min={np.min(vals):.2f}, max={np.max(vals):.2f}")
    
    if 'Silencing_Score_norm' in adata.obs.columns and 'Barrier_Score_norm' in adata.obs.columns:
        diff = adata.obs['Silencing_Score_norm'] - adata.obs['Barrier_Score_norm']
        print(f"  Sil-Bar diff: mean={diff.mean():.4f}, std={diff.std():.4f}")
        print(f"  diff > 0.1: {(diff > 0.1).sum():,} spots → serán Desert")
        print(f"  diff < -0.1: {(diff < -0.1).sum():,} spots → serán Excluded")
        print(f"  |diff| <= 0.1: {(diff.abs() <= 0.1).sum():,} spots → serán Ambiguous")
    
    # 5. Clasificar (MISMA función que Discovery)
    print(f"\n[5/6] Clasificando fenotipos...")
    adata = classify_phenotype_mechanistic(adata, use_normalized=PHENOTYPE_PARAMS.NORMALIZE_SCORES)
    adata = calculate_classification_confidence(adata)
    
    # 6. Guardar
    output_dir = BASE_DIR / "results" / "validation_gse213688"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = output_dir / "adata_gse213688_classified_v3.h5ad"
    adata.write_h5ad(out_path)
    print(f"\n[6/6] Guardado: {out_path}")
    
    # Resumen de clasificación
    print(f"\n{'='*80}")
    print("DISTRIBUCIÓN FINAL (reclasificada con phenotype_classifier.py):")
    print(f"{'='*80}")
    for p, n in adata.obs['Phenotype'].value_counts().items():
        print(f"  {p}: {n:,} ({100*n/adata.n_obs:.1f}%)")
    
    n_desert = (adata.obs['Phenotype'] == 'Immune_Desert').sum()
    if n_desert == 0:
        print(f"\n  ATENCIÓN: Aún 0 Immune_Desert después de reclasificación.")
        print(f"  Posibles causas:")
        print(f"    - Los 15 pacientes de GSE213688 no tienen perfil Desert real")
        print(f"    - Necesita ajuste de parámetros (umbral de ambigüedad más alto)")
        print(f"    - Reportar como hallazgo: 'Validation cohort lacks Desert phenotype'")
    else:
        print(f"\n  {n_desert:,} Immune_Desert spots encontrados.")
        print(f"  La reclasificación restauró la distribución esperada.")
    
    # Generar resumen por muestra
    try:
        summary = summarize_phenotypes_by_sample(adata)
        summary_path = output_dir / "phenotype_summary_v3.csv"
        summary.to_csv(summary_path, index=False)
        print(f"  Resumen por muestra: {summary_path}")
    except Exception as e:
        print(f"  [WARN] No se pudo generar resumen por muestra: {e}")
    
    print(f"\n{'='*80}")
    print("INSTRUCCIONES SIGUIENTES:")
    print(f"{'='*80}")
    print(f"  1. Verificar distribución de fenotipos arriba")
    print(f"  2. Si n_Desert > 0: ejecutar validation.py (detectará v3 automáticamente)")
    print(f"  3. Si n_Desert = 0: discutir como limitación en el paper")


if __name__ == '__main__':
    main()
