"""
fix_validation_celltype_normalization.py
Recalcula Cohen's d para Desert vs Excluded usando proporciones de cell types
(abundancia / total por spot) para corregir sesgo sistemático por library size.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from scipy.stats import mannwhitneyu

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = Path("/home/external/frjimenez/fabian/genoma")
VAL_ADATA   = BASE / "results/validation_gse213688/adata_gse213688_classified_v3.h5ad"
OUT_DIR     = BASE / "results/validation_final/tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Discovery reference d values ──────────────────────────────────────────────
DISCOVERY_D = {
    'CAF': -0.624, 'Endothelial': -0.331, 'PVL': -0.151,
    'B_Cell': -0.101, 'Macrophage': -0.044, 'Myeloid_Cycling': 0.024,
    'CD4_T': 0.071, 'CD8_T': 0.073, 'Monocyte': 0.082,
    'cDC1': 0.128, 'NKT': 0.156, 'NK': 0.243,
    'T_Cell_Cycling': 0.333, 'Normal_Epithelial': 0.471, 'Tumor': 0.511,
}

# ── Stats helpers ──────────────────────────────────────────────────────────────
def cohens_d_pooled(g1, g2):
    g1 = np.asarray(g1, float); g2 = np.asarray(g2, float)
    g1 = g1[np.isfinite(g1)];   g2 = g2[np.isfinite(g2)]
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    sp = np.sqrt(((n1-1)*np.var(g1, ddof=1) + (n2-1)*np.var(g2, ddof=1)) / (n1+n2-2))
    return float((g1.mean() - g2.mean()) / sp) if sp > 1e-10 else 0.0

def safe_mw(a, b):
    try:
        _, p = mannwhitneyu(a, b, alternative='two-sided')
        return float(p)
    except Exception:
        return np.nan

# ── Main ───────────────────────────────────────────────────────────────────────
print("=" * 70)
print("FIX 1: CELL TYPE NORMALIZATION — PROPORCIONES")
print("=" * 70)

print(f"\nCargando: {VAL_ADATA}")
adata = sc.read_h5ad(VAL_ADATA)
print(f"  Spots: {adata.n_obs:,} | Fenotipos: {adata.obs['Phenotype'].value_counts().to_dict()}")

# ── Extraer abundancias ────────────────────────────────────────────────────────
OBSM_KEY = 'means_cell_abundance_w_sf'
if OBSM_KEY not in adata.obsm:
    raise KeyError(f"obsm['{OBSM_KEY}'] no encontrado. Keys: {list(adata.obsm.keys())}")

ab = adata.obsm[OBSM_KEY]
if not isinstance(ab, pd.DataFrame):
    ab = pd.DataFrame(ab, index=adata.obs_names)

# Limpiar nombres de columnas (strip prefijos)
clean = {}
for c in ab.columns:
    name = c
    for prefix in ['meanscell_abundance_w_sf_', 'means_cell_abundance_w_sf_']:
        name = name.replace(prefix, '')
    clean[c] = name
ab = ab.rename(columns=clean)

print(f"\n  Cell types disponibles ({len(ab.columns)}): {list(ab.columns)}")

# ── Máscaras de fenotipo ───────────────────────────────────────────────────────
desert_mask   = adata.obs['Phenotype'] == 'Immune_Desert'
excluded_mask = adata.obs['Phenotype'] == 'Immune_Excluded'
print(f"\n  Desert: {desert_mask.sum():,} | Excluded: {excluded_mask.sum():,}")

# ── Abundancia total por spot ──────────────────────────────────────────────────
total_ab = ab.sum(axis=1)
print(f"\n  Abundancia TOTAL por spot:")
for pheno, mask in [('Desert', desert_mask), ('Excluded', excluded_mask), ('Inflamed', adata.obs['Phenotype']=='Inflamed')]:
    vals = total_ab[mask]
    print(f"    {pheno}: median={vals.median():.2f}, mean={vals.mean():.2f}")

# ── Proporciones ──────────────────────────────────────────────────────────────
total_safe = total_ab.replace(0, np.nan)
ab_prop = ab.div(total_safe, axis=0)

print(f"\n  Proporción CAF — Desert vs Excluded:")
print(f"    Desert  raw median:  {ab.loc[desert_mask,   'CAF'].median():.4f}")
print(f"    Excluded raw median: {ab.loc[excluded_mask, 'CAF'].median():.4f}")
print(f"    Desert  prop median: {ab_prop.loc[desert_mask,   'CAF'].median():.4f}")
print(f"    Excluded prop median:{ab_prop.loc[excluded_mask, 'CAF'].median():.4f}")

# ── Calcular d para todos los cell types ──────────────────────────────────────
rows = []
for ct in ab.columns:
    disc_d = DISCOVERY_D.get(ct, np.nan)

    # Raw
    r_des = ab.loc[desert_mask,   ct].values.astype(float)
    r_exc = ab.loc[excluded_mask, ct].values.astype(float)
    d_raw = cohens_d_pooled(r_des, r_exc)
    p_raw = safe_mw(r_des[np.isfinite(r_des)], r_exc[np.isfinite(r_exc)])

    # Proportion-normalized
    p_des = ab_prop.loc[desert_mask,   ct].values.astype(float)
    p_exc = ab_prop.loc[excluded_mask, ct].values.astype(float)
    p_des = p_des[np.isfinite(p_des)]; p_exc = p_exc[np.isfinite(p_exc)]
    d_prop = cohens_d_pooled(p_des, p_exc)
    p_prop = safe_mw(p_des, p_exc)

    # Concordancia con Discovery (mismo signo)
    concordant_raw  = (np.sign(d_raw)  == np.sign(disc_d)) if np.isfinite(disc_d) and np.isfinite(d_raw)  else False
    concordant_prop = (np.sign(d_prop) == np.sign(disc_d)) if np.isfinite(disc_d) and np.isfinite(d_prop) else False

    rows.append({
        'cell_type':             ct,
        'discovery_d':           disc_d,
        'd_raw':                 round(d_raw,  4),
        'p_raw':                 p_raw,
        'd_proportion':          round(d_prop, 4),
        'p_proportion':          p_prop,
        'concordant_raw':        concordant_raw,
        'concordant_proportion': concordant_prop,
        'sign_changed':          (np.sign(d_raw) != np.sign(d_prop)) if np.isfinite(d_raw) and np.isfinite(d_prop) else False,
    })

df = pd.DataFrame(rows).sort_values('discovery_d')

# ── Resumen ───────────────────────────────────────────────────────────────────
n_conc_raw  = df['concordant_raw'].sum()
n_conc_prop = df['concordant_proportion'].sum()
n_changed   = df['sign_changed'].sum()

print("\n" + "=" * 70)
print("RESULTADOS")
print("=" * 70)
print(df[['cell_type','discovery_d','d_raw','d_proportion',
          'concordant_raw','concordant_proportion','sign_changed']].to_string(index=False))

print(f"\n  Concordantes con Discovery (mismo signo):")
print(f"    Raw:         {n_conc_raw}/15")
print(f"    Proportion:  {n_conc_prop}/15")
print(f"    Cambios de signo tras normalizar: {n_changed}/15")

# Cell types que cambian de signo
changed = df[df['sign_changed']]
if not changed.empty:
    print(f"\n  Cell types que cambian de signo:")
    for _, r in changed.iterrows():
        print(f"    {r['cell_type']}: raw={r['d_raw']:.3f} → prop={r['d_proportion']:.3f} "
              f"(Discovery={r['discovery_d']:.3f})")

# ── Guardar ───────────────────────────────────────────────────────────────────
out_csv = OUT_DIR / 'celltype_effect_sizes_normalized.csv'
df.to_csv(out_csv, index=False)
print(f"\n  Guardado: {out_csv}")

print("\n FIX 1 COMPLETADO")
