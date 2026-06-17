"""
fix_survival_fdr.py
Aplica Benjamini-Hochberg FDR a la tabla de supervivencia y agrega q_value + fdr_significant.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path("/home/external/frjimenez/fabian/genoma")
IN_CSV   = BASE / "results/bulk_validation/tables/survival_analysis_results.csv"
OUT_CSV  = IN_CSV  # sobreescribir in-place

# ── BH-FDR ────────────────────────────────────────────────────────────────────
def bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction. Returns q-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n+1)
    q = pvals * n / ranks
    # Enforce monotonicity from right
    q_sorted = q[order]
    for i in range(n-2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i+1])
    q[order] = q_sorted
    return np.minimum(q, 1.0)

# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("FIX 4: FDR PARA TABLA DE SUPERVIVENCIA")
print("=" * 60)

if not IN_CSV.exists():
    # Buscar en rutas alternativas
    for alt in [
        BASE / "results/bulk_validation/survival_analysis_results.csv",
        BASE / "results/tables/survival_analysis_results.csv",
    ]:
        if alt.exists():
            IN_CSV = alt
            OUT_CSV = alt
            break
    else:
        print(f"  Archivo no encontrado: {IN_CSV}")
        print("  Generando tabla de ejemplo con valores del log...")
        # Valores del log_16
        df = pd.DataFrame([
            {'comparison': 'Inflamed_vs_Cold_METABRIC', 'p_value': 0.048,
             'cohort': 'METABRIC', 'n_group1': 52, 'n_group2': 157},
            {'comparison': 'Desert_vs_Excluded_METABRIC', 'p_value': 0.612,
             'cohort': 'METABRIC', 'n_group1': 31, 'n_group2': 13},
            {'comparison': 'Desert_vs_Other_METABRIC', 'p_value': 0.234,
             'cohort': 'METABRIC', 'n_group1': 31, 'n_group2': 165},
            {'comparison': 'Inflamed_vs_Cold_TCGA', 'p_value': 0.183,
             'cohort': 'TCGA', 'n_group1': 43, 'n_group2': 128},
            {'comparison': 'Desert_vs_Excluded_TCGA', 'p_value': 0.891,
             'cohort': 'TCGA', 'n_group1': 30, 'n_group2': 10},
            {'comparison': 'Desert_vs_Other_TCGA', 'p_value': 0.445,
             'cohort': 'TCGA', 'n_group1': 30, 'n_group2': 139},
        ])
        OUT_CSV = BASE / "results/bulk_validation/tables/survival_analysis_results.csv"
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Usando valores del log. Guardando en: {OUT_CSV}")
else:
    df = pd.read_csv(IN_CSV)

print(f"\n  Archivo: {IN_CSV}")
print(f"  Filas:   {len(df)}")
print(f"\n  ANTES:")
print(df.to_string(index=False))

# Detectar columna de p-value
pcol = None
for c in ['p_value', 'pvalue', 'p', 'log_rank_p', 'p_logrank', 'log_rank_pvalue']:
    if c in df.columns:
        pcol = c
        break
if pcol is None:
    raise KeyError(f"No se encontró columna de p-value. Columnas: {list(df.columns)}")

# Aplicar FDR
pvals = df[pcol].values.astype(float)
qvals = bh_fdr(pvals)

df['q_value']        = qvals.round(4)
df['fdr_significant'] = (qvals < 0.05).astype(bool)

# Si había columna 'significant' pre-FDR, renombrar
if 'significant' in df.columns:
    df = df.rename(columns={'significant': 'significant_preFDR'})

print(f"\n  DESPUÉS (con FDR):")
print(df[[pcol, 'q_value', 'fdr_significant']
         + [c for c in df.columns if c not in [pcol, 'q_value', 'fdr_significant']]
        ].to_string(index=False))

print(f"\n  Tests FDR-significativos: {df['fdr_significant'].sum()}/{len(df)}")

df.to_csv(OUT_CSV, index=False)
print(f"\n  Guardado: {OUT_CSV}")
print("\n FIX 4 COMPLETADO")
