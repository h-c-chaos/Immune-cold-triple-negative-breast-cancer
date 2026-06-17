"""
fix_patient_level_validation.py
Calcula CAF patient-level en validation (paired Wilcoxon, d, bootstrap CI, figura).
"""

import numpy as np
import pandas as pd
import scanpy as sc
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wilcoxon, mannwhitneyu

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path("/home/external/frjimenez/fabian/genoma")
VAL_ADATA = BASE / "results/validation_gse213688/adata_gse213688_classified_v3.h5ad"
OUT_DIR   = BASE / "results/validation_final"
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

# ── Stats helpers ──────────────────────────────────────────────────────────────
def cohens_d_pooled(g1, g2):
    g1 = np.asarray(g1, float); g2 = np.asarray(g2, float)
    g1 = g1[np.isfinite(g1)];   g2 = g2[np.isfinite(g2)]
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    sp = np.sqrt(((n1-1)*np.var(g1,ddof=1)+(n2-1)*np.var(g2,ddof=1))/(n1+n2-2))
    return float((g1.mean()-g2.mean())/sp) if sp > 1e-10 else 0.0

def bootstrap_ci(d_vals_des, d_vals_exc, n_boot=1000, seed=42):
    """Bootstrap 95% CI for Cohen's d (paired, patient-level medians)."""
    rng = np.random.default_rng(seed)
    n = len(d_vals_des)
    ds = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ds.append(cohens_d_pooled(d_vals_des[idx], d_vals_exc[idx]))
    ds = np.array([x for x in ds if np.isfinite(x)])
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))

# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("FIX 5: PATIENT-LEVEL CAF VALIDATION")
print("=" * 70)

print(f"\nCargando: {VAL_ADATA}")
adata = sc.read_h5ad(VAL_ADATA)
print(f"  Spots: {adata.n_obs:,}")

# CAF abundance
OBSM_KEY = 'means_cell_abundance_w_sf'
if OBSM_KEY not in adata.obsm:
    raise KeyError(f"obsm['{OBSM_KEY}'] no encontrado")

ab = adata.obsm[OBSM_KEY]
if not isinstance(ab, pd.DataFrame):
    ab = pd.DataFrame(ab, index=adata.obs_names)

caf_cols = [c for c in ab.columns if 'caf' in c.lower()]
if not caf_cols:
    raise KeyError(f"No CAF column found in {list(ab.columns[:5])}")
caf_col = caf_cols[0]
print(f"  CAF column: {caf_col}")

adata.obs['_CAF'] = ab[caf_col].values

# ── Patient-level medianas ────────────────────────────────────────────────────
SAMPLE_COL = 'sample_id'
rows = []
desert_medians   = []
excluded_medians = []
patients_both = []

for sid in sorted(adata.obs[SAMPLE_COL].unique()):
    mask = adata.obs[SAMPLE_COL] == sid
    adata_p = adata[mask]
    phenos  = adata_p.obs['Phenotype']

    n_des = (phenos == 'Immune_Desert').sum()
    n_exc = (phenos == 'Immune_Excluded').sum()

    caf_des = adata_p.obs.loc[phenos == 'Immune_Desert',   '_CAF'].dropna()
    caf_exc = adata_p.obs.loc[phenos == 'Immune_Excluded', '_CAF'].dropna()

    med_des = float(caf_des.median()) if len(caf_des) > 0 else np.nan
    med_exc = float(caf_exc.median()) if len(caf_exc) > 0 else np.nan

    rows.append({
        'patient':         sid,
        'n_desert_spots':  int(n_des),
        'n_excluded_spots':int(n_exc),
        'CAF_median_desert':   round(med_des, 4) if np.isfinite(med_des)  else np.nan,
        'CAF_median_excluded': round(med_exc, 4) if np.isfinite(med_exc) else np.nan,
        'has_both': np.isfinite(med_des) and np.isfinite(med_exc),
    })

    if np.isfinite(med_des) and np.isfinite(med_exc):
        desert_medians.append(med_des)
        excluded_medians.append(med_exc)
        patients_both.append(sid)

df = pd.DataFrame(rows)
print(f"\n  Pacientes totales: {len(df)}")
print(f"  Con ambos fenotipos (≥1 spot cada uno): {len(patients_both)}")
print(f"\n  Patient-level CAF:")
print(f"    Desert   median-of-medians: {np.median(desert_medians):.4f}")
print(f"    Excluded median-of-medians: {np.median(excluded_medians):.4f}")

# ── Tests estadísticos ────────────────────────────────────────────────────────
des_arr = np.array(desert_medians)
exc_arr = np.array(excluded_medians)
diff    = exc_arr - des_arr

# Paired Wilcoxon
try:
    wstat, wpval = wilcoxon(des_arr, exc_arr, alternative='two-sided')
    print(f"\n  Paired Wilcoxon: stat={wstat:.1f}, p={wpval:.4e}")
except Exception as e:
    wpval = np.nan
    print(f"  Wilcoxon error: {e}")

# Cohen's d (patient-level)
d_patient = cohens_d_pooled(des_arr, exc_arr)
ci_lo, ci_hi = bootstrap_ci(des_arr, exc_arr)
print(f"  Cohen's d (patient-level): {d_patient:.4f}")
print(f"  Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

# Dirección: cuántos pacientes tienen Excluded > Desert
n_exc_gt = (exc_arr > des_arr).sum()
print(f"  Excluded > Desert: {n_exc_gt}/{len(patients_both)} pacientes")

# Discovery reference
print(f"\n  Discovery reference:")
print(f"    d=-0.572, 95% CI=[-1.021,-0.248], Wilcoxon p=2.27e-13, N=43")

# ── Figura: paired dot-plot ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 7))
fig.patch.set_facecolor('white')

x_des, x_exc = 0, 1

for i, (d, e) in enumerate(zip(des_arr, exc_arr)):
    color = '#CD5C5C' if e > d else '#6495ED'
    ax.plot([x_des, x_exc], [d, e], '-', color=color, alpha=0.5, linewidth=1.2)
    ax.plot(x_des, d, 'o', color='#F4A460', markersize=6, zorder=3)
    ax.plot(x_exc, e, 'o', color='#CD5C5C',  markersize=6, zorder=3)

# Medias
ax.plot(x_des, np.mean(des_arr), 'D', color='black', markersize=10,
        zorder=5, label=f'Mean Desert={np.mean(des_arr):.3f}')
ax.plot(x_exc, np.mean(exc_arr), 'D', color='black', markersize=10,
        zorder=5, label=f'Mean Excluded={np.mean(exc_arr):.3f}')

ax.set_xticks([x_des, x_exc])
ax.set_xticklabels(['Immune\nDesert', 'Immune\nExcluded'], fontsize=13)
ax.set_ylabel('CAF Abundance (patient median)', fontsize=12)
ax.set_title('Patient-Level CAF: Desert vs Excluded\n(Validation Dataset — GSE213688)',
             fontsize=12, fontweight='bold')

pval_str = f'p={wpval:.4f}' if np.isfinite(wpval) else 'p=N/A'
ax.text(0.5, 0.97,
        f'd={d_patient:.3f}, 95%CI=[{ci_lo:.3f},{ci_hi:.3f}]\n'
        f'Wilcoxon {pval_str}, N={len(patients_both)} patients',
        ha='center', va='top', transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

ax.set_facecolor('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)
plt.tight_layout()

out_png = OUT_DIR / 'figures' / 'FigS_patient_CAF_validation.png'
out_pdf = OUT_DIR / 'figures' / 'FigS_patient_CAF_validation.pdf'
plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n  ✓ Figura: {out_png}")

# ── Guardar CSV y JSON ────────────────────────────────────────────────────────
out_csv = OUT_DIR / 'tables' / 'patient_level_CAF_validation.csv'
df.to_csv(out_csv, index=False)
print(f"  ✓ CSV:    {out_csv}")

summary = {
    'n_patients_total':  len(df),
    'n_patients_both':   len(patients_both),
    'n_excluded_gt_desert': int(n_exc_gt),
    'cohens_d_patient':  float(d_patient),
    'ci_95_lo':          float(ci_lo),
    'ci_95_hi':          float(ci_hi),
    'wilcoxon_p':        float(wpval) if np.isfinite(wpval) else None,
    'desert_mean':       float(np.mean(des_arr)),
    'excluded_mean':     float(np.mean(exc_arr)),
    'discovery_reference': {
        'd': -0.572, 'ci': [-1.021, -0.248], 'wilcoxon_p': 2.27e-13, 'n': 43
    },
}
out_json = OUT_DIR / 'tables' / 'patient_level_CAF_validation.json'
with open(out_json, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  ✓ JSON:   {out_json}")

print("\n FIX 5 COMPLETADO")
