#!/usr/bin/env python3
"""
fix_b2_gene_dropout_correct.py — Gene Dropout CORREGIDO 
================================================================================
v1 FALLÓ con MemoryError: .raw tiene 29,946 genes × 74,131 spots = 8 GB denso.
v2 trabaja con SPARSE matrices y computa scores gene-by-gene sin densificar.
================================================================================
"""

import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
import scipy.sparse as sp

warnings.filterwarnings('ignore')

POSSIBLE_BASES = [Path('/home/external/frjimenez/fabian/genoma'), Path.home()/'genoma', Path('.')]
BASE_DIR = next((p for p in POSSIBLE_BASES if p.exists()), None)
if BASE_DIR is None:
    print("ERROR: No se encontró directorio base"); sys.exit(1)

DATA_DIR = BASE_DIR / 'data' / 'processed'
RESULTS_DIR = BASE_DIR / 'results' / 'robustness_stress_tests'
FIG_DIR = RESULTS_DIR / 'figures'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Firmas COMPLETAS de config.py
SILENCING_REPRESSORS = ['MYC','EZH2','SUZ12','CTNNB1','ATF3','STAT3','DNMT1']
STING_PATHWAY = ['TMEM173','TBK1','IRF3','CGAS','MB21D1']
PHYSICAL_BARRIER = ['COL1A1','COL1A2','COL10A1','FN1','POSTN','TGFB1','ACTA2','FAP','THBS2']
CD8_T_CELLS = ['CD8A','CD8B','CD3D','CD3E','GZMA','GZMB','PRF1']
CHEMOKINE_SIGNALS = ['CCL5','CXCL9','CXCL10','CXCL11','HLA-A','B2M']
TUMOR_MARKERS = ['EPCAM','KRT18','KRT19','MKI67','TOP2A']
DESERT_STROMA = ['VIM','THY1','PDGFRA','PDGFRB','S100A4']
MECHANISM_GENES = ['MYC','EZH2','DNMT1','STAT3']

TUMOR_PCT, CD8_PCT, AMBIGUITY = 60, 75, 0.1
DROPOUT_FRACTIONS = [0.05, 0.10, 0.20, 0.30, 0.50]
N_REPEATS = 10


def cohens_d_pooled(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2: return np.nan
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    ps = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
    return (np.mean(g1) - np.mean(g2)) / ps if ps > 0 else 0.0


def sparse_gene_score(X_sparse, var_names, gene_list):
    """Calcula score como mean de genes disponibles. X_sparse: spots × genes (CSR/CSC)."""
    var_list = list(var_names)
    indices = []
    for g in gene_list:
        if g in var_list:
            indices.append(var_list.index(g))
    if not indices:
        return np.zeros(X_sparse.shape[0])
    # Extraer solo las columnas necesarias (eficiente en CSC)
    sub = X_sparse[:, indices]
    if sp.issparse(sub):
        return np.asarray(sub.mean(axis=1)).flatten()
    return np.mean(sub, axis=1)


def classify_from_scores(tumor_s, cd8_s, sil_s, bar_s, sample_ids):
    """Clasifica spots a partir de scores pre-computados."""
    n = len(tumor_s)

    # Z-score por muestra
    for arr in [tumor_s, cd8_s, sil_s, bar_s]:
        for s in np.unique(sample_ids):
            m = sample_ids == s
            v = arr[m]
            sd = np.std(v)
            arr[m] = (v - np.mean(v)) / sd if sd > 0 else 0.0

    t_thresh = np.percentile(tumor_s, TUMOR_PCT)
    c_thresh = np.percentile(cd8_s, CD8_PCT)

    pheno = np.full(n, 'Normal_Stroma', dtype='U20')
    hot = (tumor_s > t_thresh) & (cd8_s > c_thresh)
    cold = (tumor_s > t_thresh) & (cd8_s <= c_thresh)
    pheno[hot] = 'Inflamed'

    diff = sil_s - bar_s
    pheno[cold & (diff > AMBIGUITY)] = 'Immune_Desert'
    pheno[cold & (diff < -AMBIGUITY)] = 'Immune_Excluded'
    pheno[cold & (np.abs(diff) <= AMBIGUITY)] = 'Ambiguous_Cold'

    return pheno


def ari(l1, l2):
    """Adjusted Rand Index."""
    n = len(l1)
    if n < 2: return 0.0
    c1 = np.unique(l1); c2 = np.unique(l2)
    ct = np.zeros((len(c1), len(c2)), dtype=int)
    m1 = {c: i for i, c in enumerate(c1)}
    m2 = {c: i for i, c in enumerate(c2)}
    for i in range(n):
        ct[m1[l1[i]], m2[l2[i]]] += 1
    sc = sum(v*(v-1)//2 for v in ct.flatten())
    sa = sum(v*(v-1)//2 for v in ct.sum(1))
    sb = sum(v*(v-1)//2 for v in ct.sum(0))
    nc = n*(n-1)//2
    ex = sa*sb/nc if nc > 0 else 0
    mx = (sa+sb)/2
    return (sc-ex)/(mx-ex) if mx != ex else (1.0 if sc == ex else 0.0)


def main():
    print("="*72)
    print("FIX B2: GENE DROPOUT — CORREGIDO v2 (SPARSE)")
    print("="*72)
    print(f"  Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  FIX: Trabaja con matrices sparse para evitar MemoryError\n")

    import scanpy as sc

    adata_path = DATA_DIR / 'adata_with_mechanism.h5ad'
    if not adata_path.exists():
        adata_path = DATA_DIR / 'adata_with_phenotypes.h5ad'
    print(f"  Cargando: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    print(f"  Spots: {adata.n_obs:,} | Genes (.X): {adata.n_vars:,}")

    # Usar .raw (sparse, 29K genes)
    if adata.raw is not None:
        X_raw = adata.raw.X  # sparse matrix spots × genes
        var_names = list(adata.raw.var_names)
        if not sp.issparse(X_raw):
            X_raw = sp.csc_matrix(X_raw)
        else:
            X_raw = X_raw.tocsc()  # CSC para slicing eficiente por columna
        print(f"  .raw: {X_raw.shape[0]:,} spots × {X_raw.shape[1]:,} genes (sparse {X_raw.format})")
    else:
        X_raw = adata.X
        var_names = list(adata.var_names)
        if not sp.issparse(X_raw):
            X_raw = sp.csc_matrix(X_raw)
        else:
            X_raw = X_raw.tocsc()
        print(f"  .raw=None — usando .X: {X_raw.shape}")

    ref_pheno = adata.obs['Phenotype'].values
    sample_ids = adata.obs['sample_id'].values
    n_desert = (ref_pheno == 'Immune_Desert').sum()
    n_excluded = (ref_pheno == 'Immune_Excluded').sum()
    print(f"  Desert: {n_desert:,} | Excluded: {n_excluded:,}")

    # CAF abundance
    caf_abundance = None
    obsm_key = 'means_cell_abundance_w_sf'
    if obsm_key in adata.obsm:
        obsm_df = pd.DataFrame(adata.obsm[obsm_key], index=adata.obs_names)
        for col in obsm_df.columns:
            if 'CAF' in str(col):
                caf_abundance = obsm_df[col].values
                print(f"  ✓ CAF: {col}")
                break

    if caf_abundance is None:
        caf_abundance = sparse_gene_score(X_raw, var_names, ['COL1A1','ACTA2','FN1','FAP','POSTN'])
        print(f"  CAF proxy desde marker genes")

    baseline_d = cohens_d_pooled(
        caf_abundance[ref_pheno == 'Immune_Desert'],
        caf_abundance[ref_pheno == 'Immune_Excluded']
    )
    print(f"  Baseline CAF d: {baseline_d:.4f}")

    # Verificar genes
    print(f"\n  Genes en dataset:")
    gene_set = set(var_names)
    sigs = {
        'SILENCING (7)': SILENCING_REPRESSORS, 'STING (5)': STING_PATHWAY,
        'BARRIER (9)': PHYSICAL_BARRIER, 'CD8 (7)': CD8_T_CELLS,
        'CHEMOKINE (6)': CHEMOKINE_SIGNALS, 'MECHANISM (4)': MECHANISM_GENES
    }
    for name, genes in sigs.items():
        p = sum(1 for g in genes if g in gene_set)
        miss = [g for g in genes if g not in gene_set]
        s = "✓" if p == len(genes) else "⚠"
        print(f"    {s} {name}: {p}/{len(genes)}", end="")
        if miss: print(f"  (faltan: {', '.join(miss)})", end="")
        print()

    # ── GENE DROPOUT ──────────────────────────────────────────
    print(f"\n{'='*72}")
    print("GENE DROPOUT SIMULATION (SPARSE, CORREGIDO)")
    print(f"{'='*72}")
    print(f"  Genes totales: {len(var_names):,}")
    print(f"  Fracciones: {DROPOUT_FRACTIONS}")
    print(f"  Repeticiones: {N_REPEATS} (seed=42)")

    rng = np.random.RandomState(42)
    n_genes_total = len(var_names)
    all_results = []

    for frac in DROPOUT_FRACTIONS:
        n_drop = max(1, int(frac * n_genes_total))
        d_vals, ari_vals = [], []
        sig_examples = {}

        print(f"\n  Dropout = {frac:.0%} ({n_drop:,} genes eliminados):")

        for rep in range(N_REPEATS):
            # Crear máscara de genes que sobreviven
            drop_idx = rng.choice(n_genes_total, size=n_drop, replace=False)
            keep_mask = np.ones(n_genes_total, dtype=bool)
            keep_mask[drop_idx] = False
            kept_names = [var_names[i] for i in range(n_genes_total) if keep_mask[i]]
            kept_set = set(kept_names)

            # Extraer submatriz sparse (solo columnas que sobreviven)
            keep_indices = np.where(keep_mask)[0]
            X_sub = X_raw[:, keep_indices]  # sparse slicing por columna (CSC eficiente)

            # Recomputar scores con genes reducidos
            tumor_s = sparse_gene_score(X_sub, kept_names, TUMOR_MARKERS).copy()
            cd8_s = sparse_gene_score(X_sub, kept_names, CD8_T_CELLS).copy()
            sil_s = sparse_gene_score(X_sub, kept_names, SILENCING_REPRESSORS).copy()
            bar_s = sparse_gene_score(X_sub, kept_names, PHYSICAL_BARRIER).copy()

            # Reclasificar
            new_pheno = classify_from_scores(tumor_s, cd8_s, sil_s, bar_s, sample_ids)

            # Cohen's d CAF
            dm = new_pheno == 'Immune_Desert'
            em = new_pheno == 'Immune_Excluded'
            if dm.sum() > 1 and em.sum() > 1:
                d = cohens_d_pooled(caf_abundance[dm], caf_abundance[em])
            else:
                d = np.nan
            d_vals.append(d)

            # ARI
            a = ari(ref_pheno, new_pheno)
            ari_vals.append(a)

            # Monitor signatures (primera repetición)
            if rep == 0:
                sig_examples = {
                    'silencing': f"{sum(1 for g in SILENCING_REPRESSORS if g in kept_set)}/{len(SILENCING_REPRESSORS)}",
                    'barrier': f"{sum(1 for g in PHYSICAL_BARRIER if g in kept_set)}/{len(PHYSICAL_BARRIER)}",
                    'cd8': f"{sum(1 for g in CD8_T_CELLS if g in kept_set)}/{len(CD8_T_CELLS)}",
                    'sting': f"{sum(1 for g in STING_PATHWAY if g in kept_set)}/{len(STING_PATHWAY)}",
                    'mechanism': f"{sum(1 for g in MECHANISM_GENES if g in kept_set)}/{len(MECHANISM_GENES)}"
                }

        d_mean = np.nanmean(d_vals)
        d_std = np.nanstd(d_vals)
        ari_mean = np.mean(ari_vals)
        robust = abs(d_mean) >= 0.5

        print(f"    d medio: {d_mean:.4f} ± {d_std:.4f}")
        print(f"    ARI medio: {ari_mean:.3f}")
        print(f"    Silencing: {sig_examples.get('silencing','?')} | "
              f"Barrier: {sig_examples.get('barrier','?')} | "
              f"CD8: {sig_examples.get('cd8','?')}")
        print(f"    Mechanism: {sig_examples.get('mechanism','?')} | "
              f"STING: {sig_examples.get('sting','?')}")
        print(f"    {'✓' if robust else '⚠'} |d| {'≥' if robust else '<'} 0.5")

        all_results.append(dict(
            dropout_fraction=frac, d_mean=round(d_mean,4), d_std=round(d_std,4),
            ari_mean=round(ari_mean,3), ari_std=round(np.std(ari_vals),3),
            robust=robust
        ))

    # ── RESUMEN ───────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    n_robust = df['robust'].sum()

    print(f"\n{'='*72}")
    print("RESUMEN GENE DROPOUT CORREGIDO v2")
    print(f"{'='*72}")
    print(f"  Baseline d: {baseline_d:.4f}")
    print(f"  Robusto: {n_robust}/{len(DROPOUT_FRACTIONS)} niveles")
    d20 = df.loc[df['dropout_fraction']==0.2, 'd_mean']
    if len(d20) > 0:
        print(f"  d @ 20%: {d20.iloc[0]:.4f}")

    if n_robust == len(DROPOUT_FRACTIONS):
        print(f"  EFECTO ROBUSTO en TODOS los niveles")
    else:
        print(f"  Degradación en algún nivel")

    print(f"\n  vs Original (H1-02, genes incorrectos): d = −0.82 @ 20%")
    if len(d20) > 0:
        print(f"  vs Corregido (genes correctos):         d = {d20.iloc[0]:.4f} @ 20%")

    out = RESULTS_DIR / 'gene_dropout_results_CORRECTED.csv'
    df.to_csv(out, index=False)
    print(f"\n  CSV: {out}")

    # ── Figura ────────────────────────────────────────────────
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fracs = df['dropout_fraction'].values * 100

        ax1.errorbar(fracs, df['d_mean'], yerr=df['d_std'],
                     marker='o', capsize=3, color='#2C7BB6', lw=2)
        ax1.axhline(-0.5, color='red', ls='--', alpha=.5, label='|d|=0.5')
        ax1.axhline(baseline_d, color='gray', ls=':', alpha=.5, label=f'Baseline {baseline_d:.3f}')
        ax1.set_xlabel('Gene Dropout (%)')
        ax1.set_ylabel("Cohen's d (CAF)")
        ax1.set_title('A) Effect Size Stability', fontweight='bold')
        ax1.legend(fontsize=9)

        ax2.errorbar(fracs, df['ari_mean'], yerr=df['ari_std'],
                     marker='s', capsize=3, color='#D7191C', lw=2)
        ax2.set_xlabel('Gene Dropout (%)')
        ax2.set_ylabel('ARI vs Reference')
        ax2.set_title('B) Classification Stability', fontweight='bold')
        ax2.set_ylim(0, 1.05)

        fig.suptitle('Gene Dropout Robustness (Corrected — all signatures monitored)',
                     fontweight='bold', y=1.02)
        plt.tight_layout()
        fig.savefig(FIG_DIR/'Fig_gene_dropout_CORRECTED.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(FIG_DIR/'Fig_gene_dropout_CORRECTED.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Figura: {FIG_DIR/'Fig_gene_dropout_CORRECTED.pdf'}")
    except Exception as e:
        print(f"  Figura: {e}")

    print(f"\n FIX B2 v2 COMPLETADO")

if __name__ == '__main__':
    main()
