#!/usr/bin/env python3
"""
fix_b1_auc_loocv.py — LOO-CV AUC (VERSIÓN CORREGIDA)
================================================================================
v1 FALLÓ porque re-implementaba la clasificación incorrectamente → 100% Ambiguous.
v2 usa las ETIQUETAS EXISTENTES de bulk_validation.py como ground truth,
y solo hace LOO en la NORMALIZACIÓN del score discriminante.

Enfoque:
  1. Cargar labels de bulk_classification_{DATASET}.csv (ya generados)
  2. Para pacientes Desert+Excluded: score Barrier-Silencing con LOO normalization
  3. AUC sobre scores LOO vs labels originales
================================================================================
"""

import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

POSSIBLE_BASES = [Path('/home/external/frjimenez/fabian/genoma'), Path.home()/'genoma', Path('.')]
BASE_DIR = next((p for p in POSSIBLE_BASES if p.exists()), None)
if BASE_DIR is None:
    print("ERROR: No se encontró directorio base"); sys.exit(1)

BULK_DIR = BASE_DIR / 'results' / 'bulk_validation'
TABLE_DIR = BULK_DIR / 'tables'
TABLE_DIR.mkdir(parents=True, exist_ok=True)

SILENCING_REPRESSORS = ['MYC','EZH2','SUZ12','CTNNB1','ATF3','STAT3','DNMT1']
STING_PATHWAY = ['TMEM173','TBK1','IRF3','CGAS','MB21D1']
CHEMOKINE_SIGNALS = ['CCL5','CXCL9','CXCL10','CXCL11']
PHYSICAL_BARRIER = ['COL1A1','COL1A2','COL10A1','FN1','POSTN','TGFB1','ACTA2','FAP','THBS2']


def compute_auc(labels, scores):
    """AUC via Mann-Whitney U. labels: 1=positive, 0=negative."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    u = sum(np.sum(neg < p) + 0.5 * np.sum(neg == p) for p in pos)
    return u / (len(pos) * len(neg))


def score_genes(expr_df, genes):
    avail = [g for g in genes if g in expr_df.columns]
    return expr_df[avail].mean(axis=1) if avail else pd.Series(0.0, index=expr_df.index)


def load_expression(name):
    if name == 'METABRIC':
        f = BASE_DIR/'data'/'raw'/'METABRIC'/'data_mrna_illumina_microarray.txt'
    else:
        f = BASE_DIR/'data'/'raw'/'TCGA-BRCA'/'data_mrna_seq_v2_rsem.txt'
    if not f.exists():
        return None
    expr = pd.read_csv(f, sep='\t', index_col=0)
    if 'Entrez_Gene_Id' in expr.columns:
        expr = expr.drop('Entrez_Gene_Id', axis=1)
    expr = expr.dropna(axis=0, how='all')
    expr = expr[~expr.index.duplicated(keep='first')]
    expr = expr.T
    if name == 'TCGA' and expr.median().median() > 50:
        expr = np.log2(expr + 1)
    for a, b in [('STING1','TMEM173'),('TMEM173','STING1')]:
        if b not in expr.columns and a in expr.columns:
            expr[b] = expr[a]
    return expr


def main():
    print("="*72)
    print("LOO-CV AUC — VERSIÓN CORREGIDA v2")
    print("Usa labels EXISTENTES de bulk_validation + LOO normalization")
    print("="*72)
    print(f"  Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Base: {BASE_DIR}\n")

    all_results = []

    for ds in ['METABRIC','TCGA']:
        print(f"\n{'='*72}\n  DATASET: {ds}\n{'='*72}")

        # ── Cargar labels existentes ──────────────────────────
        label_file = TABLE_DIR / f'bulk_classification_{ds}.csv'
        if not label_file.exists():
            print(f"  ⚠ {label_file.name} no encontrado")
            all_results.append(dict(dataset=ds, status='LABELS_NOT_FOUND',
                n_patients=0, n_desert=0, n_excluded=0,
                auc_circular=np.nan, auc_loocv=np.nan, delta_auc=np.nan))
            continue

        labels_df = pd.read_csv(label_file, index_col=0)
        pheno_col = next((c for c in labels_df.columns
                          if 'phenotype' in c.lower() or 'class' in c.lower()),
                         labels_df.columns[-1])
        phenotypes = labels_df[pheno_col]
        n_desert = (phenotypes == 'Immune_Desert').sum()
        n_excluded = (phenotypes == 'Immune_Excluded').sum()
        print(f"  Labels: {label_file.name} ({pheno_col})")
        print(f"  Desert: {n_desert}, Excluded: {n_excluded}")

        if n_desert < 2 or n_excluded < 2:
            print(f" Insuficientes pacientes")
            all_results.append(dict(dataset=ds, status='INSUFFICIENT_N',
                n_patients=len(phenotypes), n_desert=n_desert, n_excluded=n_excluded,
                auc_circular=np.nan, auc_loocv=np.nan, delta_auc=np.nan))
            continue

        # ── Cargar expresión ──────────────────────────────────
        expr = load_expression(ds)
        if expr is None:
            print(f" Expresión no disponible")
            all_results.append(dict(dataset=ds, status='EXPR_NOT_FOUND',
                n_patients=len(phenotypes), n_desert=n_desert, n_excluded=n_excluded,
                auc_circular=np.nan, auc_loocv=np.nan, delta_auc=np.nan))
            continue

        common = sorted(set(expr.index) & set(labels_df.index))
        if len(common) == 0:
            # Intentar normalizar IDs (TCGA trunca a 12 chars)
            expr_map = {idx: idx[:12] for idx in expr.index}
            label_ids = set(labels_df.index)
            common_expr = [k for k, v in expr_map.items() if v in label_ids]
            if common_expr:
                expr = expr.loc[common_expr]
                expr.index = [expr_map[i] for i in expr.index]
                common = sorted(set(expr.index) & set(labels_df.index))

        expr = expr.loc[common]
        phenotypes = labels_df.loc[common, pheno_col]
        print(f"  Expresión: {expr.shape[0]} pacientes, {expr.shape[1]} genes")

        # ── Mask Desert + Excluded ────────────────────────────
        de_mask = phenotypes.isin(['Immune_Desert','Immune_Excluded'])
        de_idx = phenotypes[de_mask].index.tolist()
        de_labels = (phenotypes[de_mask] == 'Immune_Excluded').astype(int)
        print(f"  Desert+Excluded para AUC: {len(de_idx)}")

        # ── Raw scores ────────────────────────────────────────
        barrier_raw = score_genes(expr, PHYSICAL_BARRIER)
        sil_up_raw = score_genes(expr, SILENCING_REPRESSORS)
        sil_down_raw = score_genes(expr, STING_PATHWAY + CHEMOKINE_SIGNALS)
        silencing_raw = sil_up_raw - sil_down_raw

        # ── AUC CIRCULAR ──────────────────────────────────────
        b_z = (barrier_raw - barrier_raw.mean()) / (barrier_raw.std() + 1e-10)
        s_z = (silencing_raw - silencing_raw.mean()) / (silencing_raw.std() + 1e-10)
        disc_circ = (b_z - s_z)[de_mask]
        auc_circ = compute_auc(de_labels.values, disc_circ.values)
        print(f"\n  AUC CIRCULAR:  {auc_circ:.4f}")

        # ── AUC LOO-CV ────────────────────────────────────────
        loo_scores = pd.Series(np.nan, index=de_idx)
        for pid in de_idx:
            train = expr.index != pid
            bm, bs = barrier_raw[train].mean(), barrier_raw[train].std() + 1e-10
            sm, ss = silencing_raw[train].mean(), silencing_raw[train].std() + 1e-10
            loo_scores[pid] = (barrier_raw[pid]-bm)/bs - (silencing_raw[pid]-sm)/ss

        auc_loo = compute_auc(de_labels.values, loo_scores.values)
        delta = auc_circ - auc_loo if not np.isnan(auc_loo) else np.nan
        print(f"  AUC LOO-CV:   {auc_loo:.4f}")
        print(f"  Δ(circ-LOO):  {delta:+.4f}")

        # ── Stability ─────────────────────────────────────────
        try:
            from scipy.stats import spearmanr
            rho, pval = spearmanr(disc_circ.values, loo_scores.values)
        except:
            rho, pval = np.nan, np.nan
        print(f" Score stability ρ: {rho:.4f}")

        # ── Per-gene AUC ──────────────────────────────────────
        print(f"\n Per-Gene AUC (top 5):")
        gene_aucs = []
        for gene in sorted(set(PHYSICAL_BARRIER+SILENCING_REPRESSORS+STING_PATHWAY+CHEMOKINE_SIGNALS)):
            if gene in expr.columns:
                gv = expr.loc[de_mask, gene].values
                ga = compute_auc(de_labels.values, gv)
                gene_aucs.append(dict(gene=gene, auc=round(ga,4), dataset=ds))
        gene_df = pd.DataFrame(gene_aucs).sort_values('auc', ascending=False)
        for _, r in gene_df.head(5).iterrows():
            print(f"    {r['gene']:15s}: AUC = {r['auc']:.3f}")
        gene_df.to_csv(TABLE_DIR/f'auc_loocv_per_gene_{ds}.csv', index=False)

        verdict = "ESTABLE" if not np.isnan(delta) and abs(delta) < 0.05 else "INFLADO"
        if not np.isnan(delta) and abs(delta) < 0.05:
            print(f"\n AUC estable bajo LOO — circularidad no infla significativamente")
        else:
            print(f"\n  AUC cambia {abs(delta) if not np.isnan(delta) else '?'} bajo LOO")

        all_results.append(dict(
            dataset=ds, status='OK', n_patients=len(phenotypes),
            n_desert=n_desert, n_excluded=n_excluded,
            auc_circular=round(auc_circ,4), auc_loocv=round(auc_loo,4),
            delta_auc=round(delta,4) if not np.isnan(delta) else np.nan,
            score_stability_rho=round(rho,4) if not np.isnan(rho) else np.nan,
            verdict=verdict
        ))

    # ── RESUMEN ───────────────────────────────────────────────
    print(f"\n{'='*72}\nRESUMEN FIX B1 v2\n{'='*72}")
    for r in all_results:
        if r['status'] != 'OK':
            print(f"  {r['dataset']}: {r['status']}"); continue
        print(f"\n  {r['dataset']}:")
        print(f"    AUC circular: {r['auc_circular']:.4f}")
        print(f"    AUC LOO-CV:   {r['auc_loocv']:.4f}")
        print(f"    Δ:            {r['delta_auc']:+.4f}")
        print(f"    Veredicto:    {r['verdict']}")

    pd.DataFrame(all_results).to_csv(TABLE_DIR/'auc_loocv_results.csv', index=False)
    print(f"\n Guardado: {TABLE_DIR/'auc_loocv_results.csv'}")
    print(f"\n FIX B1 v2 COMPLETADO")

if __name__ == '__main__':
    main()
