"""
================================================================================
Immune Checkpoint Landscape per Spatial Niche

PROPÓSITO
---------
Caracteriza el paisaje de checkpoints inmunes inhibitorios y co-estimulatorios
en cada nicho tumoral espacial (Desert, Excluded, Inflamed).

HIPÓTESIS CIENTÍFICAS (documentadas a priori)
---------------------------------------------
H1: PD-L1 (CD274) NO discrimina Desert vs Inflamed (ya confirmado:
    d = −0.12, dirección inversa en myc_sting_investigation.py)
    → Este módulo extiende y confirma ese resultado.

H2: Checkpoints anti-fagocíticos (CD47) son más altos en Desert
    (consistente con MYC→CD47, Casey 2016; ya mostrado ρ = +0.20)

H3: T cell exhaustion markers (LAG3, TIGIT, TIM3, HAVCR2) son más altos en
    Inflamed (donde hay T cells), no en Desert (donde no hay T cells)
    → Desert evade sin necesitar suprimir T cells exhaustos porque no hay T cells

H4: Co-estimulación (CD28, ICOS) debería ser más baja en Desert y Excluded
    (sin señal de activación en nichos fríos)

ANTI-CIRCULARIDAD — DOCUMENTADO EXPLÍCITAMENTE
-----------------------------------------------
Los genes de checkpoint analizados en este módulo son:
  CD274, PDCD1, CTLA4, LAG3, HAVCR2, TIGIT, VSIR, CD47, CD244,
  CD28, ICOS, TNFRSF4, LGALS9, NECTIN2, PVR, TNFRSF9

NINGUNO de estos genes aparece en ninguna firma de clasificación:
  - SILENCING_REPRESSORS: [MYC, EZH2, SUZ12, CTNNB1, ATF3, STAT3, DNMT1]
  - PHYSICAL_BARRIER: [COL1A1, COL1A2, COL3A1, ACTA2, FN1, FAP, POSTN, VCAN, PDPN]
  - TUMOR_MARKERS: [EPCAM, KRT8, KRT18, KRT19, ...]
  - CD8_T_CELLS: [CD8A, CD8B, GZMA, GZMB, PRF1, NKG7, CD3D, CD3E]

CIRCULARIDAD: Los resultados no están contaminados por clasificación.

ANÁLISIS
--------
A) Expresión media de cada checkpoint por fenotipo (n=16 genes)
B) Mann-Whitney vs Inflamed (referencia) + FDR BH
   Comparaciones: Desert vs Inflamed, Excluded vs Inflamed, Desert vs Excluded
C) Cohen's d para todos los pares
D) Heatmap Z-scored de expresión por fenotipo (16 genes × 4 fenotipos)
E) Violin plots para los 6 genes clínicamente más relevantes
F) Checkpoint ratio: inhibitory/co-stimulatory index por fenotipo
G) Correlaciones checkpoint × MYC_TF_activity (si está disponible en adata.obs)
================================================================================
"""

import gc
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_BASE = _SCRIPT_DIR.parent

DATA_PROCESSED = _BASE / "data" / "processed"
RESULTS_DIR    = _BASE / "results" / "checkpoint_landscape"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Preferir adata con MYC_TF_activity; fallback a mechanism
ADATA_PRIMARY  = DATA_PROCESSED / "adata_with_myc_tf.h5ad"
ADATA_FALLBACK = DATA_PROCESSED / "adata_with_mechanism.h5ad"

SEED = 42
np.random.seed(SEED)

# ── Labels (verificados — iguales a todo el pipeline) ─────────────────────────
PHENOTYPE_COL  = "Phenotype"
DESERT_LABEL   = "Immune_Desert"
EXCLUDED_LABEL = "Immune_Excluded"
INFLAMED_LABEL = "Inflamed"
STROMA_LABEL   = "Normal_Stroma"

PHENOTYPES_ORDERED = [INFLAMED_LABEL, EXCLUDED_LABEL, DESERT_LABEL]
PHENOTYPE_SHORT    = {
    INFLAMED_LABEL:  "Inflamed",
    EXCLUDED_LABEL:  "Excluded",
    DESERT_LABEL:    "Desert",
    STROMA_LABEL:    "Stroma",
}

# ── Checkpoint genes — ANTI-CIRCULARIDAD: ninguno en firmas de clasificación ──
# Organizados por categoría funcional

# Inhibitorios principales (con terapéuticos aprobados o en ensayos clínicos)
CHECKPOINT_INHIBITORY_MAIN = [
    "CD274",   # PD-L1 — anti-PD-L1 aprobado (atezolizumab, durvalumab)
    "PDCD1",   # PD-1  — anti-PD-1 aprobado (pembrolizumab, nivolumab)
    "CTLA4",   # CTLA-4 — anti-CTLA4 (ipilimumab)
    "LAG3",    # LAG-3 — anti-LAG3 en ensayos (relatlimab aprobado melanoma)
    "HAVCR2",  # TIM-3 — anti-TIM3 en ensayos clínicos
    "TIGIT",   # TIGIT — anti-TIGIT en ensayos clínicos
]

# Inhibitorios secundarios (exploración)
CHECKPOINT_INHIBITORY_SEC = [
    "VSIR",   # VISTA — checkpoint en mieloides, ensayos clínicos
    "CD244",  # 2B4/SLAMF4 — receptor inhibitorio NK y T cells
    "CD47",   # 'don't eat me' — anti-fagocítico, MYC-dependiente (Casey 2016)
]

# Ligandos inhibitorios (expresados en tumor o estroma)
CHECKPOINT_LIGANDS_INH = [
    "LGALS9",  # Galectin-9, ligando TIM-3
    "NECTIN2", # CD112, ligando TIGIT/CD226
    "PVR",     # CD155, ligando TIGIT/CD96
    "CD86",    # ligando CTLA-4 y CD28
]

# Co-estimulatorios (señales activadoras)
CHECKPOINT_COSTIM = [
    "CD28",     # co-estimulatorio T — señal CD28:CD80/CD86
    "ICOS",     # ICOS — co-estimulatorio, T helper y Treg
    "TNFRSF9",  # 4-1BB (CD137) — co-estimulatorio citotóxico
    "TNFRSF4",  # OX40 (CD134) — co-estimulatorio memoria T
]

# Todos los genes del análisis
ALL_CHECKPOINT_GENES = (
    CHECKPOINT_INHIBITORY_MAIN +
    CHECKPOINT_INHIBITORY_SEC +
    CHECKPOINT_LIGANDS_INH +
    CHECKPOINT_COSTIM
)

# Genes prioritarios para figuras principales (clinicamente más relevantes)
CLINICAL_PRIORITY_GENES = [
    "CD274", "PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT",
    "CD47", "LGALS9", "CD28",
]

# Etiquetas bonitas para figuras
GENE_LABELS = {
    "CD274": "PD-L1\n(CD274)",
    "PDCD1": "PD-1\n(PDCD1)",
    "CTLA4": "CTLA-4",
    "LAG3":  "LAG-3",
    "HAVCR2": "TIM-3\n(HAVCR2)",
    "TIGIT": "TIGIT",
    "VSIR":  "VISTA\n(VSIR)",
    "CD244": "2B4\n(CD244)",
    "CD47":  "CD47",
    "LGALS9": "Galectin-9\n(LGALS9)",
    "NECTIN2": "NECTIN2",
    "PVR":   "CD155\n(PVR)",
    "CD86":  "CD86",
    "CD28":  "CD28",
    "ICOS":  "ICOS",
    "TNFRSF9": "4-1BB\n(CD137)",
    "TNFRSF4": "OX40\n(CD134)",
}

# Paleta de colores por fenotipo — consistente con pipeline
PHENO_COLORS = {
    INFLAMED_LABEL:  "#E74C3C",
    EXCLUDED_LABEL:  "#E67E22",
    DESERT_LABEL:    "#3498DB",
    STROMA_LABEL:    "#BDC3C7",
}


# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES ESTADÍSTICAS 
# ═══════════════════════════════════════════════════════════════════════════════

def cohens_d_pooled(g1: np.ndarray, g2: np.ndarray) -> float:
    """Cohen's d pooled con ddof=1 — canónico del pipeline."""
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var_pool = ((n1 - 1) * g1.var(ddof=1) + (n2 - 1) * g2.var(ddof=1)) / (n1 + n2 - 2)
    if var_pool == 0:
        return 0.0
    return (g1.mean() - g2.mean()) / np.sqrt(var_pool)


def fdr_correct(pvals: list, alpha: float = 0.05) -> np.ndarray:
    """BH FDR — igual al pipeline."""
    pvals_arr = np.array(pvals, dtype=float)
    valid = ~np.isnan(pvals_arr)
    qvals = np.full_like(pvals_arr, np.nan)
    if valid.sum() > 0:
        _, q, _, _ = multipletests(pvals_arr[valid], alpha=alpha, method="fdr_bh")
        qvals[valid] = q
    return qvals


def safe_toarray(X):
    if sp.issparse(X):
        return np.asarray(X.toarray(), dtype=float)
    return np.asarray(X, dtype=float)


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACCIÓN DE EXPRESIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def get_expression_matrix_raw(adata) -> tuple:
    """Retorna (X, var_names) desde .raw preferentemente."""
    if adata.raw is not None:
        X = adata.raw.X
        var_names = pd.Index(adata.raw.var_names)
        logging.info(f"  Expresión desde .raw ({len(var_names):,} genes)")
        return X, var_names
    else:
        logging.warning("  .raw is None — usando .X como fallback")
        return adata.X, adata.var_names


def get_gene_vector(gene: str, X, var_names) -> np.ndarray | None:
    """Extrae vector de expresión para un gen. Retorna None si no está presente."""
    if gene not in var_names:
        return None
    idx = list(var_names).index(gene)
    col = X[:, idx]
    return safe_toarray(col).ravel()


def extract_checkpoint_expression(adata, X, var_names) -> pd.DataFrame:
    """Extrae expresión de todos los genes checkpoint como DataFrame.

    Returns
    -------
    df: DataFrame (spots × checkpoint_genes), NaN para genes ausentes
    available_genes: lista de genes encontrados
    absent_genes: lista de genes ausentes
    """
    n_spots = adata.n_obs
    data = {}
    available = []
    absent = []

    for gene in ALL_CHECKPOINT_GENES:
        vec = get_gene_vector(gene, X, var_names)
        if vec is not None:
            data[gene] = vec
            available.append(gene)
        else:
            data[gene] = np.full(n_spots, np.nan)
            absent.append(gene)

    df = pd.DataFrame(data, index=adata.obs_names)
    logging.info(f"  Checkpoint genes disponibles: {len(available)}/{len(ALL_CHECKPOINT_GENES)}")
    if absent:
        logging.warning(f"  Ausentes: {absent}")
    logging.info(f"  Disponibles: {available}")

    return df, available, absent


# ═══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS A+B+C: ESTADÍSTICAS POR FENOTIPO
# ═══════════════════════════════════════════════════════════════════════════════

def compute_checkpoint_stats(adata, checkpoint_df: pd.DataFrame,
                              available_genes: list, logger) -> pd.DataFrame:
    """Calcula estadísticas completas para cada gen × comparación de fenotipos.

    Tests realizados:
    - Desert vs Inflamed
    - Excluded vs Inflamed
    - Desert vs Excluded

    Métricas: mean por fenotipo, Cohen's d, Mann-Whitney, FDR BH.
    """
    phenos = adata.obs[PHENOTYPE_COL].values

    comparisons = [
        (DESERT_LABEL,   INFLAMED_LABEL,  "Desert_vs_Inflamed"),
        (EXCLUDED_LABEL, INFLAMED_LABEL,  "Excluded_vs_Inflamed"),
        (DESERT_LABEL,   EXCLUDED_LABEL,  "Desert_vs_Excluded"),
    ]

    rows = []

    for gene in available_genes:
        vals = checkpoint_df[gene].values

        # Medias por fenotipo
        means = {}
        for p in PHENOTYPES_ORDERED + [STROMA_LABEL]:
            mask = phenos == p
            g_vals = vals[mask & ~np.isnan(vals)]
            means[p] = float(g_vals.mean()) if len(g_vals) > 0 else np.nan

        # Tests estadísticos
        for g1_name, g2_name, comp_id in comparisons:
            m1 = phenos == g1_name
            m2 = phenos == g2_name

            g1 = vals[m1 & ~np.isnan(vals)]
            g2 = vals[m2 & ~np.isnan(vals)]

            if len(g1) < 10 or len(g2) < 10:
                logger.warning(f"  {gene} {comp_id}: grupos insuficientes (n1={len(g1)}, n2={len(g2)}) → skip")
                continue

            stat_mw, pval = mannwhitneyu(g1, g2, alternative="two-sided")
            d = cohens_d_pooled(g1, g2)

            # Categoría del gen
            if gene in CHECKPOINT_INHIBITORY_MAIN:
                category = "inhibitory_main"
            elif gene in CHECKPOINT_INHIBITORY_SEC:
                category = "inhibitory_secondary"
            elif gene in CHECKPOINT_LIGANDS_INH:
                category = "inhibitory_ligand"
            elif gene in CHECKPOINT_COSTIM:
                category = "costimulatory"
            else:
                category = "other"

            rows.append({
                "gene": gene,
                "gene_label": GENE_LABELS.get(gene, gene),
                "category": category,
                "comparison": comp_id,
                "group1": g1_name,
                "group2": g2_name,
                "n_g1": int(len(g1)),
                "n_g2": int(len(g2)),
                "mean_g1": float(g1.mean()),
                "mean_g2": float(g2.mean()),
                "mean_Desert":   means.get(DESERT_LABEL, np.nan),
                "mean_Excluded": means.get(EXCLUDED_LABEL, np.nan),
                "mean_Inflamed": means.get(INFLAMED_LABEL, np.nan),
                "mean_Stroma":   means.get(STROMA_LABEL, np.nan),
                "cohens_d": float(d),
                "mw_statistic": float(stat_mw),
                "p_value": float(pval),
                "q_value": np.nan,  # se rellena después por comparación
                "fdr_significant": False,
            })

    df_stats = pd.DataFrame(rows)

    # FDR por comparación (no global — cada comparación es un conjunto independiente)
    for comp_id in [c[2] for c in comparisons]:
        mask_comp = df_stats["comparison"] == comp_id
        if mask_comp.sum() == 0:
            continue
        pvals_comp = df_stats.loc[mask_comp, "p_value"].values
        qvals = fdr_correct(pvals_comp)
        df_stats.loc[mask_comp, "q_value"] = qvals
        df_stats.loc[mask_comp, "fdr_significant"] = qvals < 0.05

    return df_stats


def compute_mean_expression_table(adata, checkpoint_df: pd.DataFrame,
                                  available_genes: list) -> pd.DataFrame:
    """Tabla de expresión media por fenotipo para cada gen checkpoint."""
    phenos = adata.obs[PHENOTYPE_COL].values
    rows = []
    for gene in available_genes:
        vals = checkpoint_df[gene].values
        row = {"gene": gene, "category": "inhibitory" if gene in
               CHECKPOINT_INHIBITORY_MAIN + CHECKPOINT_INHIBITORY_SEC + CHECKPOINT_LIGANDS_INH
               else "costimulatory"}
        for p in PHENOTYPES_ORDERED + [STROMA_LABEL]:
            mask = (phenos == p) & ~np.isnan(vals)
            row[f"mean_{PHENOTYPE_SHORT[p]}"] = float(vals[mask].mean()) if mask.sum() > 0 else np.nan
            row[f"n_{PHENOTYPE_SHORT[p]}"] = int(mask.sum())
        rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS F: CHECKPOINT INDEX (inhibitory/costimulatory ratio)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_checkpoint_index(adata, checkpoint_df: pd.DataFrame,
                              available_genes: list, results: list, logger) -> pd.Series:
    """F: Checkpoint suppression index = mean(inhibitory) / (mean(costim) + ε).

    Un índice > 1 indica dominancia de señales inhibitorias sobre co-estimulatorias.
    Se espera: Desert < Inflamed (Desert carece de T cells para señalizar checkpoints).
    """
    logger.info("=" * 70)
    logger.info("ANÁLISIS F: Checkpoint inhibitory/co-stimulatory index")

    inh_genes  = [g for g in CHECKPOINT_INHIBITORY_MAIN + CHECKPOINT_INHIBITORY_SEC
                  if g in available_genes and not np.all(np.isnan(checkpoint_df[g].values))]
    costim_genes = [g for g in CHECKPOINT_COSTIM
                    if g in available_genes and not np.all(np.isnan(checkpoint_df[g].values))]

    if not inh_genes or not costim_genes:
        logger.warning("  Genes insuficientes para checkpoint index → skip")
        return None

    logger.info(f"  Inhibitory genes: {inh_genes}")
    logger.info(f"  Co-stimulatory genes: {costim_genes}")

    # Calcular index por spot
    inh_mean   = checkpoint_df[inh_genes].mean(axis=1).values
    costim_mean = checkpoint_df[costim_genes].mean(axis=1).values
    epsilon = 1e-6
    index = inh_mean / (costim_mean + epsilon)
    adata.obs["Checkpoint_Index"] = index

    phenos = adata.obs[PHENOTYPE_COL].values
    groups = {p: index[phenos == p] for p in PHENOTYPES_ORDERED}

    pvals = []
    for g1_name, g2_name, comp_id in [
        (DESERT_LABEL, INFLAMED_LABEL, "CheckpointIndex_Desert_vs_Inflamed"),
        (EXCLUDED_LABEL, INFLAMED_LABEL, "CheckpointIndex_Excluded_vs_Inflamed"),
    ]:
        g1 = groups[g1_name]
        g2 = groups[g2_name]
        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]
        _, pval = mannwhitneyu(g1, g2, alternative="two-sided")
        d = cohens_d_pooled(g1, g2)
        pvals.append(pval)
        results.append({
            "analysis": "F",
            "test_id": comp_id,
            "gene": "Checkpoint_Index",
            "comparison": comp_id,
            "cohens_d": float(d),
            "p_value": float(pval),
            "n_g1": int(len(g1)),
            "n_g2": int(len(g2)),
        })
        logger.info(f"  {comp_id}: d={d:.4f}, p={pval:.3e}")

    qvals = fdr_correct(pvals)
    for i, r in enumerate([r for r in results if r.get("analysis") == "F"]):
        r["q_value"] = float(qvals[i])
        r["fdr_significant"] = bool(qvals[i] < 0.05)

    return index


# ═══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS G: CORRELACIONES CON MYC_TF_ACTIVITY
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_G_myctf_correlations(adata, checkpoint_df: pd.DataFrame,
                                   available_genes: list, results: list,
                                   logger) -> None:
    """G: Spearman checkpoint × MYC_TF_activity (si disponible).

    Esperado:
    - CD47 ρ > 0 (MYC activa CD47 — ya mostrado ρ = +0.20 con MYC mRNA)
    - MHC-I ligands (CD274) ρ < 0 o ~ 0 (MYC reprime)
    - LAG3, TIGIT, HAVCR2 ρ ~ 0 (son exhaustion markers de T cells,
      no regulados por MYC directamente)
    """
    if "MYC_TF_activity" not in adata.obs.columns:
        logger.info("  MYC_TF_activity no disponible → skip análisis G")
        return

    logger.info("=" * 70)
    logger.info("ANÁLISIS G: Checkpoint × MYC_TF_activity correlaciones")

    tf_vals = adata.obs["MYC_TF_activity"].values

    # Solo en spots tumorales (Desert + Excluded + Inflamed)
    tumor_mask = adata.obs[PHENOTYPE_COL].isin(PHENOTYPES_ORDERED)
    tf_tumor = tf_vals[tumor_mask]

    pvals = []
    for gene in available_genes:
        gene_vals = checkpoint_df[gene].values[tumor_mask]
        valid = ~(np.isnan(tf_tumor) | np.isnan(gene_vals))
        if valid.sum() < 30:
            continue
        rho, pval = spearmanr(tf_tumor[valid], gene_vals[valid])
        pvals.append(pval)
        results.append({
            "analysis": "G",
            "test_id": f"MYC_TF_vs_{gene}_tumor",
            "gene": gene,
            "comparison": "MYC_TF_vs_checkpoint_tumor",
            "statistic": float(rho),
            "p_value": float(pval),
            "n_g1": int(valid.sum()),
            "hypothesis": (
                f"ρ > 0 if MYC activates {gene}; ρ < 0 if MYC represses {gene}"
            ),
        })
        logger.info(f"  MYC_TF vs {gene} (tumor spots): ρ={rho:.4f}, p={pval:.3e}")

    if pvals:
        qvals = fdr_correct(pvals)
        g_results = [r for r in results if r.get("analysis") == "G"]
        for i, r in enumerate(g_results):
            r["q_value"] = float(qvals[i])
            r["fdr_significant"] = bool(qvals[i] < 0.05)


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIONES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_checkpoint_heatmap(df_means: pd.DataFrame, available_genes: list,
                             out_dir: Path) -> None:
    """D: Heatmap Z-scored de expresión checkpoint por fenotipo."""
    try:
        import seaborn as sns
    except ImportError:
        logging.warning("  seaborn no disponible para heatmap → skip")
        return

    # Construir matriz (genes × fenotipos)
    col_order = ["mean_Inflamed", "mean_Excluded", "mean_Desert"]
    col_labels = ["Inflamed", "Excluded", "Desert"]

    available_cols = [c for c in col_order if c in df_means.columns]
    if not available_cols:
        return

    mat = df_means.set_index("gene")[available_cols].copy()
    mat.columns = col_labels[:len(available_cols)]

    # Z-score por gen (normalizar para visualización)
    mat_z = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1).replace(0, 1), axis=0)

    # Ordenar genes por categoría
    gene_order = (
        [g for g in CHECKPOINT_INHIBITORY_MAIN if g in mat_z.index] +
        [g for g in CHECKPOINT_INHIBITORY_SEC if g in mat_z.index] +
        [g for g in CHECKPOINT_LIGANDS_INH if g in mat_z.index] +
        [g for g in CHECKPOINT_COSTIM if g in mat_z.index]
    )
    gene_order = [g for g in gene_order if g in mat_z.index]
    mat_z = mat_z.reindex(gene_order)

    # Etiquetas con nombre del gen
    yticklabels = [GENE_LABELS.get(g, g).replace("\n", " ") for g in mat_z.index]

    fig, ax = plt.subplots(figsize=(5, max(6, len(mat_z) * 0.45 + 2)))
    im = ax.imshow(mat_z.values, cmap="RdBu_r", aspect="auto",
                   vmin=-2, vmax=2, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Z-score (per gene)", fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(mat_z.columns)))
    ax.set_xticklabels(mat_z.columns, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(mat_z.index)))
    ax.set_yticklabels(yticklabels, fontsize=8)

    # Líneas divisorias entre categorías
    n_main    = len([g for g in CHECKPOINT_INHIBITORY_MAIN if g in mat_z.index])
    n_sec     = len([g for g in CHECKPOINT_INHIBITORY_SEC if g in mat_z.index])
    n_lig     = len([g for g in CHECKPOINT_LIGANDS_INH if g in mat_z.index])
    for sep in [n_main - 0.5, n_main + n_sec - 0.5, n_main + n_sec + n_lig - 0.5]:
        if 0 < sep < len(mat_z) - 1:
            ax.axhline(sep, color="white", linewidth=2, linestyle="--")

    # Anotar valores
    for i in range(len(mat_z.index)):
        for j in range(len(mat_z.columns)):
            val = mat_z.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(val) < 1.5 else "white")

    ax.set_title("Immune Checkpoint Landscape\nby Spatial Niche (z-scored per gene)",
                 fontsize=11, fontweight="bold", pad=12)

    # Anotaciones de categoría (lado izquierdo)
    cat_labels = [
        (n_main / 2 - 0.5, "Inhibitory\n(main)"),
        (n_main + n_sec / 2 - 0.5, "Inhibitory\n(secondary)"),
        (n_main + n_sec + n_lig / 2 - 0.5, "Inhibitory\nligands"),
        (n_main + n_sec + n_lig + len([g for g in CHECKPOINT_COSTIM if g in mat_z.index]) / 2 - 0.5, "Co-\nstimulatory"),
    ]
    for ypos, label in cat_labels:
        if 0 <= ypos < len(mat_z):
            ax.text(-0.7, ypos, label, ha="right", va="center",
                    fontsize=7, style="italic", color="#555555",
                    transform=ax.get_yaxis_transform())

    plt.tight_layout()
    fpath = out_dir / "fig_checkpoint_heatmap.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"  Guardada: {fpath}")


def plot_checkpoint_violins(adata, checkpoint_df: pd.DataFrame,
                             available_genes: list, df_stats: pd.DataFrame,
                             out_dir: Path) -> None:
    """E: Violin plots para genes prioritarios clínicamente."""
    try:
        import seaborn as sns
    except ImportError:
        return

    priority = [g for g in CLINICAL_PRIORITY_GENES if g in available_genes]
    if not priority:
        return

    n_genes = len(priority)
    fig, axes = plt.subplots(2, (n_genes + 1) // 2,
                              figsize=(4 * ((n_genes + 1) // 2), 9))
    axes = axes.flatten()
    fig.suptitle("Immune Checkpoint Expression by Spatial Niche",
                 fontsize=13, fontweight="bold")

    phenos = adata.obs[PHENOTYPE_COL].values
    palette = {PHENOTYPE_SHORT[p]: PHENO_COLORS[p] for p in PHENOTYPES_ORDERED}

    for i, gene in enumerate(priority):
        ax = axes[i]
        vals = checkpoint_df[gene].values
        df_plot = pd.DataFrame({
            "expr": vals,
            "phenotype": [PHENOTYPE_SHORT.get(p, p) for p in phenos]
        })
        df_plot = df_plot[df_plot["phenotype"].isin(["Inflamed", "Excluded", "Desert"])]
        df_plot = df_plot[~np.isnan(df_plot["expr"])]

        sns.violinplot(
            data=df_plot, x="phenotype", y="expr",
            order=["Inflamed", "Excluded", "Desert"],
            palette=palette, inner="box", cut=0, linewidth=0.8, ax=ax
        )
        ax.set_title(GENE_LABELS.get(gene, gene).replace("\n", " "),
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("log1p expression", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)

        # Anotar q-values FDR para Desert vs Inflamed
        q_di = df_stats.loc[
            (df_stats["gene"] == gene) &
            (df_stats["comparison"] == "Desert_vs_Inflamed"),
            "q_value"
        ]
        if len(q_di) > 0 and not np.isnan(q_di.iloc[0]):
            q = q_di.iloc[0]
            sig = "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else "ns"
            ax.text(0.5, 1.02, f"Desert vs Inflamed: {sig} (q={q:.3f})",
                    ha="center", transform=ax.transAxes, fontsize=7, color="#333333")

    # Ocultar ejes sobrantes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    fpath = out_dir / "fig_checkpoint_violins.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"  Guardada: {fpath}")


def plot_checkpoint_summary_bar(df_stats: pd.DataFrame, out_dir: Path) -> None:
    """Bar plot — Cohen's d para Desert vs Inflamed por gen checkpoint."""
    comp = "Desert_vs_Inflamed"
    df_comp = df_stats[df_stats["comparison"] == comp].copy()
    if df_comp.empty:
        return

    # Ordenar por d
    df_comp = df_comp.sort_values("cohens_d", ascending=True)

    # Color por significancia FDR
    bar_colors = ["#E74C3C" if sig else "#BDC3C7"
                  for sig in df_comp["fdr_significant"].values]

    fig, ax = plt.subplots(figsize=(7, max(5, len(df_comp) * 0.4 + 2)))
    bars = ax.barh(range(len(df_comp)), df_comp["cohens_d"].values,
                   color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(df_comp)))
    ax.set_yticklabels(
        [GENE_LABELS.get(g, g).replace("\n", " ") for g in df_comp["gene"]],
        fontsize=9
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(-0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Cohen's d (Desert vs Inflamed)", fontsize=10)
    ax.set_title("Checkpoint Expression: Desert vs Inflamed\nCohen's d (red = FDR < 0.05)",
                 fontsize=11, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(fc="#E74C3C", label="FDR < 0.05"),
        Patch(fc="#BDC3C7", label="FDR ≥ 0.05 (ns)")
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=9)

    plt.tight_layout()
    fpath = out_dir / "fig_checkpoint_summary_bar.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"  Guardada: {fpath}")


def plot_checkpoint_vs_myctf(adata, checkpoint_df: pd.DataFrame,
                               available_genes: list, results: list,
                               out_dir: Path) -> None:
    """Scatter de checkpoints clínicamente relevantes vs MYC_TF_activity."""
    if "MYC_TF_activity" not in adata.obs.columns:
        return

    genes_to_plot = [g for g in ["CD274", "CD47", "LAG3", "HAVCR2", "LGALS9"]
                     if g in available_genes]
    if not genes_to_plot:
        return

    tf_vals = adata.obs["MYC_TF_activity"].values
    phenos = adata.obs[PHENOTYPE_COL].values

    # Solo tumor spots, subsample
    tumor_mask = np.isin(phenos, PHENOTYPES_ORDERED)
    rng = np.random.default_rng(SEED)
    n_sample = min(5000, tumor_mask.sum())
    idx = rng.choice(np.where(tumor_mask)[0], n_sample, replace=False)

    colors_per_spot = np.array([PHENO_COLORS.get(phenos[i], "#AAAAAA") for i in idx])

    fig, axes = plt.subplots(1, len(genes_to_plot),
                              figsize=(4.5 * len(genes_to_plot), 5))
    if len(genes_to_plot) == 1:
        axes = [axes]

    fig.suptitle("Checkpoint Expression vs MYC TF Activity\n(tumor spots, n=5000 subsample)",
                 fontsize=12, fontweight="bold")

    for ax, gene in zip(axes, genes_to_plot):
        y = checkpoint_df[gene].values[idx]
        x = tf_vals[idx]
        valid = ~(np.isnan(x) | np.isnan(y))
        ax.scatter(x[valid], y[valid], c=colors_per_spot[valid],
                   alpha=0.3, s=3, rasterized=True)

        rho, pval = spearmanr(x[valid], y[valid])
        ax.set_xlabel("MYC TF Activity\n(ULM z-score)", fontsize=9)
        ax.set_ylabel(f"{gene} expression (log1p)", fontsize=9)
        ax.set_title(
            f"{GENE_LABELS.get(gene, gene).replace(chr(10), ' ')}\nρ={rho:.3f}, p={pval:.2e}",
            fontsize=9, fontweight="bold"
        )

    # Legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(fc=PHENO_COLORS[p], label=PHENOTYPE_SHORT[p])
        for p in PHENOTYPES_ORDERED
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fpath = out_dir / "fig_checkpoint_vs_myctf.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"  Guardada: {fpath}")


# ═══════════════════════════════════════════════════════════════════════════════
# SETUP LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(out_dir: Path) -> logging.Logger:
    log_file = out_dir / "checkpoint_landscape.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger("checkpoint")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    logger = setup_logging(RESULTS_DIR)

    logger.info("=" * 70)
    logger.info("IMMUNE CHECKPOINT LANDSCAPE PER SPATIAL NICHE")
    logger.info(f"Output: {RESULTS_DIR}")
    logger.info("=" * 70)
    logger.info(
        "ANTI-CIRCULARIDAD: Los genes de checkpoint analizados (CD274, PDCD1, "
        "CTLA4, LAG3, HAVCR2, TIGIT, etc.) NO aparecen en ninguna firma de "
        "clasificación. Circularidad: CERO."
    )

    # ── Cargar datos ──────────────────────────────────────────────────────────
    if ADATA_PRIMARY.exists():
        logger.info(f"  Cargando {ADATA_PRIMARY.name} (con MYC_TF_activity)...")
        adata = sc.read_h5ad(ADATA_PRIMARY)
        has_tf = "MYC_TF_activity" in adata.obs.columns
        if has_tf:
            logger.info("  MYC_TF_activity disponible → análisis G activado")
        else:
            logger.warning("  MYC_TF_activity no en adata.obs → análisis G skip")
    elif ADATA_FALLBACK.exists():
        logger.warning(f"  {ADATA_PRIMARY.name} no encontrado → usando {ADATA_FALLBACK.name}")
        adata = sc.read_h5ad(ADATA_FALLBACK)
        has_tf = False
    else:
        logger.error(f"  FATAL: No se encontró ningún adata en {DATA_PROCESSED}")
        sys.exit(1)

    logger.info(f"  Spots: {adata.n_obs:,} | Genes .X: {adata.n_vars:,}")

    # Verificar columna de fenotipos
    if PHENOTYPE_COL not in adata.obs.columns:
        for alt in ["phenotype", "Phenotype_v2"]:
            if alt in adata.obs.columns:
                adata.obs[PHENOTYPE_COL] = adata.obs[alt]
                logger.warning(f"  Usando '{alt}' como '{PHENOTYPE_COL}'")
                break
        else:
            logger.error(f"  FATAL: columna '{PHENOTYPE_COL}' no encontrada")
            sys.exit(1)

    dist = adata.obs[PHENOTYPE_COL].value_counts()
    logger.info(f"  Fenotipos:\n{dist.to_string()}")

    for required in [DESERT_LABEL, INFLAMED_LABEL, EXCLUDED_LABEL]:
        if required not in dist.index or dist[required] == 0:
            logger.error(f"  FATAL: 0 spots '{required}'")
            sys.exit(1)

    # ── Extraer expresión ─────────────────────────────────────────────────────
    logger.info("  Extrayendo expresión desde .raw...")
    X_raw, var_names = get_expression_matrix_raw(adata)

    # ── Extraer todos los genes checkpoint ───────────────────────────────────
    logger.info("  Extrayendo expresión de genes checkpoint...")
    checkpoint_df, available_genes, absent_genes = extract_checkpoint_expression(
        adata, X_raw, var_names
    )

    if len(available_genes) < 3:
        logger.error("  FATAL: <3 genes checkpoint disponibles")
        sys.exit(1)

    # ── Estadísticas principales ─────────────────────────────────────────────
    logger.info("\n  Calculando estadísticas por fenotipo y comparación...")
    df_stats = compute_checkpoint_stats(adata, checkpoint_df, available_genes, logger)

    df_means = compute_mean_expression_table(adata, checkpoint_df, available_genes)

    # ── Análisis F: checkpoint index ─────────────────────────────────────────
    extra_results: list = []
    checkpoint_index = compute_checkpoint_index(
        adata, checkpoint_df, available_genes, extra_results, logger
    )

    # ── Análisis G: correlaciones con MYC_TF ─────────────────────────────────
    if has_tf:
        analysis_G_myctf_correlations(
            adata, checkpoint_df, available_genes, extra_results, logger
        )

    # ── Log de resultados clave ───────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("RESULTADOS CLAVE — Desert vs Inflamed")
    logger.info("=" * 70)
    comp_di = df_stats[df_stats["comparison"] == "Desert_vs_Inflamed"].copy()
    comp_di = comp_di.sort_values("cohens_d", key=abs, ascending=False)
    for _, row in comp_di.iterrows():
        sig_str = "FDR-sig" if row["fdr_significant"] else "ns"
        logger.info(
            f"  {row['gene']:12s}: d={row['cohens_d']:+.4f}, "
            f"p={row['p_value']:.3e}, q={row['q_value']:.3e}  [{sig_str}]"
        )

    logger.info("\n" + "=" * 70)
    logger.info("RESULTADOS CLAVE — Excluded vs Inflamed")
    logger.info("=" * 70)
    comp_ei = df_stats[df_stats["comparison"] == "Excluded_vs_Inflamed"].copy()
    comp_ei = comp_ei.sort_values("cohens_d", key=abs, ascending=False)
    for _, row in comp_ei.head(8).iterrows():
        sig_str = "FDR-sig" if row["fdr_significant"] else "ns"
        logger.info(
            f"  {row['gene']:12s}: d={row['cohens_d']:+.4f}, "
            f"p={row['p_value']:.3e}, q={row['q_value']:.3e}  [{sig_str}]"
        )

    # ── Guardar resultados ────────────────────────────────────────────────────
    df_stats.to_csv(RESULTS_DIR / "checkpoint_stats_all_tests.csv", index=False)
    df_means.to_csv(RESULTS_DIR / "checkpoint_expression_by_phenotype.csv", index=False)

    if extra_results:
        pd.DataFrame(extra_results).to_csv(
            RESULTS_DIR / "checkpoint_index_and_myctf.csv", index=False
        )

    logger.info(f"\n  Archivos guardados en {RESULTS_DIR}")

    # ── Figuras ───────────────────────────────────────────────────────────────
    logger.info("\n  Generando figuras...")
    try:
        plot_checkpoint_heatmap(df_means, available_genes, RESULTS_DIR)
        plot_checkpoint_violins(adata, checkpoint_df, available_genes,
                                df_stats, RESULTS_DIR)
        plot_checkpoint_summary_bar(df_stats, RESULTS_DIR)
        if has_tf:
            plot_checkpoint_vs_myctf(adata, checkpoint_df, available_genes,
                                     extra_results, RESULTS_DIR)
    except Exception as e:
        logger.warning(f"  Error en figuras: {e}")
        import traceback
        traceback.print_exc()

    # ── Resumen interpretativo para paper ────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("INTERPRETACIÓN PARA PAPER (checkpoint landscape)")
    logger.info("=" * 70)

    # Identificar genes FDR-sig en Desert vs Inflamed
    sig_desert = comp_di[comp_di["fdr_significant"]]
    sig_up   = sig_desert[sig_desert["cohens_d"] > 0.1]
    sig_down = sig_desert[sig_desert["cohens_d"] < -0.1]

    if len(sig_down) > 0:
        logger.info(
            f"  Checkpoints BAJOS en Desert vs Inflamed (FDR-sig): "
            f"{list(sig_down['gene'].values)}"
        )
        logger.info(
            "  → Desert evade sin upregular checkpoints: mecanismo por ausencia\n"
            "    de reconocimiento (MHC-I down), no por checkpoint inhibition."
        )
    if len(sig_up) > 0:
        logger.info(
            f"  Checkpoints ALTOS en Desert vs Inflamed (FDR-sig): "
            f"{list(sig_up['gene'].values)}"
        )

    # Verificar H1 (PD-L1 no discrimina)
    pdl1 = comp_di[comp_di["gene"] == "CD274"]
    if len(pdl1) > 0:
        pdl1_d = pdl1["cohens_d"].iloc[0]
        pdl1_sig = pdl1["fdr_significant"].iloc[0]
        logger.info(
            f"\n  PD-L1 (CD274) Desert vs Inflamed: d={pdl1_d:.4f}, "
            f"FDR-sig={pdl1_sig}"
        )
        if pdl1_d < 0 and not pdl1_sig:
            logger.info(
                "  → CONFIRMA H1: PD-L1 no discrimina nicho Desert.\n"
                "    Evasión Desert NO es por checkpoint upregulation.\n"
                "    Consistente con: Desert evade por antigen presentation loss."
            )

    logger.info(f"\n  Tiempo total: {time.time()-t0:.1f}s")
    logger.info("  STATUS: COMPLETADO")


if __name__ == "__main__":
    main()
