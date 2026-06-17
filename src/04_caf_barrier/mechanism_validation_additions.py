"""
================================================================================
ADICIONES A mechanism_validation.py — v2.4 → v3.0
================================================================================
Contexto del problema:
  Las correlaciones MYC-STING actuales son biológicamente débiles
  (ρ < 0.06) porque se calculan sobre TODOS los spots. Esto es esperado:
  la represión MYC→STING es epigenética (proteína/enhancer), no mRNA-mRNA.
  
BLOQUE 1: Desert-Only Correlations
  Filtra SOLO spots Immune_Desert y recalcula correlaciones.
  Si ρ es más fuerte ahí → confirma nicho-especificidad.
  Si ρ sigue débil → consistente con represión epigenética.

BLOQUE 2: MYC Hallmark Gene Scores
  En vez de medir mRNA de MYC (un solo gen), calcula scores de
  MYC_TARGETS_V1 y V2 de MSigDB (200 genes downstream de MYC).
  Esto captura la ACTIVIDAD de MYC, no solo su expresión.
  Hipótesis: MYC_hallmark_score alto en Desert, bajo en Inflamed.

BLOQUE 3: Sub-análisis Macrófagos SPP1/CXCL9
  Valida el ratio CXCL9:SPP1 de Bill et al. (2023, Science) en TNBC.
  En spots con alta abundancia de macrófagos:
  - Desert: SPP1 alto, CXCL9 bajo (TAM pro-tumoral)
  - Excluded: SPP1 moderado, CXCL9 moderado (TAM mixto)
  - Inflamed: SPP1 bajo, CXCL9 alto (TAM anti-tumoral)
================================================================================
"""

import numpy as np
import pandas as pd
import anndata as ad
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr, mannwhitneyu, kruskal
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Importar Cohen's d canónico de utils_stats
try:
    from utils_stats import cohens_d_pooled
except ImportError:
    def cohens_d_pooled(g1, g2):
        g1, g2 = np.asarray(g1, float), np.asarray(g2, float)
        g1, g2 = g1[np.isfinite(g1)], g2[np.isfinite(g2)]
        n1, n2 = len(g1), len(g2)
        if n1 < 2 or n2 < 2: return 0.0
        sp = np.sqrt(((n1-1)*np.var(g1,ddof=1)+(n2-1)*np.var(g2,ddof=1))/(n1+n2-2))
        return float((g1.mean()-g2.mean())/sp) if sp > 1e-10 else 0.0


# ============================================================================
# BLOQUE 1: DESERT-ONLY CORRELATIONS
# ============================================================================

def validate_desert_only_correlations(
    adata: ad.AnnData,
    phenotype_col: str = 'Phenotype',
    desert_label: str = 'Immune_Desert',
    min_spots: int = 50,
) -> pd.DataFrame:
    """
    Recalcula correlaciones MYC/EZH2/DNMT1/STAT3 vs STING SOLO en spots
    clasificados como Immune_Desert.
    
    JUSTIFICACIÓN CIENTÍFICA:
    La hipótesis es que MYC reprime STING específicamente en el nicho
    Desert. Al calcular correlaciones sobre todos los spots (como hace
    v2.4), se diluye la señal con spots Inflamed/Excluded donde la
    relación MYC-STING es irrelevante.
    
    INTERPRETACIÓN DE RESULTADOS:
    - Si ρ más fuerte en Desert vs global → nicho-especificidad confirmada
    - Si ρ sigue débil → consistente con represión epigenética (esperado)
    - Si ρ cambia de signo → hallazgo novel, requiere explicación
    
    Parameters
    ----------
    adata : AnnData con expresión génica y columna Phenotype
    phenotype_col : str — columna de fenotipos
    desert_label : str — etiqueta del fenotipo Desert
    min_spots : int — mínimo de spots para calcular correlación
    
    Returns
    -------
    DataFrame con correlaciones por subgrupo y comparación
    """
    # Importar helper del módulo existente
    try:
        from mechanism_validation import get_gene_expression
    except ImportError:
        # Si se ejecuta como parte del módulo, la función ya está disponible
        pass
    
    print("\n" + "=" * 80)
    print("BLOQUE 1: CORRELACIONES DESERT-ONLY")
    print("=" * 80)
    
    repressors = ['MYC', 'EZH2', 'DNMT1', 'STAT3']
    sting_aliases = ['TMEM173', 'STING1', 'MITA']
    
    # Obtener expresión de STING
    sting_expr = _safe_get_gene_expression(adata, 'TMEM173', aliases=['STING1', 'MITA'])
    
    if sting_expr is None or np.all(sting_expr == 0):
        print("  No se encontró expresión de STING (TMEM173/STING1)")
        return pd.DataFrame()
    
    # Verificar columna de fenotipos
    if phenotype_col not in adata.obs.columns:
        print(f"  Columna '{phenotype_col}' no encontrada")
        return pd.DataFrame()
    
    # Definir subgrupos a analizar
    subgroups = {
        'ALL_SPOTS': np.ones(adata.n_obs, dtype=bool),
        'Desert_ONLY': adata.obs[phenotype_col].values == desert_label,
        'Excluded_ONLY': adata.obs[phenotype_col].values == 'Immune_Excluded',
        'Inflamed_ONLY': adata.obs[phenotype_col].values == 'Inflamed',
        'Cold_ONLY': np.isin(adata.obs[phenotype_col].values, 
                             [desert_label, 'Immune_Excluded']),
    }
    
    results = []
    
    for repressor in repressors:
        rep_expr = _safe_get_gene_expression(adata, repressor)
        
        if rep_expr is None:
            print(f"  {repressor}: no encontrado")
            continue
        
        for subgroup_name, mask in subgroups.items():
            n_spots = mask.sum()
            
            if n_spots < min_spots:
                continue
            
            rep_sub = rep_expr[mask]
            sting_sub = sting_expr[mask]
            
            # Verificar varianza suficiente
            if np.std(rep_sub) < 1e-10 or np.std(sting_sub) < 1e-10:
                continue
            
            corr, pval = spearmanr(rep_sub, sting_sub)
            
            results.append({
                'Repressor': repressor,
                'Target': 'STING',
                'Subgroup': subgroup_name,
                'N_spots': int(n_spots),
                'Spearman_rho': float(corr),
                'P_value': float(pval),
                'Significant': pval < 0.05,
                'Direction': 'Negative' if corr < 0 else 'Positive',
                'Abs_rho': abs(corr),
            })
    
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print("No se pudieron calcular correlaciones")
        return df
    
    # Resumen comparativo
    print("\n  === COMPARACIÓN DESERT vs GLOBAL ===")
    print(f"  {'Repressor':<10} {'Global ρ':>10} {'Desert ρ':>10} {'Δρ':>8} {'Interpretación'}")
    print(f"  {'-'*60}")
    
    for rep in repressors:
        global_row = df[(df['Repressor'] == rep) & (df['Subgroup'] == 'ALL_SPOTS')]
        desert_row = df[(df['Repressor'] == rep) & (df['Subgroup'] == 'Desert_ONLY')]
        
        if len(global_row) > 0 and len(desert_row) > 0:
            rho_global = global_row['Spearman_rho'].values[0]
            rho_desert = desert_row['Spearman_rho'].values[0]
            delta = rho_desert - rho_global
            
            if abs(rho_desert) > abs(rho_global) * 1.5:
                interp = "STRONGER in Desert "
            elif abs(rho_desert) < abs(rho_global) * 0.5:
                interp = "WEAKER in Desert "
            else:
                interp = "Similar"
            
            print(f"  {rep:<10} {rho_global:>10.4f} {rho_desert:>10.4f} {delta:>+8.4f} {interp}")
    
    # Texto para Methods
    n_desert = subgroups['Desert_ONLY'].sum()
    print(f"\n  Methods text: 'Correlations were recalculated within "
          f"Immune Desert spots only (n={n_desert:,}) to test "
          f"niche-specificity of the MYC-STING axis.'")
    
    return df


# ============================================================================
# BLOQUE 2: MYC HALLMARK GENE SCORES
# ============================================================================

# Gene sets from MSigDB Hallmark collection
# Source: https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_MYC_TARGETS_V1.html

MYC_TARGETS_V1 = [
    'AIMP2', 'AP3S1', 'ATPIF1', 'BUB3', 'C1QBP', 'CAD', 'CANX', 'CBX3',
    'CCT2', 'CCT3', 'CCT4', 'CCT5', 'CCT7', 'CDK4', 'CLNS1A', 'CNBP',
    'COPS5', 'COX5A', 'CUL1', 'CYC1', 'DDX18', 'DDX21', 'DEK', 'DHX15',
    'DUT', 'EEF1B2', 'EIF1AX', 'EIF2S1', 'EIF2S2', 'EIF3B', 'EIF3D',
    'EIF3J', 'EIF4A1', 'EIF4E', 'EIF4G2', 'EIF4H', 'ELAVL1', 'ENO1',
    'EXOSC7', 'FAM120A', 'FBL', 'G3BP1', 'GNL3', 'GOT2', 'GPI', 'GSPT1',
    'HDAC2', 'HDGF', 'HNRNPA1', 'HNRNPA2B1', 'HNRNPC', 'HNRNPD',
    'HNRNPR', 'HNRNPU', 'HPRT1', 'HSP90AB1', 'HSPD1', 'HSPE1',
    'IARS1', 'ILF2', 'IMPDH2', 'IPO4', 'KARS1', 'KPNA2', 'KPNB1',
    'LDHA', 'LSM2', 'LSM7', 'MCM2', 'MCM4', 'MCM5', 'MCM6', 'MCM7',
    'MRPL9', 'MRTO4', 'MYBBP1A', 'MYC', 'NAP1L1', 'NCBP1', 'NCBP2',
    'NDUFAB1', 'NHP2', 'NME1', 'NOLC1', 'NOP16', 'NOP56', 'NPM1',
    'ODC1', 'ORC2', 'PA2G4', 'PABPC1', 'PCNA', 'PGK1', 'PHB', 'PHB2',
    'POLD2', 'POLE3', 'PPAT', 'PRDX3', 'PRDX4', 'PRPS2', 'PSMA1',
    'PSMA2', 'PSMA4', 'PSMA7', 'PSMB2', 'PSMB3', 'PSMC4', 'PSMC6',
    'PSMD1', 'PSMD14', 'PSMD3', 'PTGES3', 'PWP1', 'RACK1', 'RAD23B',
    'RAN', 'RANBP1', 'RFC4', 'RNPS1', 'RPL14', 'RPL18', 'RPL22',
    'RPL34', 'RPL6', 'RPLP0', 'RPS10', 'RPS2', 'RPS3', 'RPS5', 'RPS6',
    'RQCD1', 'RRP9', 'RSL1D1', 'RUVBL2', 'SERBP1', 'SET', 'SF3A1',
    'SF3B3', 'SLC25A3', 'SNRPA', 'SNRPA1', 'SNRPB2', 'SNRPD1',
    'SNRPD2', 'SNRPD3', 'SNRPG', 'SRM', 'SRSF1', 'SRSF2', 'SRSF3',
    'SSB', 'SSBP1', 'TCP1', 'TFDP1', 'TOMM70', 'TRA2B', 'TRIM28',
    'TUFM', 'TXNL4A', 'U2AF1', 'UBA2', 'UBE2E1', 'UBE2L3', 'USP1',
    'VBP1', 'VDAC1', 'XPOT', 'XPO5', 'XRCC6', 'YWHAE', 'YWHAQ',
]

MYC_TARGETS_V2 = [
    'AIMP2', 'BUB3', 'BYSL', 'CBX3', 'CDK4', 'CLNS1A', 'CNBP',
    'COX5A', 'CYC1', 'DDX18', 'DDX21', 'DUSP2', 'EXOSC5', 'FBL',
    'GNL3', 'GRWD1', 'HK2', 'HNRNPA1', 'HNRNPA2B1', 'HNRNPC',
    'HNRNPU', 'HSPD1', 'ILF2', 'IRAK1', 'KPNB1', 'LAS1L', 'LDHA',
    'MAP3K6', 'MCM4', 'MCM5', 'MCM6', 'MPHOSPH10', 'MYC', 'NAP1L1',
    'NCBP2', 'NDUFAB1', 'NIP7', 'NME1', 'NOC4L', 'NOLC1', 'NOP16',
    'NOP56', 'NPM1', 'ODC1', 'PA2G4', 'PCNA', 'PES1', 'PHB', 'PLK1',
    'PLK4', 'PPAN', 'PPAT', 'PPRC1', 'PRMT3', 'PUS1', 'RABEPK',
    'RACK1', 'RANBP1', 'RCL1', 'RFC4', 'RNPS1', 'RPF2', 'RPL14',
    'RPL22', 'RPL34', 'RPLP0', 'RPS2', 'RPS3', 'RPS5', 'RPS6',
    'RRP9', 'RSL1D1', 'SERBP1', 'SET', 'SF3B3', 'SLC19A1', 'SLC29A2',
    'SNRPA', 'SNRPD2', 'SNRPG', 'SRM', 'SRSF1', 'SRSF2', 'SRSF3',
    'SSB', 'SUPV3L1', 'TCP1', 'TFDP1', 'TOMM70', 'TRA2B', 'TRIM28',
    'TUFM', 'UBA2', 'UBE2E1', 'UNG', 'UTP20', 'WDR43', 'XPOT',
    'XPO5', 'YWHAE',
]


def calculate_myc_hallmark_scores(
    adata: ad.AnnData,
    phenotype_col: str = 'Phenotype',
    min_genes_required: int = 10,
) -> Dict:
    """
    Calcula MYC Hallmark scores usando gene sets de MSigDB.
    
    JUSTIFICACIÓN:
    mRNA de MYC (un solo gen) es un proxy pobre de actividad MYC.
    MYC es un factor de transcripción con >200 targets conocidos.
    El hallmark score captura la CASCADA COMPLETA: si los targets están
    activos, MYC está funcionalmente activo, independientemente del 
    nivel de mRNA de MYC mismo.
    
    HIPÓTESIS:
    - Desert: MYC_hallmark ALTO (MYC activo → STING reprimido)
    - Inflamed: MYC_hallmark BAJO (MYC inactivo → STING activo → CXCL9)
    - Excluded: MYC_hallmark VARIABLE (mecanismo es barrera, no MYC)
    
    Parameters
    ----------
    adata : AnnData con expresión génica y fenotipos
    phenotype_col : str — columna de fenotipos
    min_genes_required : int — mínimo de genes del set presentes
    
    Returns
    -------
    dict : {
        'scores_added': bool,
        'v1_genes_found': int,
        'v2_genes_found': int,
        'per_phenotype_stats': DataFrame,
        'kruskal_test': dict,
        'desert_vs_inflamed': dict,
    }
    """
    import scanpy as sc
    
    print("\n" + "=" * 80)
    print("BLOQUE 2: MYC HALLMARK GENE SCORES (MSigDB)")
    print("=" * 80)
    
    results = {
        'scores_added': False,
        'v1_genes_found': 0,
        'v2_genes_found': 0,
    }
    
    # Verificar genes disponibles
    available_genes = set(adata.var_names)
    
    v1_present = [g for g in MYC_TARGETS_V1 if g in available_genes]
    v2_present = [g for g in MYC_TARGETS_V2 if g in available_genes]
    
    results['v1_genes_found'] = len(v1_present)
    results['v2_genes_found'] = len(v2_present)
    
    print(f"  MYC_TARGETS_V1: {len(v1_present)}/{len(MYC_TARGETS_V1)} genes presentes")
    print(f"  MYC_TARGETS_V2: {len(v2_present)}/{len(MYC_TARGETS_V2)} genes presentes")
    
    # Calcular scores con scanpy
    scores_to_add = {}
    
    if len(v1_present) >= min_genes_required:
        try:
            sc.tl.score_genes(adata, v1_present, score_name='MYC_Hallmark_V1',
                              use_raw=False)
            scores_to_add['MYC_Hallmark_V1'] = True
            print(f"  MYC_Hallmark_V1 calculado ({len(v1_present)} genes)")
        except Exception as e:
            print(f"  Error calculando V1: {e}")
            # Fallback: media de z-scores
            _calculate_manual_score(adata, v1_present, 'MYC_Hallmark_V1')
            scores_to_add['MYC_Hallmark_V1'] = True
    else:
        print(f"  Insuficientes genes para V1 (mínimo {min_genes_required})")
    
    if len(v2_present) >= min_genes_required:
        try:
            sc.tl.score_genes(adata, v2_present, score_name='MYC_Hallmark_V2',
                              use_raw=False)
            scores_to_add['MYC_Hallmark_V2'] = True
            print(f"  MYC_Hallmark_V2 calculado ({len(v2_present)} genes)")
        except Exception as e:
            print(f"  Error calculando V2: {e}")
            _calculate_manual_score(adata, v2_present, 'MYC_Hallmark_V2')
            scores_to_add['MYC_Hallmark_V2'] = True
    
    # Score combinado (promedio V1+V2)
    if 'MYC_Hallmark_V1' in scores_to_add and 'MYC_Hallmark_V2' in scores_to_add:
        adata.obs['MYC_Hallmark_Combined'] = (
            adata.obs['MYC_Hallmark_V1'] + adata.obs['MYC_Hallmark_V2']
        ) / 2.0
        scores_to_add['MYC_Hallmark_Combined'] = True
        print(f"  MYC_Hallmark_Combined calculado")
    
    results['scores_added'] = len(scores_to_add) > 0
    
    if not results['scores_added']:
        print("  No se pudieron calcular hallmark scores")
        return results
    
    # --- Análisis por fenotipo ---
    if phenotype_col not in adata.obs.columns:
        print(f"  Columna '{phenotype_col}' no encontrada")
        return results
    
    phenotypes = ['Immune_Desert', 'Immune_Excluded', 'Inflamed']
    score_cols = [c for c in ['MYC_Hallmark_V1', 'MYC_Hallmark_V2', 
                               'MYC_Hallmark_Combined'] if c in adata.obs.columns]
    
    stats_rows = []
    
    for score_col in score_cols:
        print(f"\n  --- {score_col} por fenotipo ---")
        
        groups = {}
        for pheno in phenotypes:
            mask = adata.obs[phenotype_col].values == pheno
            vals = adata.obs.loc[mask, score_col].values
            vals = vals[np.isfinite(vals)]
            
            if len(vals) > 10:
                groups[pheno] = vals
                stats_rows.append({
                    'Score': score_col,
                    'Phenotype': pheno,
                    'N': len(vals),
                    'Mean': float(np.mean(vals)),
                    'Std': float(np.std(vals)),
                    'Median': float(np.median(vals)),
                })
                print(f"    {pheno}: mean={np.mean(vals):.4f} ± {np.std(vals):.4f} (n={len(vals)})")
        
        # Kruskal-Wallis entre todos los fenotipos
        if len(groups) >= 2:
            group_arrays = list(groups.values())
            try:
                kw_stat, kw_pval = kruskal(*group_arrays)
                results[f'kruskal_{score_col}'] = {
                    'statistic': float(kw_stat),
                    'pval': float(kw_pval),
                    'significant': kw_pval < 0.05,
                }
                print(f"    Kruskal-Wallis: H={kw_stat:.2f}, p={kw_pval:.2e}")
            except Exception:
                pass
        
        # Test específico: Desert vs Inflamed
        if 'Immune_Desert' in groups and 'Inflamed' in groups:
            stat, pval = mannwhitneyu(
                groups['Immune_Desert'], groups['Inflamed'], 
                alternative='greater'
            )
            # FIX-02 AUDIT v3: Usar cohens_d_pooled canónico (ddof=1, pooled ponderada)
            cohens_d = cohens_d_pooled(groups['Immune_Desert'], groups['Inflamed'])
            
            results[f'desert_vs_inflamed_{score_col}'] = {
                'MW_stat': float(stat),
                'MW_pval': float(pval),
                'cohens_d': float(cohens_d),
                'desert_mean': float(np.mean(groups['Immune_Desert'])),
                'inflamed_mean': float(np.mean(groups['Inflamed'])),
                'significant': pval < 0.05,
            }
            
            sig = "✅" if pval < 0.05 else "⚠️"
            direction = "higher" if cohens_d > 0 else "lower"
            print(f"    Desert vs Inflamed: d={cohens_d:.3f} ({direction}), "
                  f"p={pval:.2e} {sig}")
    
    results['per_phenotype_stats'] = pd.DataFrame(stats_rows)
    
    # Correlación hallmark vs STING
    print(f"\n  --- Correlación Hallmark vs STING ---")
    sting_expr = _safe_get_gene_expression(adata, 'TMEM173', aliases=['STING1', 'MITA'])
    
    if sting_expr is not None:
        for score_col in score_cols:
            score_vals = adata.obs[score_col].values
            valid = np.isfinite(score_vals) & np.isfinite(sting_expr) & (sting_expr > 0)
            
            if valid.sum() > 50:
                corr, pval = spearmanr(score_vals[valid], sting_expr[valid])
                results[f'hallmark_vs_sting_{score_col}'] = {
                    'spearman_rho': float(corr),
                    'pval': float(pval),
                }
                print(f"    {score_col} vs STING: ρ={corr:.4f}, p={pval:.2e}")
    
    return results


def _calculate_manual_score(adata, gene_list, score_name):
    """
    Fallback: calcular gene score como media de z-scores.
    Usado cuando sc.tl.score_genes() falla (e.g., datos en formato inesperado).
    
    FIX H6: Añadido np.asarray().ravel() defensivo para .mean(axis=0) y
    .std(axis=0). Si el input es scipy.sparse.csr_matrix (path no estándar),
    .mean() retorna np.matrix 2D en vez de array 1D, causando broadcasting
    incorrecto en z_scores. El fix es idempotente para arrays normales.
    """
    from scipy import sparse
    
    # Obtener índices de genes
    gene_idx = [i for i, g in enumerate(adata.var_names) if g in gene_list]
    
    if len(gene_idx) == 0:
        adata.obs[score_name] = 0.0
        return
    
    # Extraer expresión
    X = adata.X
    if sparse.issparse(X):
        X_sub = X[:, gene_idx].toarray()
    else:
        X_sub = np.asarray(X[:, gene_idx])
    
    # .ravel() garantiza 1D incluso si .mean() devuelve np.matrix
    means = np.asarray(X_sub.mean(axis=0)).ravel()
    stds = np.asarray(X_sub.std(axis=0)).ravel()
    stds[stds == 0] = 1.0  # Evitar división por cero
    
    z_scores = (X_sub - means) / stds
    
    # Score = media de z-scores
    adata.obs[score_name] = z_scores.mean(axis=1)


# ============================================================================
# BLOQUE 3: SUB-ANÁLISIS MACRÓFAGOS SPP1/CXCL9
# ============================================================================

def analyze_macrophage_polarization(
    adata: ad.AnnData,
    phenotype_col: str = 'Phenotype',
    macrophage_percentile: float = 75.0,
    min_spots: int = 30,
) -> Dict:
    """
    Analiza polarización de macrófagos (SPP1 vs CXCL9) por fenotipo.
    
    CONTEXTO:
    Bill et al. (2023, Science) demostró que el ratio CXCL9:SPP1 en
    macrófagos asociados a tumor (TAMs) es un biomarcador superior a la
    clasificación M1/M2 clásica. El ratio predice respuesta a ICI en
    múltiples tipos de cáncer.
    
    HIPÓTESIS:
    En spots con alta abundancia de macrófagos:
    - Desert: CXCL9↓, SPP1↑ → ratio bajo → TAM pro-tumoral
    - Excluded: CXCL9 moderado, SPP1 moderado → ratio intermedio
    - Inflamed: CXCL9↑, SPP1↓ → ratio alto → TAM anti-tumoral
    
    Parameters
    ----------
    adata : AnnData con expresión, deconvolución y fenotipos
    phenotype_col : str — columna de fenotipos
    macrophage_percentile : float — percentil para definir "alta abundancia de TAMs"
    min_spots : int — mínimo de spots por fenotipo
    
    Returns
    -------
    dict con resultados del análisis
    """
    print("\n" + "=" * 80)
    print("BLOQUE 3: POLARIZACIÓN DE MACRÓFAGOS (SPP1 vs CXCL9)")
    print("=" * 80)
    
    results = {
        'ratio_calculated': False,
        'per_phenotype': {},
    }
    
    # 1. Obtener expresión de SPP1 y CXCL9
    spp1_expr = _safe_get_gene_expression(adata, 'SPP1')
    cxcl9_expr = _safe_get_gene_expression(adata, 'CXCL9')
    
    if spp1_expr is None or cxcl9_expr is None:
        print("  SPP1 o CXCL9 no encontrados en datos")
        return results
    
    # 2. Encontrar spots con alta abundancia de macrófagos
    macro_col = _find_abundance_column_v3(adata, 'Macrophage')
    
    if macro_col is None:
        print("  Columna de abundancia de Macrófagos no encontrada")
        # Alternativa: usar todos los spots sin filtrar por macrófagos
        print("  → Usando todos los spots (sin filtro por macrófagos)")
        high_macro_mask = np.ones(adata.n_obs, dtype=bool)
    else:
        macro_vals = adata.obs[macro_col].values.astype(float)
        macro_thresh = np.percentile(macro_vals, macrophage_percentile)
        high_macro_mask = macro_vals >= macro_thresh
        print(f"  Columna macrófagos: {macro_col}")
        print(f"  Threshold (p{macrophage_percentile:.0f}): {macro_thresh:.3f}")
        print(f"  Spots con alta abundancia TAM: {high_macro_mask.sum():,}")
    
    # 3. Calcular ratio CXCL9:SPP1 (log2)
    # Si adata.X está en espacio log1p (post-scanpy normalize_total + log1p),
    # los valores de expresión son log(1+X), NO counts crudos.
    # Calcular log2(log(1+A) / log(1+B)) es INCORRECTO para fold change.
    # Debemos revertir a espacio lineal con expm1() antes de dividir.
    #
    # Auto-detección: si max(expresión) < 20, probablemente log-transformado.
    # Si max > 100, probablemente raw counts.
    
    max_expr = max(
        np.nanmax(spp1_expr) if len(spp1_expr) > 0 else 0,
        np.nanmax(cxcl9_expr) if len(cxcl9_expr) > 0 else 0
    )
    
    # Pseudocount 1.0 consistente con mechanism_validation.py (no 0.1)
    pseudocount = 1.0
    
    if max_expr < 20:
        # Datos en espacio log1p → revertir a lineal antes de ratio
        print(f"  Auto-detectado: datos en espacio log1p (max={max_expr:.2f})")
        print(f"  Aplicando np.expm1() antes del ratio (FIX H3)")
        cxcl9_linear = np.expm1(cxcl9_expr)  # e^x - 1 = inversa de log1p
        spp1_linear = np.expm1(spp1_expr)
        ratio = (cxcl9_linear + pseudocount) / (spp1_linear + pseudocount)
    else:
        # Raw counts o RPKM → ratio directo
        print(f"  Auto-detectado: datos en espacio lineal (max={max_expr:.2f})")
        ratio = (cxcl9_expr + pseudocount) / (spp1_expr + pseudocount)
    
    log2_ratio = np.log2(ratio)
    
    # NO sobreescribir CXCL9_SPP1_log2ratio si mechanism_validation.py
    # ya lo calculó (con pseudocount=1.0 desde .raw). Si existe, reusar.
    # Si no existe, guardar como 'CXCL9_SPP1_log2ratio_macro' para evitar conflicto.
    if 'CXCL9_SPP1_log2ratio' in adata.obs.columns:
        print(f"  [INFO] CXCL9_SPP1_log2ratio ya existe (de mechanism_validation.py). Reusando.")
        log2_ratio_existing = adata.obs['CXCL9_SPP1_log2ratio'].values
        # Usar el existente para análisis downstream (consistencia con pipeline)
        log2_ratio = log2_ratio_existing
    else:
        adata.obs['CXCL9_SPP1_log2ratio'] = log2_ratio
    results['ratio_calculated'] = True
    
    print(f"\n  Ratio CXCL9:SPP1 global: median={np.median(log2_ratio):.3f}")
    
    # 4. Analizar por fenotipo EN SPOTS CON ALTOS MACRÓFAGOS
    if phenotype_col not in adata.obs.columns:
        print(f"  Columna '{phenotype_col}' no encontrada")
        return results
    
    phenotypes = ['Immune_Desert', 'Immune_Excluded', 'Inflamed']
    phenotype_data = {}
    
    print(f"\n  === RATIO CXCL9:SPP1 EN NICHOS TAM-RICOS ===")
    print(f"  {'Phenotype':<20} {'N':<8} {'CXCL9':<12} {'SPP1':<12} {'Log2 Ratio':<12}")
    print(f"  {'-'*64}")
    
    for pheno in phenotypes:
        combined_mask = (
            (adata.obs[phenotype_col].values == pheno) & 
            high_macro_mask
        )
        
        n_spots = combined_mask.sum()
        
        if n_spots < min_spots:
            print(f"  {pheno:<20} {n_spots:<8} (insuficientes)")
            continue
        
        cxcl9_sub = cxcl9_expr[combined_mask]
        spp1_sub = spp1_expr[combined_mask]
        ratio_sub = log2_ratio[combined_mask]
        ratio_sub = ratio_sub[np.isfinite(ratio_sub)]
        
        phenotype_data[pheno] = {
            'cxcl9': cxcl9_sub,
            'spp1': spp1_sub,
            'ratio': ratio_sub,
        }
        
        results['per_phenotype'][pheno] = {
            'n_spots': int(n_spots),
            'cxcl9_mean': float(np.mean(cxcl9_sub)),
            'cxcl9_median': float(np.median(cxcl9_sub)),
            'spp1_mean': float(np.mean(spp1_sub)),
            'spp1_median': float(np.median(spp1_sub)),
            'ratio_mean': float(np.mean(ratio_sub)),
            'ratio_median': float(np.median(ratio_sub)),
            'ratio_std': float(np.std(ratio_sub)),
        }
        
        print(f"  {pheno:<20} {n_spots:<8} "
              f"{np.mean(cxcl9_sub):<12.4f} {np.mean(spp1_sub):<12.4f} "
              f"{np.mean(ratio_sub):<12.4f}")
    
    # 5. Tests estadísticos
    print(f"\n  === TESTS ESTADÍSTICOS ===")
    
    # Kruskal-Wallis
    if len(phenotype_data) >= 2:
        ratio_groups = [d['ratio'] for d in phenotype_data.values() if len(d['ratio']) > 10]
        if len(ratio_groups) >= 2:
            kw_stat, kw_pval = kruskal(*ratio_groups)
            results['kruskal_ratio'] = {
                'statistic': float(kw_stat),
                'pval': float(kw_pval),
            }
            print(f"  Kruskal-Wallis (ratio): H={kw_stat:.2f}, p={kw_pval:.2e}")
    
    # Pairwise comparisons
    pairs = [
        ('Immune_Desert', 'Inflamed', 'less'),      # Desert < Inflamed expected
        ('Immune_Desert', 'Immune_Excluded', 'less'), # Desert < Excluded expected
        ('Immune_Excluded', 'Inflamed', 'less'),     # Excluded < Inflamed expected
    ]
    
    pairwise_results = []
    
    for pheno1, pheno2, alternative in pairs:
        if pheno1 in phenotype_data and pheno2 in phenotype_data:
            r1 = phenotype_data[pheno1]['ratio']
            r2 = phenotype_data[pheno2]['ratio']
            
            if len(r1) > 10 and len(r2) > 10:
                stat, pval = mannwhitneyu(r1, r2, alternative=alternative)
                
                # Usar cohens_d_pooled canónico (ddof=1, pooled ponderada)
                d = cohens_d_pooled(r1, r2)
                
                pairwise_results.append({
                    'Comparison': f'{pheno1} vs {pheno2}',
                    'Alternative': alternative,
                    'MW_stat': float(stat),
                    'P_value': float(pval),
                    'Cohens_d': float(d),
                    'Significant': pval < 0.05,
                })
                
                sig = "✅" if pval < 0.05 else "⚠️"
                print(f"  {pheno1} vs {pheno2}: d={d:.3f}, p={pval:.2e} {sig}")
    
    results['pairwise_tests'] = pd.DataFrame(pairwise_results)
    
    # Corrección FDR (Benjamini-Hochberg) para tests pairwise múltiples
    # Con 3 comparaciones pairwise del mismo dataset, el riesgo de falsos positivos
    # es moderado. 
    if len(pairwise_results) > 1:
        try:
            from statsmodels.stats.multitest import multipletests
            raw_pvals = [r['P_value'] for r in pairwise_results]
            reject, pvals_corrected, _, _ = multipletests(raw_pvals, method='fdr_bh')
            
            for i, row in enumerate(pairwise_results):
                row['P_value_FDR'] = float(pvals_corrected[i])
                row['Significant_FDR'] = bool(reject[i])
            
            results['pairwise_tests'] = pd.DataFrame(pairwise_results)
            print(f"\n  FDR corregido (Benjamini-Hochberg):")
            for r in pairwise_results:
                sig = "✅" if r['Significant_FDR'] else "⚠️"
                print(f"    {r['Comparison']}: p_raw={r['P_value']:.2e}, "
                      f"p_FDR={r['P_value_FDR']:.2e} {sig}")
        except ImportError:
            print("  statsmodels no disponible. FDR no aplicado.")
            print("    pip install statsmodels --break-system-packages")
            # Fallback: corrección manual de Bonferroni (más conservadora)
            n_tests = len(pairwise_results)
            for r in pairwise_results:
                r['P_value_FDR'] = min(r['P_value'] * n_tests, 1.0)
                r['Significant_FDR'] = r['P_value_FDR'] < 0.05
            results['pairwise_tests'] = pd.DataFrame(pairwise_results)
            print(f"  Fallback: Bonferroni aplicado (n_tests={n_tests})")
    
    # 6. Correlación ratio con STING
    sting_expr = _safe_get_gene_expression(adata, 'TMEM173', aliases=['STING1', 'MITA'])
    
    if sting_expr is not None:
        valid = np.isfinite(log2_ratio) & (sting_expr > 0) & high_macro_mask
        if valid.sum() > 50:
            corr, pval = spearmanr(log2_ratio[valid], sting_expr[valid])
            results['ratio_vs_sting'] = {
                'spearman_rho': float(corr),
                'pval': float(pval),
            }
            print(f"\n  Ratio vs STING (en TAM-ricos): ρ={corr:.4f}, p={pval:.2e}")
    
    return results


# ============================================================================
# HELPERS COMPARTIDOS
# ============================================================================

def _safe_get_gene_expression(
    adata: ad.AnnData,
    gene: str,
    aliases: list = None,
) -> np.ndarray:
    """
    Extrae expresión de un gen de forma segura, soportando sparse/dense
    y aliases genéticos.
    
    Prioriza .raw (log1p-normalizado) sobre .X para
    consistencia con get_gene_expression() de mechanism_validation.py.
    """
    from scipy import sparse
    
    genes_to_try = [gene] + (aliases or [])
    
    # Priorizar .raw (log1p-normalizado, consistente con mechanism_validation.py)
    if hasattr(adata, 'raw') and adata.raw is not None:
        for g in genes_to_try:
            if g in adata.raw.var_names:
                X_raw = adata.raw[:, g].X
                if sparse.issparse(X_raw):
                    return np.asarray(X_raw.todense()).ravel().astype(float)
                return np.asarray(X_raw).ravel().astype(float)
    
    # Fallback a .X
    for g in genes_to_try:
        if g in adata.var_names:
            X = adata[:, g].X
            if sparse.issparse(X):
                return np.asarray(X.todense()).ravel().astype(float)
            return np.asarray(X).ravel().astype(float)
    
    return None


def _find_abundance_column_v3(
    adata: ad.AnnData,
    cell_type: str,
) -> str:
    """
    Busca columna de abundancia celular en adata.obs.
    Versión v3 con más candidatos que v2.4.
    """
    candidates = [
        # Añadir formato REAL de HPC (meanscell_ sin underscore)
        f'meanscell_abundance_w_sf_{cell_type}',
        f'q05cell_abundance_w_sf_{cell_type}',
        f'q05_cell_abundance_w_sf_{cell_type}',
        f'q05_{cell_type}',
        f'{cell_type}_q05',
        f'means_cell_abundance_w_sf_{cell_type}',
        f'means_{cell_type}',
        f'{cell_type}_means',
        f'{cell_type}_abundance',
        cell_type,
    ]
    
    for col in candidates:
        if col in adata.obs.columns:
            return col
    
    # Fuzzy search
    for col in adata.obs.columns:
        if cell_type.lower() in col.lower():
            return col
    
    return None


# ============================================================================
# WRAPPER: run_extended_mechanism_validation()
# ============================================================================

def run_extended_mechanism_validation(
    adata: ad.AnnData,
    save_dir: str = None,
) -> Dict:
    """
    Ejecuta los 3 bloques nuevos de validación mecanística.
    
    Llamar DESPUÉS de validate_silencing_mechanism() existente.
    
    Parameters
    ----------
    adata : AnnData completo con deconvolución y fenotipos
    save_dir : str — directorio para guardar CSVs y JSONs
    
    Returns
    -------
    dict con todos los resultados
    """
    from pathlib import Path
    import json
    
    print("\n" + "=" * 80)
    print("EXTENDED MECHANISM VALIDATION v3.0")
    print("3 bloques nuevos del Dictamen de Crítica")
    print("=" * 80)
    
    all_results = {}
    
    # BLOQUE 1: Desert-only correlations
    try:
        desert_corr_df = validate_desert_only_correlations(adata)
        all_results['desert_only_correlations'] = desert_corr_df
    except Exception as e:
        print(f"\n  BLOQUE 1 falló: {e}")
        import traceback
        traceback.print_exc()
    
    # BLOQUE 2: MYC Hallmark scores
    try:
        hallmark_results = calculate_myc_hallmark_scores(adata)
        all_results['myc_hallmark'] = hallmark_results
    except Exception as e:
        print(f"\n  BLOQUE 2 falló: {e}")
        import traceback
        traceback.print_exc()
    
    # BLOQUE 3: Macrophage polarization
    try:
        macro_results = analyze_macrophage_polarization(adata)
        all_results['macrophage_polarization'] = macro_results
    except Exception as e:
        print(f"\n  BLOQUE 3 falló: {e}")
        import traceback
        traceback.print_exc()
    
    # Guardar resultados
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # CSV de correlaciones Desert-only
        if 'desert_only_correlations' in all_results:
            df = all_results['desert_only_correlations']
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                df.to_csv(save_path / 'desert_only_correlations.csv', index=False)
                print(f"\n  ✓ Saved: {save_path / 'desert_only_correlations.csv'}")
        
        # CSV de hallmark stats
        if 'myc_hallmark' in all_results:
            h = all_results['myc_hallmark']
            if 'per_phenotype_stats' in h and isinstance(h['per_phenotype_stats'], pd.DataFrame):
                h['per_phenotype_stats'].to_csv(
                    save_path / 'myc_hallmark_by_phenotype.csv', index=False
                )
                print(f"  ✓ Saved: {save_path / 'myc_hallmark_by_phenotype.csv'}")
        
        # CSV de macrophage pairwise tests
        if 'macrophage_polarization' in all_results:
            m = all_results['macrophage_polarization']
            if 'pairwise_tests' in m and isinstance(m['pairwise_tests'], pd.DataFrame):
                m['pairwise_tests'].to_csv(
                    save_path / 'macrophage_polarization_tests.csv', index=False
                )
                print(f"  ✓ Saved: {save_path / 'macrophage_polarization_tests.csv'}")
        
        # JSON resumen
        summary = {}
        for key, val in all_results.items():
            if isinstance(val, dict):
                # Filtrar DataFrames para JSON
                clean = {k: v for k, v in val.items() 
                         if not isinstance(v, (pd.DataFrame, np.ndarray))}
                summary[key] = clean
        
        with open(save_path / 'extended_mechanism_v3_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  ✓ Saved: {save_path / 'extended_mechanism_v3_summary.json'}")
    
    # Resumen final
    print(f"\n{'=' * 80}")
    print(f"EXTENDED MECHANISM VALIDATION v3.0 COMPLETADO")
    print(f"{'=' * 80}")
    
    return all_results

# ============================================================================
# MAIN (testing independiente)
# ============================================================================

if __name__ == '__main__':
    import scanpy as sc
    import sys
    from pathlib import Path
    
    print("=" * 80)
    print("MECHANISM VALIDATION — TESTING INDEPENDIENTE")
    print("=" * 80)
    
    # Buscar adata
    possible_paths = [
        Path("/home/external/frjimenez/fabian/genoma/data/processed/adata_with_phenotypes.h5ad"),
        Path("data/processed/adata_with_phenotypes.h5ad"),
    ]
    
    adata_path = None
    for p in possible_paths:
        if p.exists():
            adata_path = p
            break
    
    if adata_path is None:
        print("No se encontró adata. Rutas buscadas:")
        for p in possible_paths:
            print(f"  {p}")
        sys.exit(1)
    
    print(f"\nCargando: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    print(f"  Spots: {adata.n_obs:,} | Genes: {adata.n_vars:,}")
    
    if 'Phenotype' not in adata.obs.columns:
        print("No hay columna 'Phenotype'")
        sys.exit(1)
    
    # Ejecutar
    save_dir = Path("results/tables")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results = run_extended_mechanism_validation(adata, save_dir=str(save_dir))
    
    print(f"\n Completado. Resultados en: {save_dir}")
