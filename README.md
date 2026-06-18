# Spatial decomposition of immune-cold TNBC: a CAF barrier and the limits of driver inference

Analysis code for the manuscript *"A fibroblast barrier and the limits of driver inference in immune-cold triple-negative breast cancer."*

This repository contains the computational pipeline used to (i) classify Visium spatial transcriptomics spots into microenvironment phenotypes, (ii) quantify a cancer-associated fibroblast (CAF) stromal barrier under controls against classifier circularity, (iii) test tumour-intrinsic MYC activity as a candidate driver of the Immune Desert niche under regulon-composition and tumour-fraction control, and (iv) assess recoverability of the phenotype signature in bulk cohorts.

The code is released so that the analyses and figures in the manuscript can be reproduced. No raw data are redistributed; all datasets are obtained from their public repositories (see *Data*).

\---

## 1\. Citation

If you use this code, please cite the manuscript (full citation on acceptance) and this repository:

```
\[Author list]. A fibroblast barrier and the limits of driver inference in
immune-cold triple-negative breast cancer. \[Journal], \[year].
```

\---

## 2\. Environments

The pipeline uses **two Conda environments**, because the Bayesian deconvolution and transcription-factor-activity stack (`cell2location`, `scvi-tools`, `decoupler`) requires `numpy < 2` and `scanpy 1.9.x`, which are incompatible with the newer versions used for the rest of the analysis.

|Environment file|Name|Used for|
|-|-|-|
|`environment\_main.yml`|`spatial\_tnbc\_a`|Classification, CAF barrier, robustness, checkpoint landscape, chemotaxis, bulk recoverability, figures (Stages 1, 3, 4, 5, 6, 8 and figures)|
|`environment\_deconv\_myc.yml`|`tnbc\_spatial`|Cell2Location deconvolution (Stage 2) and MYC transcription-factor activity (Stage 7)|

```bash
# main analysis environment
conda env create -f environment\_main.yml
conda activate spatial\_tnbc\_a

# deconvolution + MYC environment
conda env create -f environment\_deconv\_myc.yml
conda activate tnbc\_spatial
```

Key versions — main (`spatial\_tnbc\_a`): Python 3.10.18, scanpy 1.11.4, anndata 0.11.4, squidpy 1.6.5, scikit-learn 1.7.2, statsmodels 0.14.5, lifelines 0.30.0, numpy 2.2.6. Deconvolution/MYC (`tnbc\_spatial`): cell2location 0.1.5, scvi-tools 1.1.2, decoupler 2.1.4, scanpy 1.9.8, numpy 1.23.5.

**HPC note.** Several scripts begin with a minimal `torch` stub inserted into `sys.modules` *before* `anndata`/`scanpy` are imported. On HPC nodes with older kernels, `anndata.experimental.pytorch` imports `torch`, which can exhaust the static TLS space; the stub satisfies the import without loading the shared library. Keep it if you run on a similar cluster. `torch` is pinned to CUDA builds (cu121 in the main environment, cu118 in the deconvolution/MYC environment); on a machine without an NVIDIA GPU, install the corresponding CPU build instead.

\---

## 3\. Data

|Dataset|Accession|Role|Access|
|-|-|-|-|
|Discovery (43 sections, 22 patients)|GSE210616|Spatial discovery cohort|NCBI GEO|
|Validation (15 sections, 11 tumours)|GSE213688|Independent spatial validation|NCBI GEO|
|scRNA-seq reference (26 tumours)|GSE176078|Cell2Location reference|NCBI GEO|
|METABRIC (n = 209 basal-like)|—|Bulk recoverability|cBioPortal|
|TCGA-BRCA (n = 171 basal-like)|—|Bulk recoverability|NCI GDC|

Processed AnnData objects are archived at Zenodo (DOI to be assigned).

\---

## 4\. Repository layout

```
.
├── README.md
├── environment.yml
├── LICENSE
│
├── src/
│   ├── config/
│   │   ├── config.py                  # canonical signatures, palette, thresholds
│   │   ├── config\_additions.py        # additional shared constants
│   │   └── utils\_stats.py             # Cohen's d, BH-FDR, correlation helpers
│   │
│   ├── 01\_preprocessing/
│   │   ├── preprocessing.py
│   │   └── reclassify\_validation.py
│   │
│   ├── 02\_deconvolution/
│   │   └── deconvolution.py
│   │
│   ├── 03\_classification/
│   │   └── phenotype\_classifier.py
│   │
│   ├── 04\_caf\_barrier/
│   │   ├── mechanism\_validation.py
│   │   ├── mechanism\_validation\_additions.py
│   │   ├── patient\_level\_analysis.py
│   │   ├── marker\_gene\_scoring.py
│   │   ├── orthogonal\_validation.py
│   │   └── spatial\_analysis\_v2.py
│   │
│   ├── 05\_robustness/
│   │   ├── sensitivity\_analysis.py
│   │   ├── sensitivity\_analysis\_additions.py
│   │   ├── robustness\_stress\_tests.py
│   │   ├── spatial\_coherence\_analysis.py
│   │   ├── comprehensive\_celltype\_analysis.py
│   │   └── ambiguity\_tradeoff.py
│   │
│   ├── 06\_checkpoint\_chemokine/
│   │   ├── validation.py
│   │   └── checkpoint\_landscape.py
│   │
│   ├── 07\_myc\_inference/
│   │   ├── myc\_tf\_activity\_decoupler.py
│   │   ├── fix\_myc\_tf\_clean\_regulon.py
│   │   ├── fix\_myc\_tf\_clean\_regulon\_wrapper.py
│   │   ├── fix\_myc\_tf\_proliferation\_confound.py
│   │   └── investigate\_myc\_sting\_mechanism.py
│   │
│   ├── 08\_bulk/
│   │   └── bulk\_validation.py
│   │
│   └── fixes/
│       ├── fix\_validation\_celltype\_normalization.py
│       ├── fix\_survival\_fdr.py
│       ├── fix\_patient\_level\_validation.py
│       ├── fix\_b1\_auc\_loocv.py
│       └── fix\_b2\_gene\_dropout\_correct.py
│
├── figures/
│   ├── publication\_figures\_v10\_main.py
│   └── publication\_figures\_v10\_supp.py
│
└── deprecated/                        # archived; not part of the reported pipeline
    ├── geodesic\_benchmark.py
    ├── weighted\_geodesic.py
    ├── spatial\_analysis\_v2\_additions.py
    └── README.md
```

The directory numbering follows the data flow: each stage consumes the AnnData object written by the previous one.

\---

## 5\. How to reproduce (execution order)

Stages 2 and 6-MYC run in the `tnbc\_spatial` environment; all other stages run in `spatial\_tnbc\_a` (see §2). Switch environments where indicated.

```bash
# 0. Preprocessing and QC  ->  adata_preprocessed.h5ad (29,946 genes)
#    >>> environment: spatial_tnbc_a  (and all stages below unless noted)
python src/01_preprocessing/preprocessing.py
python src/01_preprocessing/reclassify_validation.py

# 1. Cell2Location deconvolution (run once; GPU recommended; reads step 0 output)
#    >>> environment: tnbc_spatial
python src/02_deconvolution/deconvolution.py

# 2. Phenotype classification (five niches)
#    >>> environment: spatial_tnbc_a
python src/03_classification/phenotype_classifier.py

# 3. CAF barrier: seven convergent estimators
python src/04_caf_barrier/mechanism_validation.py
python src/04_caf_barrier/mechanism_validation_additions.py
python src/04_caf_barrier/patient_level_analysis.py
python src/04_caf_barrier/marker_gene_scoring.py
python src/04_caf_barrier/orthogonal_validation.py
python src/04_caf_barrier/spatial_analysis_v2.py

# 4. Robustness
python src/05_robustness/sensitivity_analysis.py
python src/05_robustness/sensitivity_analysis_additions.py
python src/05_robustness/robustness_stress_tests.py
python src/05_robustness/spatial_coherence_analysis.py
python src/05_robustness/comprehensive_celltype_analysis.py
python src/05_robustness/ambiguity_tradeoff.py

# 5. Checkpoint landscape, chemotaxis, cross-cohort validation
python src/06_checkpoint_chemokine/validation.py
python src/06_checkpoint_chemokine/checkpoint_landscape.py

# 6. MYC inference under confound control (see §7 for the regulon note)
#    >>> environment: tnbc_spatial
python src/07_myc_inference/fix_myc_tf_clean_regulon_wrapper.py
python src/07_myc_inference/fix_myc_tf_proliferation_confound.py

# 7. Bulk recoverability + survival
#    >>> environment: spatial_tnbc_a
python src/08_bulk/bulk_validation.py

# 8. Post-hoc corrections applied after the primary run (§8)
python src/fixes/fix_validation_celltype_normalization.py
python src/fixes/fix_survival_fdr.py
python src/fixes/fix_patient_level_validation.py
python src/fixes/fix_b1_auc_loocv.py
python src/fixes/fix_b2_gene_dropout_correct.py

# 9. Figures
python figures/publication_figures_v10_main.py
python figures/publication_figures_v10_supp.py
```

A fixed random seed is set in the stochastic procedures.

\---

## 6\. Module-by-module description

### Configuration and shared utilities (`src/config/`)

| Module | Purpose | Manuscript link |
|---|---|---|
| `config.py` | Canonical definitions of all gene signatures (Tumour, Silencing, Immune, Barrier, MHC-I, ISG, chemokine), the phenotype colour palette and global thresholds. | Methods → *Spatial phenotype classification*, *Gene-signature readouts* |
| `config_additions.py` | Additional shared constants used by later stages (see §7). | Methods |
| `utils_stats.py` | Cohen's *d* (pooled variance, ddof = 1), Benjamini-Hochberg FDR, Spearman helpers. | Methods → *Statistics and reproducibility* |

### Stage 0 — Preprocessing (`src/01_preprocessing/`)

| Module | Purpose | Manuscript link |
|---|---|---|
| `preprocessing.py` | QC filtering (≥250 genes, ≥800 UMIs, <20% mito, tissue mask), library-size normalisation to 10,000 UMIs, log1p; builds the 29,946-gene `adata.raw` used as the source for all gene-expression scores. | Methods → *Spatial data preprocessing* |
| `reclassify_validation.py` | Applies the discovery-derived classifier thresholds to GSE213688 without recalculation, including proportion normalisation of Cell2Location abundances for the validation CAF contrast. | Methods; validation proportions |

### Stage 1 — Deconvolution (`src/02_deconvolution/`)

| Module | Purpose | Manuscript link |
|---|---|---|
| `deconvolution.py` | Learns the Cell2Location reference from GSE176078 (15 cell types) and estimates per-spot abundances (q05). Reads `adata_preprocessed.h5ad` produced by Stage 0. Output is used as an orthogonal layer, not as an input to classification. | Methods → *Cell-type deconvolution*; C2L panels |

### Stage 2 — Classification (`src/03_classification/`)

| Module | Purpose | Manuscript link |
|---|---|---|
| `phenotype_classifier.py` | Hierarchical rule assigning each spot to one of five phenotypes via Tumour/Immune gating and `Mechanism_Diff = Silencing − Barrier`. Produces the phenotype proportions. | Methods; Results section 1 |

### Stage 3 — CAF barrier (`src/04_caf_barrier/`)

| Module | Purpose | Manuscript link |
|---|---|---|
| `mechanism_validation.py` + `mechanism_validation_additions.py` | Spot-level CAF contrast (Excluded vs Desert) and related per-niche comparisons. | spot-level estimator |
| `patient_level_analysis.py` | Section-level aggregation (per-section medians) to limit pseudoreplication. | section-level estimator |
| `marker_gene_scoring.py` | CAF marker-gene score from raw expression, independent of Cell2Location — anti-circularity control. | control |
| `orthogonal_validation.py` | K-means on the abundance matrix and spatial-context classification — two further anti-circularity controls. | controls |
| `spatial_analysis_v2.py` | CAF abundance gradient across niches and representative spatial maps. | Figures |

### Stage 4 — Robustness (`src/05_robustness/`)

| Module | Purpose | Manuscript link |
|---|---|---|
| `sensitivity_analysis.py` + `sensitivity_analysis_additions.py` | Parameter sweep over the classification-threshold grid. | Supplementary Figures |
| `robustness_stress_tests.py` | Label-shuffling permutation and gene-dropout simulation. | Supplementary Figures |
| `spatial_coherence_analysis.py` | Per-section Moran's I of phenotype labels. | Supplementary Figures |
| `comprehensive_celltype_analysis.py` | Effect sizes across all 15 deconvolved cell types (CAF specificity). | Supplementary Figures |
| `ambiguity_tradeoff.py` | Sensitivity of the Ambiguous Cold boundary. | Methods (robustness) |

### Stage 5 — Checkpoint landscape and chemotaxis (`src/06_checkpoint_chemokine/`)

| Module | Purpose | Manuscript link |
|---|---|---|
| `validation.py` | Consolidated cross-dataset validation and chemokine–immune-cell Spearman correlations. | chemotaxis; cross-cohort replication |
| `checkpoint_landscape.py` | Seventeen-gene immune co-regulatory landscape across phenotypes (Mann-Whitney + BH-FDR), with Inflamed as reference. | Results section 5 |

### Stage 6 — MYC inference (`src/07_myc_inference/`)

The repository includes both the initial regulon-based exploration and the composition-controlled analysis reported in the paper, so the full reasoning is transparent. **To reproduce the reported MYC result, run the `fix_myc_tf_clean_*` scripts** (step 6 of §5).

| Module | Purpose | Manuscript link |
|---|---|---|
| `myc_tf_activity_decoupler.py` | Base ULM transcription-factor-activity routine (decoupleR-style) over a CollecTRI-derived MYC regulon. Intended to be driven by the clean-regulon wrapper rather than run directly. | provides the activity routine for Fig. 7 |
| `fix_myc_tf_clean_regulon.py` | Restricts the MYC regulon to proliferation/metabolism targets disjoint from the functional readout gene sets (ISG, MHC-I, STING), removing the readout-gene overlap. | Figure |
| `fix_myc_tf_clean_regulon_wrapper.py` | Executable wrapper that runs the activity routine with the restricted regulon and writes the `_clean` outputs. Entry point for the reported analysis. | runs the Fig. 7 analysis |
| `fix_myc_tf_proliferation_confound.py` | Relates the cleaned MYC activity score to deconvolved tumour-cell abundance within Desert spots. | Figures |
| `investigate_myc_sting_mechanism.py` | Standalone exploratory analysis with the original (unrestricted) regulon; retained for transparency. Its unrestricted-regulon value is the one shown in Fig. 7a as the set-aside comparison. | Set-aside value of Fig. 7 |

### Stage 7 — Bulk (`src/08_bulk/`)

| Module | Purpose | Manuscript link |
|---|---|---|
| `bulk_validation.py` | Self-contained entry point. Scores METABRIC/TCGA basal-like samples by a Barrier-Silencing index and assesses AUC, leave-one-out CV and Cox survival. | Bulk recoverability and survival |

### Post-hoc fixes (`src/fixes/`)

Self-contained scripts applied after the primary run to refine specific outputs.

| Module | Refines |
|---|---|
| `fix_validation_celltype_normalization.py` | Proportion normalisation of validation abundances (validation CAF contrast). |
| `fix_patient_level_validation.py` | Patient-level validation CAF. |
| `fix_survival_fdr.py` | FDR correction of survival contrasts. |
| `fix_b1_auc_loocv.py` | Leave-one-out cross-validated AUC. |
| `fix_b2_gene_dropout_correct.py` | Sparse-matrix-corrected gene-dropout simulation. |

### Figures (`figures/`)

| Module | Produces |
|---|---|
| `publication_figures_v10_main.py` | Main Figure panels. |
| `publication_figures_v10_supp.py` | Supplementary Figure panels. |

Fig. 1 is a schematic prepared in BioRender and is not generated by code.

### Archived (`deprecated/`)

Retained for completeness but not part of the reported pipeline; not imported by any active module.

| Module | Note |
|---|---|
| `geodesic_benchmark.py`, `weighted_geodesic.py` | Geodesic-distance exploration not used in the final analysis. |
| `spatial_analysis_v2_additions.py` | Geodesic-support extension to the spatial-analysis module; superseded. |

### Archived (`deprecated/`)

Retained for completeness but not part of the reported pipeline; not imported by any active module.

|Module|Note|
|-|-|
|`geodesic\_benchmark.py`, `weighted\_geodesic.py`|Geodesic-distance exploration not used in the final analysis.|
|`spatial\_analysis\_v2\_additions.py`|Geodesic-support extension to the spatial-analysis module; superseded.|

---

## 7\. Notes on signatures and reproducibility

* **Phenotype colour palette.** The publication figures (`publication_figures_v10_*.py`) use the Okabe-Ito colorblind-safe palette. Intermediate outputs produced by the analytic modules use the internal palette stored in `config.py`. All panels in the manuscript use Okabe-Ito.
* The MYC analysis is reported from the restricted-regulon scripts (`fix_myc_tf_clean_*`), run with decoupler 2.1.4 in the `tnbc_spatial` environment; see Stage 6 above.
* Cell2Location (v0.1.5) abundances are estimates rather than direct counts; the three classifier-independent CAF estimators do not depend on deconvolution and provide a check against that dependence.
---

## 8\. License

Released under the MIT License (see `LICENSE`). Third-party datasets retain their original licenses and terms of use.
