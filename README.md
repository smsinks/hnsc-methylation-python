# HNSC methylation: a reproducible Python pipeline

Python functions and a Jupyter notebook for the patient-level analysis in
**DNA methylation differs between head and neck tumours from Black and White
patients**. The repository includes the processed study inputs, frozen probe
annotations, imported manuscript summaries and reference numerical results.
After the Python environment is installed, the analysis runs without data
downloads or calls to online services. MATLAB is not required.

AFR and EUR denote the TCGA clinical race categories **Black or African
American** and **White**, respectively. They are not estimates of genetic
ancestry. The analyses describe associations and discrimination within this
cohort; they do not establish causal effects or external clinical validity.

## Start with the notebook

Use Python 3.11 or newer. From the repository root, create an isolated
environment and install the package with the notebook dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[notebook]'
python -m ipykernel install --user --name hnsc-methylation --display-name "Python (HNSC methylation)"
jupyter lab notebooks/HNSC_Methylation_Pipeline.ipynb
```

On Windows, activate the environment with `.venv\Scripts\activate` in Command
Prompt, or `.venv\Scripts\Activate.ps1` in PowerShell. The remaining installation
commands use the same Python environment. `python -m pip install -r
requirements.txt` is an equivalent installation route from the repository root.

Select the **Python (HNSC methylation)** kernel and choose **Restart Kernel and
Run All Cells**. The notebook finds the repository when opened from either its
root or the `notebooks/` directory. Its cells call reusable package functions
and show selected tables and all generated manuscript figures.

The default run uses **100 classifier label permutations**, matching the paper.
Classification and exact CpG–gene association calculations are the most
substantial stages. Allow the notebook to finish; elapsed time depends on the
processor and numerical-library configuration.

For an exploratory execution check, set the notebook's `PERMUTATIONS = 2` and
`OUTPUT_NAME = "results-quick"`. This retains the analysis stages but changes
the permutation null distribution and its P value. It must not be presented as
a reproduction of the paper's 100-permutation analysis.

## Run from the command line

The same reusable implementation is available through the installed command:

```bash
hnsc-methylation verify --repo-root .
hnsc-methylation run --repo-root . --output results --permutations 100
```

To perform the exploratory execution check separately:

```bash
hnsc-methylation run --repo-root . --output results-quick --permutations 2
```

Input verification checks the bundled files against their recorded identities.
An input mismatch should be resolved before interpreting results. Compressed
inputs are restored automatically to `data/prepared/` for computation. No
personal filesystem paths are required.

## Use the functions directly

```python
from pathlib import Path
from hnsc_methylation.pipeline import PipelineConfig, MethylationPipeline

root = Path.cwd().resolve()
pipeline = MethylationPipeline(
    PipelineConfig(repo_root=root, output_dir=root / "results", permutations=100)
)
pipeline.verify_inputs()
cohort = pipeline.prepare_cohort()
pipeline.annotate_probes()
pipeline.differential_methylation()
pipeline.expression_associations()
pipeline.classification()
pipeline.survival_models()
pipeline.robustness_checks()
pipeline.figures()
validation = pipeline.validate_results()
```

Run the stages in this order. Cohort preparation starts with the unfiltered
measurement matrix; the annotation stage applies the outcome-independent
probe rule before differential modelling. Each subsequent stage uses the
current run's inputs and results. The notebook contains presentation code;
the scientific implementations live in `src/hnsc_methylation/`.

## What is recomputed

| Analysis component | Treatment in this repository |
| --- | --- |
| Clinical cohort and patient matching | Recomputed from bundled processed measurements and clinical records |
| Probe eligibility | Recomputed from frozen HM27/HM450 manifests and quality/SNP masks |
| Differential methylation | Observed M-values, clinical adjustment, HC3 standard errors and Benjamini–Hochberg correction |
| PCA and cohort descriptions | Recomputed using the eligible probes and matched patients |
| Group classification | Five-fold out-of-fold prediction, training-fold preprocessing, feature-count sensitivity and label permutations |
| Methylation–expression association | Pairwise-complete Spearman correlations for exact CpG–gene pairs, with correction by analysis group |
| Candidate survival models | Observed-value, covariate-adjusted Cox main-effect and interaction models for reportable DMPs |
| Robustness | Numerical HC3/FDR checks, HPV-negative models, separate completeness and collection-centre sensitivities |
| Broader survival-screen summaries | Frozen imported manuscript data; the original complete screen is not rerun |
| RNA-expression KEGG GSEA | Frozen imported manuscript data; no new enrichment calculation is performed |
| Manuscript figures | Generated from current patient-level results and the explicitly identified imported summaries |

The broader survival screen used a separate imputed, unmasked analysis and a
cohort-mean high/low threshold. Its gene–annotation entries must not be
described as a new count of unique CpGs. Patient-level illustrated survival
curves are recalculated from the bundled measurements. The saved GSEA table
is RNA-expression based, and its absolute signal-to-noise ranking means a
positive NES does not identify an AFR or EUR direction. Keeping these imported
summaries explicit preserves the scope of the manuscript analysis.

Raw-array background correction and normalisation are outside this
processed-matrix workflow. Raw IDAT files and germline sequencing data are not
inputs, and the pipeline does not infer genetic ancestry.

## Repository contents

```text
hnsc-methylation-python/
├── notebooks/HNSC_Methylation_Pipeline.ipynb
├── src/hnsc_methylation/       Reusable analysis and command-line implementation
├── data/                     Bundled compressed inputs and provenance
│   ├── inputs/               Processed methylation, clinical and expression inputs
│   ├── source/               Original processed methylation and TCGA source notice
│   ├── probe_annotations/    Frozen platform manifests and quality/SNP masks
│   ├── reference/            Imported GSEA and broader survival-screen summaries
│   ├── reference_results/    Frozen manuscript tables and figure source data
│   └── prepared/             Automatically restored working inputs (generated)
├── tools/                    Data preparation and notebook-generation utilities
├── results/                  Recomputed outputs (generated)
├── pyproject.toml            Package metadata and dependency specifications
├── requirements.txt          Editable package plus notebook installation
├── CITATION.cff
└── LICENSE
```

The generated output folder contains `tables/`, `figures/` and figure source
data, together with run validation records. `results/`, `results-quick/` and
`data/prepared/` are ignored by Git. Bundled inputs and the frozen baseline in
`data/reference_results/` remain available for independent inspection and
comparison. Choose a new output directory when preserving an earlier run is
important.

The notebook can be regenerated from `tools/build_notebook.py` using
`python tools/build_notebook.py`. Regeneration replaces the notebook with its
unexecuted source form; preserve any notebook edits or outputs first.

## Reproducibility and software versions

The manuscript's numerical reference results were produced with Python
3.14.6, NumPy 2.5.1, pandas 2.3.3, SciPy 1.18.0, scikit-learn 1.8.0,
lifelines 0.30.0, Matplotlib 3.11.0 and seaborn 0.13.2. The package specifications
allow compatible releases on Python 3.11 and newer; this is an installation
compatibility target, not a claim that every supported Python/dependency
combination has been tested. openpyxl is included for spreadsheet input
support. Jupyter dependencies are optional for command-line users.

The fixed analysis seed is 42. Variation between compatible numerical-library
versions can affect floating-point details. Evaluate the run's validation
report and full result tables rather than relying only on rounded figure
labels. A match to the saved manuscript outputs establishes computational
agreement for this dataset; it does not constitute validation in an independent
patient cohort.

## Data sources, citation and reuse

The [data guide](data/README.md) and [provenance manifest](data/PROVENANCE.json)
identify the bundled files, checksums and source attribution. The patient data
are processed TCGA HNSC measurements and clinical
records. TCGA data resources include the
[NCI Genomic Data Commons](https://portal.gdc.cancer.gov/) and
[cBioPortal](https://www.cbioportal.org/). The probe annotation files come from
[InfiniumAnnotationData](https://github.com/zhou-lab/InfiniumAnnotationData),
frozen at commit `6d176d6e7e56465403d4478a09fe11b983eeb721`.

Please cite this software using [CITATION.cff](CITATION.cff), as well as the
underlying resources when appropriate:

- Cancer Genome Atlas Network. *Comprehensive genomic characterization of
  head and neck squamous cell carcinomas.* Nature. 2015;517:576–582.
  [PMID: 25631445](https://pubmed.ncbi.nlm.nih.gov/25631445/).
- Zhou W, Laird PW, Shen H. *Comprehensive characterization, annotation and
  innovative use of Infinium DNA methylation BeadChip probes.* Nucleic Acids
  Research. 2017;45:e22.
  [PMID: 27924034](https://pubmed.ncbi.nlm.nih.gov/27924034/).

Code and repository documentation are licensed under the [MIT License](LICENSE),
copyright 2026 Musalula Sinkala. Third-party datasets and annotations retain
their original source terms and attribution. The code license does not
relicense those data.
