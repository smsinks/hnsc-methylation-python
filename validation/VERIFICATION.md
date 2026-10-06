# Package verification — 6 October 2026

Editor and verification record author: Musalula Sinkala.

The Jupyter notebook was executed from its first cell to its last cell with
the manuscript setting of 100 classifier label permutations. All 11 code
cells completed without errors. All 41 manuscript comparison checks passed;
the complete numerical record is in `reproduction_report.json`.

The regenerated patient-level results reproduce 495 patients (47 AFR and
448 EUR), 21,791 quality-eligible probes, 93 probes at BH q < 0.05 and 17
probes meeting both the q and effect-size criteria. Exact CpG–gene testing
uses 487 matched patients and reproduces 2,014 full-cohort associations.
Primary classifier ROC AUC is 0.9080072188449848; the completeness-restricted
and separate collection-centre sensitivities reproduce the manuscript.

Full differential-methylation, expression and candidate-survival vectors
were compared by their probe and gene identities with frozen reference
tables, not merely by selected rounded summary numbers. Maximum absolute
differences were approximately 1e-16 for differential statistics, 4.4e-15
for expression q values and 2.2e-16 for candidate survival hazard ratios.

The run generated eight main figures and three supplementary figures in
PNG, PDF and SVG formats. The executed notebook displays the figure set.
The broader survival-screen and RNA GSEA tables remain explicitly imported
manuscript summary inputs. Patient-level survival illustrations are
recomputed in Python. No MATLAB source, runtime or function is included.

Data verification checks 80 bundled files and 542,539,905 original input
bytes, including exact restoration from ordered gzip parts. Each distributed
data file is smaller than 50 MiB. The data manifest records source attribution,
dimensions, hashes and transformations.

Three targeted tests pass: BH correction against SciPy with non-finite tests,
training-filter behavior on held-out data, and output-path/input-protection
contracts. The package also builds successfully as an installable Python wheel.

One critic–defender–adjudicator review cycle identified a single supported
proposal: make the standalone `verify` command check the full distributed
package before reusing prepared input files. The defender supported the
proposal; the adjudicator accepted the exact narrow correction. The editor
changed only that command branch. Its verification run reported:

```text
Verified 80 bundled files and prepared 5 source inputs.
```

No additional scientific method, threshold, interpretation or input-data
change was introduced by the review. The orchestrator's final check confirmed
implementation of the accepted decision and preservation of unaffected code.

Notebook author, editor and last-modified-by metadata all identify Musalula
Sinkala. The software version is 0.1.0. These checks establish computational
agreement for the bundled dataset; independent-cohort validation is a separate
scientific question.
