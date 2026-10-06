"""Ordered, callable stages for the manuscript notebook and command line."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import json
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import core, expression, figures as plotting, probe_quality, robustness, survival, survival_figures


@dataclass(frozen=True)
class PipelineConfig:
    repo_root: Path
    output_dir: Path | None = None
    permutations: int = 100

    def __post_init__(self):
        root = Path(self.repo_root).resolve()
        object.__setattr__(self, "repo_root", root)
        output = Path(self.output_dir) if self.output_dir is not None else root / "results"
        if not output.is_absolute():
            output = root / output
        object.__setattr__(self, "output_dir", output.resolve())
        if self.permutations < 1:
            raise ValueError("Use at least one permutation; the manuscript setting is 100.")
        if self.output_dir == root or root / "data" == self.output_dir or root / "data" in self.output_dir.parents:
            raise ValueError("Output must be a separate directory outside the bundled data.")


def _data_tools(root: Path):
    path = root / "tools" / "prepare_data.py"
    spec = importlib.util.spec_from_file_location("_hnsc_data_preparation", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Data preparation module missing: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


class MethylationPipeline:
    """Run the reviewed models from bundled source data in dependency order.

    Primary statistics are recomputed. The manuscript's broader survival-screen
    and RNA GSEA tables are explicit, fixed inputs, used for their figures.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.root = config.repo_root
        self.output = config.output_dir
        self.tables = self.output / "tables"
        self.source_data = self.output / "source_data"
        self.figdir = self.output / "figures"
        self.completed: set[str] = set()
        self.timings: dict[str, float] = {}
        for path in [self.tables, self.source_data, self.figdir, self.output / "validation"]:
            path.mkdir(parents=True, exist_ok=True)

    def _require(self, *stages):
        missing = [s for s in stages if s not in self.completed]
        if missing:
            raise RuntimeError("Run these prerequisite stages first: " + ", ".join(missing))

    def _write(self, frame: pd.DataFrame, name: str, directory: Path | None = None):
        path = (directory or self.tables) / name
        frame.to_csv(path, sep="," if path.suffix == ".csv" else "\t", index=False)

    def _json(self, value, name, directory=None):
        ((directory or self.tables) / name).write_text(
            json.dumps(value, indent=2, default=_json_value) + "\n", encoding="utf-8"
        )

    def _finish(self, name, start, result):
        self.completed.add(name)
        self.timings[name] = round(time.monotonic() - start, 3)
        print(f"{name}: complete ({self.timings[name]:.1f} s)", flush=True)
        return result

    def verify_inputs(self):
        start = time.monotonic()
        tools = _data_tools(self.root)
        integrity = tools.verify_package(self.root)
        self.inputs = tools.materialize_inputs(self.root)
        for name in ["survival_screen.tsv.gz", "gsea_kegg_pathways.tsv.gz"]:
            if not (self.root / "data" / "reference" / name).is_file():
                raise FileNotFoundError(name)
        result = {"integrity": integrity, "prepared_inputs": {
            key: str(Path(path).relative_to(self.root)) for key, path in self.inputs.items()
        }}
        return self._finish("verify_inputs", start, result)

    def prepare_cohort(self):
        self._require("verify_inputs")
        start = time.monotonic()
        self.matrix = pd.read_csv(self.inputs["methylation_clean"], low_memory=False)
        self.clinical_raw = pd.read_csv(self.inputs["clinical_patient"], sep="\t", comment="#")
        metadata = ["Entity_Stable_Id", "HugoSymbol", "Location"]
        if not set(metadata).issubset(self.matrix):
            raise ValueError("Clean methylation matrix lacks the required probe annotations.")
        sample_cols = [c for c in self.matrix if c not in metadata]
        self.clinical = core.prepare_clinical(self.clinical_raw, sample_cols)
        samples = [c for c in sample_cols if c in set(self.clinical.matrix_sample)]
        self.clinical = self.clinical.set_index("matrix_sample").loc[samples].reset_index()
        if not self.clinical.patient_id.is_unique:
            raise ValueError("The cohort must contain one primary tumour per patient.")
        if any(c.split("_")[3][:2] != "01" for c in samples):
            raise ValueError("The manuscript cohort contains primary tumour samples only.")
        measured = self.matrix[samples].to_numpy(float)
        if np.any((measured[np.isfinite(measured)] < 0) | (measured[np.isfinite(measured)] > 1)):
            raise ValueError("Beta values must lie between zero and one.")
        self.balance = core.clinical_balance_table(self.clinical)
        self._write(self.clinical, "clinical_matched.csv")
        self._write(self.balance, "clinical_balance.tsv")
        fields = expression.clinical_source_fields(self.clinical_raw, self.clinical)
        self._write(fields, "clinical_source_fields.csv")
        return self._finish("prepare_cohort", start, self.clinical)

    def annotate_probes(self):
        self._require("prepare_cohort")
        start = time.monotonic()
        self.quality, provenance = probe_quality.annotate_probe_quality(
            self.matrix, self.root / "data" / "probe_annotations"
        )
        provenance["study_probe_universe_source"] = "data/PROVENANCE.json:methylation_clean"
        self.meta, self.beta, self.clinical, self.samples = core.prepare_methylation(
            self.matrix, self.clinical_raw, self.quality
        )
        self._write(self.quality, "probe_quality_annotations.tsv")
        self._json(provenance, "probe_quality_provenance.json")
        return self._finish("annotate_probes", start, pd.DataFrame([provenance["counts"]]))

    def differential_methylation(self):
        self._require("annotate_probes")
        start = time.monotonic()
        with threadpool_limits(limits=1):
            self.diff = core.differential_analysis(self.meta, self.beta, self.clinical)
            self.pca, self.pca_model = core.pca_analysis(self.beta, self.samples, self.clinical)
        self._write(self.diff, "differential_methylation_all_probes.tsv")
        self._write(self.pca, "pca_scores.tsv")
        return self._finish("differential_methylation", start, self.diff[self.diff.reportable_DMP])

    def expression_associations(self):
        self._require("annotate_probes")
        start = time.monotonic()
        self.corr, self.paired, notes = expression.expression_analysis(
            self.matrix, self.inputs["expression"], self.clinical, set(self.meta.Entity_Stable_Id)
        )
        self._write(self.corr, "methylation_expression_correlations.tsv")
        self._write(self.paired, "expression_matched_patients.csv")
        self._json(notes, "exact_expression_and_clinical_provenance.json")
        return self._finish("expression_associations", start, self.corr.query("Group == 'All' and regulatory_signal"))

    def classification(self):
        self._require("annotate_probes")
        start = time.monotonic()
        with threadpool_limits(limits=1):
            self.oof, self.feature_sensitivity, self.null, self.metrics = core.classifier_analysis(
                self.beta, self.samples, self.clinical, permutations=self.config.permutations
            )
        self._write(self.oof, "classifier_oof_predictions.tsv")
        self._write(self.feature_sensitivity, "classifier_feature_sensitivity.tsv")
        self._write(self.null, "classifier_permutation_null.tsv")
        self._json(self.metrics, "classifier_metrics.json")
        return self._finish("classification", start, self.metrics)

    def survival_models(self):
        self._require("differential_methylation", "expression_associations")
        start = time.monotonic()
        ids = set(self.diff.loc[self.diff.reportable_DMP, "Entity_Stable_Id"])
        self.probe_beta = self.matrix[self.matrix.Entity_Stable_Id.isin(ids)].drop_duplicates("Entity_Stable_Id")[["Entity_Stable_Id"] + self.samples].copy()
        with threadpool_limits(limits=1):
            self.survival = survival.fit_candidate_models(self.clinical, self.diff, self.probe_beta)
            self.group_cox = survival.fit_group_cox(self.clinical)
        self.evidence = survival.integrated_evidence(self.diff, self.corr, self.survival)
        self._write(self.survival, "candidate_survival_models.tsv")
        self._write(self.evidence, "integrated_candidate_evidence.tsv")
        self._json(self.group_cox, "group_survival_model.json")
        raw_screen = pd.read_csv(self.root / "data/reference/survival_screen.tsv.gz", sep="\t")
        self.screen, self.screen_candidates, self.screen_summary = survival_figures.prepare_survival_screen(raw_screen)
        self._write(self.screen, "broader_survival_screen.tsv")
        self._json(self.screen_summary, "survival_screen_summary.json")
        self.gsea = pd.read_csv(self.root / "data/reference/gsea_kegg_pathways.tsv.gz", sep="\t").rename(
            columns={"NES_pValue": "p_NES", "NES_qValue": "q_NES"}
        )
        self._write(self.gsea, "gsea_kegg_pathways.tsv")
        return self._finish("survival_models", start, self.survival)

    def robustness_checks(self):
        self._require("differential_methylation")
        start = time.monotonic()
        with threadpool_limits(limits=1):
            checks, self.hpv_negative, joined, summary = robustness.differential_robustness(
                self.beta, self.clinical, self.diff
            )
            self.robust_metrics, self.robust_predictions, folds = robustness.classifier_robustness(
                self.beta, self.clinical
            )
        self._write(checks, "independent_HC3_checks.tsv")
        self._write(self.hpv_negative, "differential_methylation_hpv_negative.tsv")
        self._write(joined, "primary_candidate_robustness.tsv")
        self._write(self.robust_metrics, "classifier_robustness_metrics.tsv")
        self._write(folds, "classifier_robustness_folds.tsv")
        self._write(self.robust_predictions, "figure3_panel_e_robustness_predictions.tsv", self.source_data)
        self._json(summary, "robustness_summary.json")
        return self._finish("robustness_checks", start, self.robust_metrics)

    def figures(self):
        self._require("differential_methylation", "expression_associations", "classification", "survival_models", "robustness_checks")
        start = time.monotonic()
        core.set_style()
        pca_assoc = core.make_figure1(self.figdir, self.clinical, self.balance, self.pca, self.pca_model)
        core.make_figure2(self.figdir, self.diff, self.beta, self.samples, self.clinical)
        core.make_figure3(self.figdir, self.oof, self.feature_sensitivity, self.null, self.metrics,
                          robustness=self.robust_metrics, robustness_predictions=self.robust_predictions,
                          source_dir=self.source_data)
        self.corr_clean = core.make_figure4(self.figdir, self.corr)
        core.write_panel_sources(self.source_data, self.diff, self.balance, self.pca, pca_assoc,
                                 self.oof, self.feature_sensitivity, self.null, self.corr_clean)
        survival.set_style()
        fig5 = survival.make_figure5(self.figdir, self.clinical, self.survival, self.probe_beta, self.group_cox)
        top, cross = survival.make_figure8(self.figdir, self.gsea, self.evidence)
        survival.make_supplementary(self.figdir, self.survival, self.clinical)
        self._write(self.survival, "figure5_candidate_survival_models.tsv", self.source_data)
        self._write(pd.DataFrame([self.group_cox]), "figure5_panel_a_group_cox.tsv", self.source_data)
        self._write(top, "figure8_panel_c_integrated_evidence.tsv", self.source_data)
        self._write(cross, "figure8_panel_d_methylation_expression_links.tsv", self.source_data)
        self._write(self.gsea, "figure8_panels_a_b_pathway_enrichment.tsv", self.source_data)
        self._write(self.screen[["GeneLocation", "pValue", "adjPvalue", "recalculated_bh_q"]],
                    "figure6_bh_verification_all_workbook_rows.tsv", self.source_data)
        survival_figures.set_style()
        survival_figures.make_candidate_summary_figure(self.screen_candidates, self.figdir, self.source_data)
        survival_figures.make_candidate_km_figure(self.screen_candidates, self.clinical, self.matrix, self.figdir, self.source_data)
        plotting.make_robustness_figures(self.figdir, self.source_data, self.diff, self.hpv_negative,
                                       self.corr_clean, self.clinical)
        self._json({**self.group_cox, **fig5,
                    "candidate_probes_tested": len(self.survival),
                    "candidate_survival_FDR_lt_0_05": int(self.survival.q_cox.lt(.05).sum()),
                    "candidate_interaction_FDR_lt_0_05": int(self.survival.q_interaction.lt(.05).sum())},
                   "survival_summary.json")
        return self._finish("figures", start, [p.name for p in sorted(self.figdir.glob("*.png"))])

    def validate_results(self):
        self._require("figures")
        start = time.monotonic()
        from .validation import compare_to_manuscript
        report = compare_to_manuscript(self)
        self._json(report, "reproduction_report.json", self.output / "validation")
        self._json({"permutations": self.config.permutations, "random_seed": 42,
                    "editor": "Musalula Sinkala", "stages_seconds": self.timings,
                    "reference_summary_inputs": ["data/reference/survival_screen.tsv.gz", "data/reference/gsea_kegg_pathways.tsv.gz"]},
                   "run_manifest.json", self.output / "validation")
        if not report["passed"]:
            raise AssertionError("Reproduction checks failed. See results/validation/reproduction_report.json")
        return self._finish("validate_results", start, pd.DataFrame(report["checks"]))

    def run(self):
        self.verify_inputs()
        self.prepare_cohort()
        self.annotate_probes()
        self.differential_methylation()
        self.expression_associations()
        self.classification()
        self.survival_models()
        self.robustness_checks()
        self.figures()
        return self.validate_results()
