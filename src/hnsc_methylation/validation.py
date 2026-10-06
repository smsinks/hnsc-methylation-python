"""Compare regenerated statistics with the frozen manuscript result tables."""
from __future__ import annotations
import numpy as np
import pandas as pd


def compare_to_manuscript(pipeline):
    checks = []

    def check(name, passed, observed, expected):
        checks.append({"check": name, "passed": bool(passed), "observed": observed, "expected": expected})

    c, d, corr = pipeline.clinical, pipeline.diff, pipeline.corr
    expected_counts = {
        "patients": (len(c), 495),
        "AFR patients": (int(c.group.eq("AFR").sum()), 47),
        "EUR patients": (int(c.group.eq("EUR").sum()), 448),
        "unique input probes": (len(pipeline.quality), 22601),
        "eligible probes": (len(d), 21791),
        "BH-significant probes": (int(d.q_adjusted.lt(.05).sum()), 93),
        "differential probes meeting effect criterion": (int(d.reportable_DMP.sum()), 17),
        "RNA-matched patients": (len(pipeline.paired), 487),
        "finite full-cohort expression tests": (int(corr.loc[corr.Group.eq("All"), "p_value"].notna().sum()), 24224),
        "full-cohort expression associations": (int(corr.query("Group == 'All'").regulatory_signal.sum()), 2014),
        "primary DMPs retained in HPV-negative disease": (int(pipeline.diff[pipeline.diff.reportable_DMP].Entity_Stable_Id.isin(pipeline.hpv_negative.loc[pipeline.hpv_negative.reportable_DMP, "Entity_Stable_Id"]).sum()), 12),
        "candidate Cox models": (len(pipeline.survival), 17),
        "FDR-supported candidate survival effects": (int(pipeline.survival.q_cox.lt(.05).sum()), 0),
        "FDR-supported survival interactions": (int(pipeline.survival.q_interaction.lt(.05).sum()), 0),
        "broader survival tests": (len(pipeline.screen), 21838),
        "broader screen candidates": (len(pipeline.screen_candidates), 77),
        "RNA GSEA pathways": (len(pipeline.gsea), 290),
        "RNA GSEA pathways at q below 0.25": (int(pipeline.gsea.q_NES.lt(.25).sum()), 0),
        "PNG scientific figures": (len(list(pipeline.figdir.glob("*.png"))), 11),
    }
    for name, (actual, expected) in expected_counts.items():
        check(name, actual == expected, actual, expected)

    refs = pipeline.root / "data" / "reference_results" / "tables"
    def reference(name):
        path = refs / name
        if not path.exists():
            path = refs / (name + ".gz")
        return pd.read_csv(path, sep="\t")

    def compare_frame(name, actual, frozen, keys, columns, atol=1e-100, rtol=1e-6):
        left = actual.set_index(keys).sort_index()
        right = frozen.set_index(keys).sort_index()
        same_index = left.index.equals(right.index)
        check(name + " test identities", same_index, len(left), len(right))
        if not same_index:
            return
        for col in columns:
            a = left[col].to_numpy(float)
            b = right[col].to_numpy(float)
            equal = np.allclose(a, b, equal_nan=True, rtol=rtol, atol=atol)
            finite = np.isfinite(a) & np.isfinite(b)
            error = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
            check(name + ": " + col, equal, error, f"max difference within rtol={rtol:g}, atol={atol:g}")

    compare_frame("differential methylation", d, reference("differential_methylation_all_probes.tsv"),
                  ["Entity_Stable_Id"], ["delta_beta_AFR_minus_EUR", "adjusted_race_coefficient_M_value", "p_adjusted", "q_adjusted"])
    compare_frame("exact expression", corr, reference("methylation_expression_correlations.tsv"),
                  ["Group", "Entity_Stable_Id", "HugoSymbol"], ["spearman_r", "p_value", "q_value", "n"])
    compare_frame("candidate survival", pipeline.survival, reference("candidate_survival_models.tsv"),
                  ["Entity_Stable_Id"], ["hr_per_0_1_beta", "p_cox", "q_cox", "p_interaction", "q_interaction"], atol=1e-10)
    compare_frame("classifier sensitivities", pipeline.robust_metrics, reference("classifier_robustness_metrics.tsv"),
                  ["model"], ["roc_auc", "average_precision", "balanced_accuracy"], atol=1e-8)
    check("primary ROC AUC", np.isclose(pipeline.metrics["roc_auc"], .9080072188449848, atol=1e-8, rtol=0),
          pipeline.metrics["roc_auc"], .9080072188449848)
    if pipeline.config.permutations == 100:
        check("manuscript permutation P", np.isclose(pipeline.metrics["permutation_p_roc_auc"], 1 / 101),
              pipeline.metrics["permutation_p_roc_auc"], 1 / 101)
    return {"passed": all(row["passed"] for row in checks),
            "manuscript_permutation_setting": pipeline.config.permutations == 100,
            "permutations": pipeline.config.permutations,
            "checks": checks,
            "scope": "Primary models recomputed from bundled source data; broad survival-screen and GSEA summaries imported as documented fixed inputs."}
