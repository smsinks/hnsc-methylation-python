#!/usr/bin/env python3
"""Candidate survival analysis plus Figures 5 and 8 for the HNSC study."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, proportional_hazard_test

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


SEED = 42
AFR = "AFR"
EUR = "EUR"
AFR_COLOR = "#D55E00"
EUR_COLOR = "#0072B2"
PURPLE = "#6A51A3"
TEAL = "#009E73"
GOLD = "#E69F00"
INK = "#1E2A32"
MID = "#6B7280"

from . import core as rebuild


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 10.3,
            "axes.labelsize": 8.6,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#3F4A52",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    sns.set_style("whitegrid", {"grid.color": "#E9EDF1", "grid.linewidth": 0.6})


def panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.07, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    vals = p[ok]
    if not vals.size:
        return out
    order = np.argsort(vals)
    ranked = vals[order]
    q = ranked * len(vals) / np.arange(1, len(vals) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    restored = np.empty_like(q)
    restored[order] = np.clip(q, 0, 1)
    out[ok] = restored
    return out


def build_cox_frame(clin: pd.DataFrame, beta: pd.Series) -> pd.DataFrame:
    d = clin.copy()
    values=d.matrix_sample.map(beta)
    d["beta_per_0_1"] = (values-values.mean()) * 10.0
    d = d[(d.os_months > 0) & d.os_event.notna() & d.beta_per_0_1.notna()].copy()
    d["age"] = d.age.fillna(d.age.median())
    d["age_z"] = (d.age - d.age.mean()) / d.age.std(ddof=0)
    d["AFR"] = (d.group == AFR).astype(int)
    stages = pd.get_dummies(d.stage, prefix="stage", dtype=float)
    for name in ["stage_II", "stage_III", "stage_IV", "stage_Unknown"]:
        if name not in stages:
            stages[name] = 0.0
    design=rebuild.clinical_design(d).drop(columns='intercept')
    out=pd.concat([d[['os_months','os_event','beta_per_0_1']],design],axis=1)
    return out.astype(float)


def fit_candidate_models(clin: pd.DataFrame, diff: pd.DataFrame, probe_beta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidates = diff[diff.reportable_DMP.astype(str).str.lower().isin(["true", "1"])].copy()
    if candidates.empty:
        raise ValueError('No candidates met the primary criteria; do not change the threshold after seeing results.')
    beta_by_probe = probe_beta.set_index("Entity_Stable_Id")
    for _, candidate in candidates.iterrows():
        probe = candidate.Entity_Stable_Id
        if probe not in beta_by_probe.index:
            continue
        series = beta_by_probe.loc[probe]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[0]
        frame = build_cox_frame(clin, series)
        base = {
            "Entity_Stable_Id": probe,
            "HugoSymbol": candidate.HugoSymbol,
            "Location": candidate.Location,
            "delta_beta_AFR_minus_EUR": candidate.delta_beta_AFR_minus_EUR,
            "q_adjusted_methylation": candidate.q_adjusted,
            "n": len(frame),
            "events": int(frame.os_event.sum()),
        }
        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(frame, duration_col="os_months", event_col="os_event", show_progress=False)
            s = cph.summary.loc["beta_per_0_1"]
            ph = proportional_hazard_test(cph, frame, time_transform="rank").summary.loc["beta_per_0_1", "p"]
            base.update(
                {
                    "hr_per_0_1_beta": float(s["exp(coef)"]),
                    "ci_low": float(s["exp(coef) lower 95%"]),
                    "ci_high": float(s["exp(coef) upper 95%"]),
                    "p_cox": float(s["p"]),
                    "ph_test_p": float(ph),
                    "concordance": float(cph.concordance_index_),
                }
            )
        except Exception as exc:
            base.update({"hr_per_0_1_beta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_cox": np.nan, "ph_test_p": np.nan, "concordance": np.nan, "error": str(exc)})

        # Separate ancestry-by-methylation interaction model; this is an
        # exploratory heterogeneity screen, not the primary survival test.
        frame_i = frame.copy()
        frame_i["beta_x_AFR"] = frame_i.beta_per_0_1 * frame_i.AFR
        try:
            cphi = CoxPHFitter(penalizer=0.02)
            cphi.fit(frame_i, duration_col="os_months", event_col="os_event", show_progress=False)
            si = cphi.summary.loc["beta_x_AFR"]
            base.update(
                {
                    "interaction_hr": float(si["exp(coef)"]),
                    "interaction_ci_low": float(si["exp(coef) lower 95%"]),
                    "interaction_ci_high": float(si["exp(coef) upper 95%"]),
                    "p_interaction": float(si["p"]),
                }
            )
        except Exception as exc:
            base.update({"interaction_hr": np.nan, "interaction_ci_low": np.nan, "interaction_ci_high": np.nan, "p_interaction": np.nan, "interaction_error": str(exc)})
        rows.append(base)
    out = pd.DataFrame(rows)
    out["q_cox"] = bh_fdr(out.p_cox.to_numpy())
    out["q_interaction"] = bh_fdr(out.p_interaction.to_numpy())
    return out.sort_values(["q_cox", "p_cox"], na_position="last")


def fit_group_cox(clin: pd.DataFrame) -> dict:
    d = clin[(clin.os_months > 0) & clin.os_event.notna()].copy()
    d["age"] = d.age.fillna(d.age.median())
    d["age_z"] = (d.age - d.age.mean()) / d.age.std(ddof=0)
    d["AFR"] = (d.group == AFR).astype(int)
    stages = pd.get_dummies(d.stage, prefix="stage", dtype=float)
    for name in ["stage_II", "stage_III", "stage_IV", "stage_Unknown"]:
        if name not in stages:
            stages[name] = 0.0
    frame=pd.concat([d[['os_months','os_event']],rebuild.clinical_design(d).drop(columns='intercept')],axis=1).astype(float)
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(frame, "os_months", "os_event")
    s = cph.summary.loc["AFR"]
    return {
        "n": len(frame),
        "events": int(frame.os_event.sum()),
        "hr_AFR_vs_EUR": float(s["exp(coef)"]),
        "ci_low": float(s["exp(coef) lower 95%"]),
        "ci_high": float(s["exp(coef) upper 95%"]),
        "p": float(s["p"]),
        "concordance": float(cph.concordance_index_),
    }


def km_plot(ax, d: pd.DataFrame, group_col: str, order: list[str], colors: list[str], title: str, labels: dict[str, str] | None = None, linestyles: list[str] | None = None) -> float:
    kmf = KaplanMeierFitter()
    linestyles = linestyles or ["-"] * len(order)
    for group, color, ls in zip(order, colors, linestyles):
        x = d[d[group_col] == group]
        kmf.fit(x.os_months, x.os_event, label=labels.get(group, group) if labels else group)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color, lw=1.9, ls=ls, censor_styles={"ms": 3.5})
    if len(order) == 2:
        a = d[d[group_col] == order[0]]
        b = d[d[group_col] == order[1]]
        p = logrank_test(a.os_months, b.os_months, a.os_event, b.os_event).p_value
    else:
        p = np.nan
    ax.set(xlabel="Overall survival (months)", ylabel="Survival probability", ylim=(0, 1.03))
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, fontsize=7)
    sns.despine(ax=ax)
    return float(p)


def make_figure5(figdir: Path, clin: pd.DataFrame, survival: pd.DataFrame, probe_beta: pd.DataFrame, group_cox: dict) -> dict:
    top = survival.dropna(subset=["p_cox"]).iloc[0]
    probe = top.Entity_Stable_Id
    genes = top.HugoSymbol
    beta = probe_beta.set_index("Entity_Stable_Id").loc[probe]
    if isinstance(beta, pd.DataFrame):
        beta = beta.iloc[0]
    d = clin.copy()
    d["beta"] = d.matrix_sample.map(beta)
    d = d[(d.os_months > 0) & d.beta.notna()].copy()
    median = float(d.beta.median())
    d["methylation_group"] = np.where(d.beta >= median, "High", "Low")

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.6), constrained_layout=True)
    ax = axes[0, 0]
    all_survival=clin[(clin.os_months>0)&clin.os_event.notna()].copy()
    p_group = km_plot(ax, all_survival, "group", [EUR, AFR], [EUR_COLOR, AFR_COLOR], "Overall survival by clinical group")
    ax.text(0.97, 0.96, f"Log-rank P = {p_group:.3g}\nAdjusted HR = {group_cox['hr_AFR_vs_EUR']:.2f}\n95% CI {group_cox['ci_low']:.2f}-{group_cox['ci_high']:.2f}", transform=ax.transAxes, ha="right", va="top", color=INK)
    panel_label(ax, "a")

    ax = axes[0, 1]
    p_candidate = km_plot(ax, d, "methylation_group", ["Low", "High"], [TEAL, PURPLE], f"Top DMP-survival candidate: {genes} | {probe}", labels={"Low": "Low methylation", "High": "High methylation"})
    ax.text(0.97, 0.96, f"Median split beta = {median:.3f}\nLog-rank P = {p_candidate:.3g}\nCox FDR q = {top.q_cox:.3g}", transform=ax.transAxes, ha="right", va="top", color=INK)
    panel_label(ax, "b")

    ax = axes[1, 0]
    forest = survival.dropna(subset=["p_cox"]).nsmallest(12, "p_cox").copy().sort_values("hr_per_0_1_beta")
    yy = np.arange(len(forest))
    colors = [AFR_COLOR if hr > 1 else EUR_COLOR for hr in forest.hr_per_0_1_beta]
    ax.hlines(yy, forest.ci_low, forest.ci_high, color=colors, lw=1.4)
    ax.scatter(forest.hr_per_0_1_beta, yy, color=colors, s=34, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(1, color="#65717A", lw=0.8)
    ax.set_xscale("log")
    ax.set_yticks(yy, [f"{g} | {p}" for g, p in zip(forest.HugoSymbol, forest.Entity_Stable_Id)], fontsize=6.4)
    ax.set_xlabel("Adjusted hazard ratio per +0.10 beta")
    ax.set_title("Adjusted CpG survival estimates", loc="left")
    sns.despine(ax=ax)
    panel_label(ax, "c")

    ax = axes[1, 1]
    s = survival.dropna(subset=["p_interaction", "p_cox"]).copy()
    ax.scatter(np.log2(s.hr_per_0_1_beta), np.log2(s.interaction_hr), s=24 + 25 * np.clip(-np.log10(s.p_cox), 0, 4), c=np.where(s.q_interaction < 0.05, GOLD, "#9FB0BD"), alpha=0.72, edgecolor="white", linewidth=0.35)
    ax.axhline(0, color="#65717A", lw=0.8)
    ax.axvline(0, color="#65717A", lw=0.8)
    ax.set_xlabel("Main methylation effect, log2 HR")
    ax.set_ylabel("Methylation x AFR interaction, log2 HR")
    ax.set_title("Methylation effects and group interactions", loc="left")
    for _, r in s.nsmallest(4, "p_interaction").iterrows():
        ax.annotate(str(r.HugoSymbol).split("/")[0], (np.log2(r.hr_per_0_1_beta), np.log2(r.interaction_hr)), xytext=(3, 3), textcoords="offset points", fontsize=6)
    ax.text(0.98, 0.96, f"Interaction FDR < 0.05: {(s.q_interaction < 0.05).sum()} of {len(s)}", transform=ax.transAxes, va="top", ha="right", color=INK)
    sns.despine(ax=ax)
    panel_label(ax, "d")
    save_figure(fig, figdir / "figure5_survival_prioritisation")
    return {"top_probe": probe, "top_gene": genes, "candidate_median_beta": median, "candidate_logrank_p": p_candidate, "group_logrank_p": p_group}


def expand_diff_genes(diff: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in diff.iterrows():
        genes = [g for g in str(r.HugoSymbol).split("/") if g and g.lower() != "nan"]
        for g in genes:
            x = r.to_dict()
            x["gene"] = g
            rows.append(x)
    return pd.DataFrame(rows)


def integrated_evidence(diff: pd.DataFrame, corr: pd.DataFrame, survival: pd.DataFrame) -> pd.DataFrame:
    dg = expand_diff_genes(diff[diff.reportable_DMP.astype(str).str.lower().isin(["true", "1"])]).copy()
    dg["methyl_strength"] = -np.log10(np.maximum(dg.q_adjusted, 1e-300)) * dg.delta_beta_AFR_minus_EUR.abs()
    dm = dg.sort_values("methyl_strength", ascending=False).drop_duplicates("gene")[["gene", "delta_beta_AFR_minus_EUR", "q_adjusted", "methyl_strength", "Entity_Stable_Id"]]

    allc = corr[corr.Group == "All"].copy()
    allc["corr_strength"] = allc.spearman_r.abs() * -np.log10(np.maximum(allc.q_value, 1e-300))
    cm = allc[["Entity_Stable_Id","HugoSymbol", "spearman_r", "q_value", "corr_strength"]].rename(columns={"HugoSymbol": "gene"})

    sg = expand_diff_genes(survival.rename(columns={"HugoSymbol": "HugoSymbol"})).copy()
    sg["survival_strength"] = -np.log10(np.maximum(sg.p_cox, 1e-300))
    sm = sg[["Entity_Stable_Id","gene", "hr_per_0_1_beta", "p_cox", "q_cox", "survival_strength"]]
    out = dm.merge(cm, on=['Entity_Stable_Id','gene'], how="left",validate='one_to_one').merge(sm, on=['Entity_Stable_Id','gene'], how="left",validate='one_to_one')
    for col in ["methyl_strength", "corr_strength", "survival_strength"]:
        out[col + "_pct"] = out[col].rank(pct=True).fillna(0)
    out["evidence_layers"] = out[["methyl_strength", "corr_strength", "survival_strength"]].notna().sum(axis=1)
    out["integrated_score"] = out[["methyl_strength_pct", "corr_strength_pct", "survival_strength_pct"]].sum(axis=1) + 0.25 * out.evidence_layers
    return out.sort_values('integrated_score', ascending=False)


def make_figure8(figdir: Path, gsea: pd.DataFrame, evidence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.7), constrained_layout=True)
    ax = axes[0, 0]
    top = gsea.nsmallest(14, "p_NES").copy().sort_values("NES")
    sizes = 28 + 45 * np.clip(-np.log10(np.maximum(top.p_NES, 1e-300)), 0, 4)
    ax.scatter(top.NES, np.arange(len(top)), s=sizes, c=-np.log10(np.maximum(top.q_NES, 1e-300)), cmap="viridis", vmin=0, vmax=max(1, -np.log10(np.maximum(gsea.q_NES, 1e-300)).max()), edgecolor="white", linewidth=0.5)
    ax.set_yticks(np.arange(len(top)), top.Term, fontsize=6.4)
    ax.set_xlabel("Normalized enrichment score")
    ax.set_title("Leading RNA expression gene sets", loc="left")
    ax.text(0.98, 0.04, "Point size: -log10 nominal P\nColour: -log10 pathway FDR", transform=ax.transAxes, ha="right", va="bottom", color=MID, fontsize=6.7)
    sns.despine(ax=ax)
    panel_label(ax, "a")

    ax = axes[0, 1]
    y = -np.log10(np.maximum(gsea.p_NES, 1e-300))
    sig = gsea.q_NES < 0.25
    ax.scatter(gsea.NES[~sig], y[~sig], s=15, c="#A9B4BD", alpha=0.65, edgecolor="none")
    if sig.any():
        ax.scatter(gsea.NES[sig], y[sig], s=23, c=GOLD, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axhline(-np.log10(0.05), color="#6B747C", ls="--", lw=0.8)
    ax.set(xlabel="Normalized enrichment score", ylabel="-log10 nominal P")
    ax.set_title("RNA expression enrichment across 290 gene sets", loc="left")
    ax.text(0.04, 0.94, f"Pathways at FDR < 0.25: {sig.sum()} of {len(gsea)}", transform=ax.transAxes, va="top", color=INK)
    sns.despine(ax=ax)
    panel_label(ax, "b")

    ax = axes[1, 0]
    top_e = evidence.head(16).copy().sort_values("integrated_score")
    layers = [
        ("Methylation", "methyl_strength_pct", "delta_beta_AFR_minus_EUR"),
        ("Expression", "corr_strength_pct", "spearman_r"),
        ("Survival", "survival_strength_pct", "hr_per_0_1_beta"),
    ]
    for xi, (label, strength, value) in enumerate(layers):
        vals = top_e[value]
        if value == "hr_per_0_1_beta":
            signed = np.log2(vals)
        else:
            signed = vals
        ax.scatter(np.repeat(xi, len(top_e)), np.arange(len(top_e)), s=20 + 120 * top_e[strength].fillna(0), c=signed, cmap="vlag", vmin=-max(0.8, np.nanmax(np.abs(signed))), vmax=max(0.8, np.nanmax(np.abs(signed))), edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(3), [x[0] for x in layers])
    ax.set_yticks(np.arange(len(top_e)), top_e.gene, fontsize=6.6)
    ax.set_xlim(-0.55, 2.55)
    ax.set_title("Evidence for differential-methylation candidates", loc="left")
    ax.text(0.01, -0.13, "Point area: within-layer evidence percentile; colour: signed effect (survival uses log2 HR).", transform=ax.transAxes, color=MID, fontsize=6.8)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=True)
    panel_label(ax, "c")

    ax = axes[1, 1]
    cross_layer = evidence.dropna(subset=["delta_beta_AFR_minus_EUR", "spearman_r"]).copy()
    cross_layer["combined_strength"] = (
        -np.log10(np.maximum(cross_layer.q_adjusted, 1e-300))
        + -np.log10(np.maximum(cross_layer.q_value, 1e-300))
    )
    point_sizes = 28 + 125 * cross_layer.combined_strength.rank(pct=True)
    points = ax.scatter(
        cross_layer.delta_beta_AFR_minus_EUR,
        cross_layer.spearman_r,
        s=point_sizes,
        c=cross_layer.delta_beta_AFR_minus_EUR,
        cmap="vlag",
        vmin=-0.22,
        vmax=0.22,
        edgecolor="white",
        linewidth=0.55,
        alpha=0.84,
    )
    ax.axhline(0, color="#65717A", lw=0.8)
    ax.axvline(0, color="#65717A", lw=0.8)
    ax.axhspan(-0.30, 0.30, color="#F4F7FA", zorder=0)
    ax.set_xlabel("Delta beta (AFR - EUR)")
    ax.set_ylabel("Matched methylation-expression Spearman rho")
    ax.set_title("Exact CpG–gene expression associations", loc="left")
    for _, row in cross_layer.nlargest(min(7, len(cross_layer)), "combined_strength").iterrows():
        ax.annotate(
            str(row.gene),
            (row.delta_beta_AFR_minus_EUR, row.spearman_r),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.2,
            color=INK,
        )
    ax.text(
        0.03,
        0.96,
        f"DMP-linked genes with matched expression evidence: {len(cross_layer)}",
        transform=ax.transAxes,
        va="top",
        color=INK,
        fontsize=7,
    )
    sns.despine(ax=ax)
    panel_label(ax, "d")
    save_figure(fig, figdir / "figure8_pathway_and_evidence_synthesis")
    return top_e, cross_layer


def make_supplementary(figdir: Path, survival: pd.DataFrame, clin: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1), constrained_layout=True)
    ax = axes[0]
    s = survival.dropna(subset=["p_cox"])
    ax.hist(s.p_cox, bins=np.linspace(0, 1, 16), color="#A9B4BD", edgecolor="white")
    ax.axvline(0.05, color=PURPLE, ls="--", lw=1)
    ax.set(xlabel="Candidate Cox P value", ylabel="CpG probes", title="Candidate survival P-value distribution")
    sns.despine(ax=ax)
    panel_label(ax, "a")
    ax = axes[1]
    ax.scatter(-np.log10(np.maximum(s.p_cox, 1e-300)), -np.log10(np.maximum(s.q_cox, 1e-300)), c=np.where(s.q_cox < 0.05, GOLD, "#9FB0BD"), s=28, alpha=0.75)
    ax.axhline(-np.log10(0.05), color="#6B747C", ls="--", lw=0.8)
    ax.set(xlabel="-log10 nominal P", ylabel="-log10 candidate-set FDR", title="Survival multiplicity correction")
    sns.despine(ax=ax)
    panel_label(ax, "b")
    ax = axes[2]
    ax.scatter(-np.log10(np.maximum(s.p_cox, 1e-300)), -np.log10(np.maximum(s.ph_test_p, 1e-300)), c=np.where(s.ph_test_p < 0.05, AFR_COLOR, TEAL), s=30, alpha=0.7)
    ax.axhline(-np.log10(0.05), color="#6B747C", ls="--", lw=0.8)
    ax.set(xlabel="-log10 Cox P", ylabel="-log10 PH diagnostic P", title="Proportional-hazards diagnostics")
    sns.despine(ax=ax)
    panel_label(ax, "c")
    save_figure(fig, figdir / "supplementary_figureS3_survival_diagnostics")
