#!/usr/bin/env python3
"""Build the 77-CpG survival summary and patient-level Kaplan-Meier figures.

The first output shows median-survival summaries from the 77 gene-annotation
rows retained by the broader survival screen. The second shows FZR1 and SLC6A20 Kaplan-Meier estimates
from matched patient-level methylation and survival records. The figures are
interpreted alongside the multiplicity-controlled Cox models in Figure 5.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test,logrank_test

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


AFR = "AFR"
EUR = "EUR"
AFR_COLOR = "#D55E00"
EUR_COLOR = "#0072B2"
AFR_LIGHT = "#E69F00"
EUR_LIGHT = "#56B4E9"
INK = "#1E2A32"
MID = "#6B7280"


def benjamini_hochberg(p_values: pd.Series) -> np.ndarray:
    p = pd.to_numeric(p_values, errors="raise").to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def prepare_survival_screen(screen: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Recheck BH correction for the fixed broad-screen manuscript data.

    This table contains the original broad survival screen; this function
    verifies its test universe and extracts candidates without rerunning or
    representing that screen as a new patient-level computation.
    """
    original_count = len(screen)
    complete = screen.drop_duplicates().copy()
    patterns = ['1stExon', '5_UTR', 'Body', 'TSS1500', 'TSS200']
    labels = ['1stExon', "5'UTR", 'Body', 'TSS1500', 'TSS200']
    complete['Location'] = np.select(
        [complete['GeneLocation'].astype(str).str.contains(p, regex=False) for p in patterns],
        labels, default='Other',
    )
    complete['recalculated_bh_q'] = benjamini_hochberg(complete['pValue'])
    complete['original_q'] = complete.adjPvalue
    complete['adjPvalue'] = complete.recalculated_bh_q
    candidates = complete[pd.to_numeric(complete['adjPvalue']) < 0.05].copy()
    summary = {
        'unique_tests': len(complete),
        'duplicate_rows_removed': original_count - len(complete),
        'candidates_q_lt_0_05': len(candidates),
        'analysis_status': 'fixed imported broad survival screen; BH verification and figure regeneration',
    }
    return complete, candidates, summary


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.7,
            "axes.labelsize": 8.6,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
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
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def make_candidate_summary_figure(candidate_screen: pd.DataFrame, figdir: Path, source_dir: Path) -> None:
    strata = ["AFR_high", "AFR_low", "EUR_high", "EUR_low"]
    labels = ["AFR high", "AFR low", "EUR high", "EUR low"]
    colors = [AFR_COLOR, AFR_LIGHT, EUR_COLOR, EUR_LIGHT]
    panels = [
        ("All candidates", None),
        ("First exon", "1stExon"),
        ("5' UTR", "5'UTR"),
        ("Gene body", "Body"),
        ("TSS1500", "TSS1500"),
        ("TSS200", "TSS200"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.7), constrained_layout=True, sharey=True)
    long_frames = []
    rng = np.random.default_rng(42)
    for idx, (ax, (title, location)) in enumerate(zip(axes.flat, panels)):
        d = candidate_screen.copy() if location is None else candidate_screen[candidate_screen.Location.eq(location)].copy()
        long = d.melt(
            id_vars=["GeneLocation", "Location", "pValue", "adjPvalue"],
            value_vars=strata,
            var_name="stratum",
            value_name="median_os_months",
        )
        long["panel"] = title
        long_frames.append(long)
        data = [pd.to_numeric(d[col], errors="coerce").dropna().to_numpy() for col in strata]
        bp = ax.boxplot(
            data,
            positions=np.arange(4),
            widths=0.54,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": INK, "linewidth": 1.1},
            whiskerprops={"color": "#84919B", "linewidth": 0.8},
            capprops={"color": "#84919B", "linewidth": 0.8},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.25)
        for x, values, color in zip(np.arange(4), data, colors):
            jitter = rng.normal(0, 0.065, size=len(values))
            ax.scatter(np.full(len(values), x) + jitter, values, s=13, color=color, alpha=0.72, edgecolor="white", linewidth=0.25)
        ax.set_xticks(np.arange(4), labels, rotation=23, ha="right")
        ax.set_title(f"{title} (n={len(d)})", loc="left")
        ax.set_ylim(0, 165)
        if idx % 3 == 0:
            ax.set_ylabel("Median overall survival (months)")
        sns.despine(ax=ax)
        panel_label(ax, chr(ord("a") + idx))
    fig.suptitle(
        f"Median overall survival across {len(candidate_screen)} survival-screen candidates",
        x=0.01,
        y=1.015,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.99,
        -0.01,
        "High methylation > cohort mean; low methylation <= cohort mean. Each point represents a selected gene–annotation test.",
        ha="right",
        color=MID,
        fontsize=7.2,
    )
    save_figure(fig, figdir / "figure6_survival_patterns_across_77_cpg_candidates")
    pd.concat(long_frames, ignore_index=True).to_csv(
        source_dir / "figure6_candidate_screen_survival_medians.tsv", sep="\t", index=False
    )


def make_candidate_km_figure(
    candidate_screen: pd.DataFrame,
    clinical: pd.DataFrame,
    matrix: pd.DataFrame,
    figdir: Path,
    source_dir: Path,
) -> None:
    targets = [
        ("FZR1", "TSS1500", "cg13183539"),
        ("SLC6A20", "TSS200", "cg04970117"),
    ]
    strata_order = ["AFR high", "AFR low", "EUR high", "EUR low"]
    styles = {
        "AFR high": (AFR_COLOR, "-"),
        "AFR low": (AFR_LIGHT, "--"),
        "EUR high": (EUR_COLOR, "-"),
        "EUR low": (EUR_LIGHT, "--"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.6))
    fig.subplots_adjust(left=.07,right=.98,bottom=.33,top=.89,wspace=.27)
    patient_sources = []
    summaries = []
    for idx, (ax, (gene, location, probe)) in enumerate(zip(axes, targets)):
        row = matrix[matrix.Entity_Stable_Id.eq(probe)].drop_duplicates("Entity_Stable_Id").iloc[0]
        d = clinical.copy()
        d["beta"] = pd.to_numeric(d.matrix_sample.map(row), errors="coerce")
        d = d[(d.os_months > 0) & d.os_event.notna() & d.beta.notna()].copy()
        split = float(d.beta.mean())
        d["methylation_level"] = np.where(d.beta > split, "high", "low")
        d["stratum"] = d.group + " " + d.methylation_level
        d["gene"] = gene
        d["location"] = location
        d["probe"] = probe
        d["global_beta_mean"] = split
        patient_sources.append(
            d[["matrix_sample", "patient_id", "group", "os_months", "os_event", "beta", "methylation_level", "stratum", "gene", "location", "probe", "global_beta_mean"]]
        )
        test = multivariate_logrank_test(d.os_months, d.stratum, d.os_event)
        fitters=[]
        for stratum in strata_order:
            x = d[d.stratum.eq(stratum)]
            km = KaplanMeierFitter()
            km.fit(x.os_months, x.os_event, label=f"{stratum} (n={len(x)})")
            fitters.append(km)
            color, linestyle = styles[stratum]
            km.plot_survival_function(
                ax=ax,
                ci_show=False,
                color=color,
                ls=linestyle,
                lw=1.9,
                censor_styles={"ms": 3.2, "mew": 0.8},
            )
            summaries.append(
                {
                    "gene": gene,
                    "location": location,
                    "probe": probe,
                    "stratum": stratum,
                    "n": len(x),
                    "events": int(x.os_event.sum()),
                    "median_os_months": float(km.median_survival_time_),
                    "global_beta_mean": split,
                    "four_group_logrank_p": float(test.p_value),
                }
            )
        candidate_row = candidate_screen[candidate_screen.GeneLocation.eq(f"{gene}_{location}")].iloc[0]
        ax.text(
            0.97,
            0.96,
            f"Four-group log-rank P = {test.p_value:.2g}\nCohort mean beta = {split:.3f}\nScreen BH q = {candidate_row.adjPvalue:.2g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.2,
            color=INK,
        )
        ax.set(xlabel="Overall survival (months)", ylabel="Survival probability", ylim=(0, 1.03))
        ax.set_xlim(0,120)
        ax.set_xticks([0,24,48,72,96,120])
        ax.text(-.06,-.25,'At risk',transform=ax.transAxes,ha='right',fontsize=7.2,fontweight='bold')
        for ri,stratum in enumerate(strata_order):
            subset=d[d.stratum.eq(stratum)]
            ypos=-.33-.075*ri
            ax.text(-.06,ypos,stratum,transform=ax.transAxes,ha='right',fontsize=7.2,color=styles[stratum][0])
            for t in [0,24,48,72,96,120]:
                ax.text(t/120,ypos,str(int((subset.os_months>=t).sum())),transform=ax.transAxes,ha='center',fontsize=7.2)
        ax.set_title(f"{gene} {location} | {probe}", loc="left")
        ax.legend(frameon=False, fontsize=6.7, loc="lower left")
        sns.despine(ax=ax)
        ax.text(-0.12, 1.02, chr(ord("a") + idx), transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    fig.suptitle(
        "Patient-level survival stratification for FZR1 and SLC6A20",
        x=0.01,
        y=.99,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    save_figure(fig, figdir / "figure7_fzr1_slc6a20_survival_stratification")
    pd.concat(patient_sources, ignore_index=True).to_csv(
        source_dir / "figure7_km_patient_data.tsv", sep="\t", index=False
    )
    summary = pd.DataFrame(summaries)
    candidate_values = candidate_screen[candidate_screen.GeneLocation.isin(["FZR1_TSS1500", "SLC6A20_TSS200"])][
        ["GeneLocation", "AFR_high", "AFR_low", "EUR_high", "EUR_low", "pValue", "adjPvalue"]
    ].copy()
    summary.to_csv(source_dir / "figure7_km_summary.tsv", sep="\t", index=False)
    candidate_values.to_csv(source_dir / "figure7_candidate_screen_summary.tsv", sep="\t", index=False)
