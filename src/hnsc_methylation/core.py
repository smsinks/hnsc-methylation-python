#!/usr/bin/env python3
"""Rebuild the TCGA-HNSC AFR/EUR methylation analysis and Figures 1-4.

The script deliberately retains CpG probe identifiers, reports biologically
interpretable delta-beta values, uses M-values for inferential testing, adjusts
for available clinical covariates, and places supervised feature selection
inside cross-validation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
LIGHT = "#E5E7EB"


class TrainingVarianceFilter(BaseEstimator, TransformerMixin):
    """Fit the unsupervised variance filter on training samples only."""
    def __init__(self, k=5000):
        self.k = k

    def fit(self, X, y=None):
        self.indices_ = np.argsort(np.nanvar(X, axis=0))[-min(self.k, X.shape[1]):]
        return self

    def transform(self, X):
        return X[:, self.indices_]


def clinical_design(c, include_group=True):
    c = c.copy()
    c['age_z'] = (c.age-c.age.mean())/c.age.std(ddof=0)
    cols = {'intercept': np.ones(len(c))}
    if include_group:
        cols['AFR'] = c.group.eq(AFR).astype(float).to_numpy()
    for key in ['age_z','male','hpv_positive','hpv_unknown']:
        cols[key] = c[key].to_numpy(float)
    for stage in ['II','III','IV','Unknown']:
        cols['stage_'+stage] = c.stage.eq(stage).astype(float).to_numpy()
    for site in ['Oropharynx','Larynx','Hypopharynx','Other']:
        cols['site_'+site] = c.site.eq(site).astype(float).to_numpy()
    return pd.DataFrame(cols, index=c.index)


def anatomical_site(value):
    code = str(value).upper()
    if code.startswith(('C01','C09','C10')) or code in {'C02.4','C05.1','C05.2'}:
        return 'Oropharynx'
    if code.startswith('C32'):
        return 'Larynx'
    if code.startswith(('C12','C13')):
        return 'Hypopharynx'
    if code.startswith(('C00','C02','C03','C04','C05','C06')):
        return 'Oral cavity'
    return 'Other'


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 10.3,
            "axes.labelsize": 8.6,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#3F4A52",
            "axes.titleweight": "bold",
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
    ax.text(
        -0.12,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color="black",
        va="top",
        ha="left",
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    ok = np.isfinite(p)
    vals = p[ok]
    if vals.size == 0:
        return out
    order = np.argsort(vals)
    ranked = vals[order]
    q = ranked * vals.size / np.arange(1, vals.size + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    restored = np.empty_like(q)
    restored[order] = q
    out[ok] = restored
    return out


def patient_id_from_matrix(sample: str) -> str:
    fields = sample.split("_")
    if len(fields) >= 3 and fields[0] == "TCGA":
        return "-".join(fields[:3])
    return sample.replace("_", "-")


def clean_stage(value: object) -> str:
    text = str(value).upper()
    if "STAGE IV" in text:
        return "IV"
    if "STAGE III" in text:
        return "III"
    if "STAGE II" in text:
        return "II"
    if "STAGE I" in text:
        return "I"
    return "Unknown"


def binary_smd(p1: float, p0: float) -> float:
    denom = math.sqrt(max((p1 * (1 - p1) + p0 * (1 - p0)) / 2, 1e-12))
    return (p1 - p0) / denom


def prepare_clinical(clinical: pd.DataFrame, sample_cols: list[str]) -> pd.DataFrame:
    c = clinical.copy()
    c["patient_id"] = c["PATIENT_ID"].astype(str).str.slice(0, 12)
    c = c.drop_duplicates("patient_id").set_index("patient_id")
    records = []
    for sample in sample_cols:
        pid = patient_id_from_matrix(sample)
        if pid not in c.index:
            continue
        row = c.loc[pid]
        race = str(row.get("RACE", ""))
        if race not in {"Black or African American", "White"}:
            continue
        os_status = str(row.get("OS_STATUS", ""))
        records.append(
            {
                "matrix_sample": sample,
                "patient_id": pid,
                "group": AFR if race == "Black or African American" else EUR,
                "source_race_label": race,
                "age": pd.to_numeric(row.get("AGE"), errors="coerce"),
                "male": 1 if str(row.get("SEX", "")).lower() == "male" else 0,
                "stage": clean_stage(row.get("AJCC_PATHOLOGIC_TUMOR_STAGE")),
                "hpv_positive": 1 if str(row.get("SUBTYPE", "")).upper().endswith("HPV+") else 0,
                "hpv_unknown": int(str(row.get("SUBTYPE", "")).upper() not in {"HNSC_HPV+", "HNSC_HPV-"}),
                "site": anatomical_site(row.get("ICD_O_3_SITE")),
                "icd_o_3_site": row.get("ICD_O_3_SITE"),
                "radiation": row.get("RADIATION_THERAPY"),
                "os_months": pd.to_numeric(row.get("OS_MONTHS"), errors="coerce"),
                "os_event": 1 if os_status.startswith("1:") else 0,
                "pfs_months": pd.to_numeric(row.get("PFS_MONTHS"), errors="coerce"),
                "weight": pd.to_numeric(row.get("WEIGHT"), errors="coerce"),
            }
        )
    out = pd.DataFrame.from_records(records)
    out["stage_iv"] = (out["stage"] == "IV").astype(int)
    return out


def clinical_balance_table(clin: pd.DataFrame) -> pd.DataFrame:
    rows = []
    a = clin[clin.group == AFR]
    e = clin[clin.group == EUR]
    pooled_age_sd = np.sqrt((np.nanvar(a.age, ddof=1) + np.nanvar(e.age, ddof=1)) / 2)
    rows.append(
        {
            "feature": "Age at diagnosis",
            "AFR": np.nanmean(a.age),
            "EUR": np.nanmean(e.age),
            "smd": (np.nanmean(a.age) - np.nanmean(e.age)) / pooled_age_sd,
            "unit": "years",
        }
    )
    for col, label in [("male", "Male sex"), ("hpv_positive", "HPV-positive subtype"), ("stage_iv", "Stage IV")]:
        p1 = float(a[col].mean())
        p0 = float(e[col].mean())
        rows.append({"feature": label, "AFR": p1, "EUR": p0, "smd": binary_smd(p1, p0), "unit": "proportion"})
    return pd.DataFrame(rows)


def prepare_methylation(
    matrix: pd.DataFrame, clinical: pd.DataFrame, quality: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, list[str]]:
    """Match patients and retain quality-eligible unique physical probes.

    Repeated source rows carry multiple annotations for one measurement.
    Preserve their unique gene/location labels and keep the first measurement
    exactly as in the reviewed analysis.
    """
    meta_cols = ['Entity_Stable_Id', 'HugoSymbol', 'Location']
    sample_cols = [c for c in matrix.columns if c not in meta_cols]
    meta = (
        matrix.groupby('Entity_Stable_Id', sort=False, dropna=False)
        .agg(
            HugoSymbol=('HugoSymbol', lambda s: '/'.join(dict.fromkeys(str(x) for x in s if pd.notna(x)))),
            Location=('Location', lambda s: '/'.join(dict.fromkeys(str(x) for x in s if pd.notna(x)))),
        )
        .reset_index()
    )
    matrix_values = matrix.drop_duplicates('Entity_Stable_Id', keep='first')[['Entity_Stable_Id'] + sample_cols]
    unique = meta.merge(matrix_values, on='Entity_Stable_Id', how='left', validate='one_to_one')
    clin = prepare_clinical(clinical, sample_cols)
    matched_samples = [c for c in sample_cols if c in set(clin.matrix_sample)]
    clin = clin.set_index('matrix_sample').loc[matched_samples].reset_index()
    quality_by_probe = quality.set_index('Entity_Stable_Id')
    assert quality_by_probe.index.is_unique
    assert set(meta.Entity_Stable_Id).issubset(quality_by_probe.index)
    eligible = meta.Entity_Stable_Id.map(quality_by_probe.eligible_primary).astype(bool).to_numpy()
    unique = unique.loc[eligible].reset_index(drop=True)
    meta = meta.loc[eligible].reset_index(drop=True)
    meta.attrs['samples'] = matched_samples
    meta.attrs['input_unique_probes'] = int(len(eligible))
    meta.attrs['quality_excluded_probes'] = int((~eligible).sum())
    beta = unique[matched_samples].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
    return meta, beta, clin, matched_samples


def differential_analysis(meta: pd.DataFrame, beta: np.ndarray, clin: pd.DataFrame) -> pd.DataFrame:
    group = clin.set_index("matrix_sample").loc[meta.attrs["samples"], "group"].to_numpy()
    ia = group == AFR
    ie = group == EUR
    clipped = np.clip(beta, 1e-5, 1 - 1e-5)
    mval = np.log2(clipped / (1 - clipped))

    mean_a = np.nanmean(beta[:, ia], axis=1)
    mean_e = np.nanmean(beta[:, ie], axis=1)
    delta = mean_a - mean_e
    var_a = np.nanvar(beta[:, ia], axis=1, ddof=1)
    var_e = np.nanvar(beta[:, ie], axis=1, ddof=1)
    n_a = np.isfinite(beta[:, ia]).sum(axis=1)
    n_e = np.isfinite(beta[:, ie]).sum(axis=1)
    v_a, v_e = var_a/n_a, var_e/n_e
    se_delta = np.sqrt(v_a+v_e)
    df_welch = (v_a+v_e)**2/(v_a**2/(n_a-1)+v_e**2/(n_e-1))
    critical = stats.t.ppf(.975,df_welch)
    t_unadj, p_unadj = stats.ttest_ind(mval[:, ia], mval[:, ie], axis=1, equal_var=False, nan_policy="omit")
    q_unadj = bh_fdr(p_unadj)

    # Observed-case regression at each CpG; HC3 standard errors allow
    # unequal residual variance. Group identical observation masks to
    # preserve vectorisation without treating imputed data as observed.
    c = clin.set_index("matrix_sample").loc[meta.attrs["samples"]].copy()
    design = clinical_design(c)
    x = design.to_numpy(dtype=float)
    y = mval.T
    race_coef=np.full(beta.shape[0],np.nan)
    race_se=race_coef.copy();race_p=race_coef.copy();ordinary_p=race_coef.copy()
    patterns={}
    for j in range(y.shape[1]):
        key=np.isfinite(y[:,j]).tobytes()
        patterns.setdefault(key,[]).append(j)
    for key,idx in patterns.items():
        ok=np.frombuffer(key,dtype=bool)
        xx=x[ok]; yy=y[np.ix_(ok,idx)]
        rank=np.linalg.matrix_rank(xx); df_resid=len(xx)-rank
        if df_resid<=0 or np.linalg.matrix_rank(xx[:,np.arange(xx.shape[1])!=1])==rank:
            continue
        inv=np.linalg.pinv(xx.T@xx); influence=xx@inv
        coef=inv@xx.T@yy;resid=yy-xx@coef
        leverage=np.sum(influence*xx,axis=1)
        hc3=np.sqrt(np.sum((influence[:,1,None]*resid/np.maximum(1-leverage[:,None],1e-8))**2,axis=0))
        ordinary_se=np.sqrt(inv[1,1]*np.sum(resid**2,axis=0)/df_resid)
        race_coef[idx]=coef[1];race_se[idx]=hc3
        race_p[idx]=2*stats.t.sf(np.abs(coef[1]/hc3),df_resid)
        ordinary_p[idx]=2*stats.t.sf(np.abs(coef[1]/ordinary_se),df_resid)
    race_t=race_coef/race_se
    race_q = bh_fdr(race_p)

    out = meta.copy()
    out["primary_gene"] = out.HugoSymbol.astype(str).str.split("/").str[0]
    out["primary_location"] = out.Location.astype(str).str.split("/").str[0]
    out["n_AFR"] = n_a
    out["n_EUR"] = n_e
    out["mean_beta_AFR"] = mean_a
    out["mean_beta_EUR"] = mean_e
    out["delta_beta_AFR_minus_EUR"] = delta
    out["delta_beta_ci_low"] = delta - critical * se_delta
    out["delta_beta_ci_high"] = delta + critical * se_delta
    out["welch_t_M_value"] = t_unadj
    out["p_unadjusted"] = p_unadj
    out["q_unadjusted"] = q_unadj
    out["adjusted_race_coefficient_M_value"] = race_coef
    out["adjusted_race_se"] = race_se
    out["adjusted_race_t"] = race_t
    out["p_adjusted"] = race_p
    out["q_adjusted"] = race_q
    out['p_adjusted_ordinary_OLS']=ordinary_p
    out['q_adjusted_ordinary_OLS']=bh_fdr(ordinary_p)
    out['missing_fraction_AFR']=1-n_a/ia.sum()
    out['missing_fraction_EUR']=1-n_e/ie.sum()
    out['p_group_missingness']=[stats.fisher_exact([[int(ia.sum()-na),int(na)],[int(ie.sum()-ne),int(ne)]]).pvalue for na,ne in zip(n_a,n_e)]
    out['q_group_missingness']=bh_fdr(out.p_group_missingness.to_numpy())
    out['high_completeness']=(out.missing_fraction_AFR<=.05)&(out.missing_fraction_EUR<=.05)
    out["missing_fraction"] = np.mean(~np.isfinite(beta), axis=1)
    out["reportable_DMP"] = (out.q_adjusted < 0.05) & (out.delta_beta_AFR_minus_EUR.abs() >= 0.05)
    out["direction"] = np.where(out.delta_beta_AFR_minus_EUR > 0, "AFR hypermethylated", "EUR hypermethylated")
    return out


def pca_analysis(beta: np.ndarray, samples: list[str], clin: pd.DataFrame) -> tuple[pd.DataFrame, PCA]:
    keep = np.argsort(np.nanvar(beta,axis=1))[-min(5000,beta.shape[0]):]
    x = SimpleImputer(strategy='mean').fit_transform(beta[keep].T)
    x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-8)
    model = PCA(n_components=6, random_state=SEED)
    scores = model.fit_transform(x)
    out = pd.DataFrame(scores, columns=[f"PC{i}" for i in range(1, 7)])
    out["matrix_sample"] = samples
    out = out.merge(clin[["matrix_sample", "patient_id", "group"]], on="matrix_sample", how="left")
    return out, model


def classifier_analysis(beta: np.ndarray, samples: list[str], clin: pd.DataFrame, permutations: int = 100):
    group = clin.set_index("matrix_sample").loc[samples, "group"]
    y = (group.to_numpy() == AFR).astype(int)
    x = beta.T
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def pipeline(k: int) -> Pipeline:
        return Pipeline(
            [
                ('variance', TrainingVarianceFilter(5000)),
                ('impute', SimpleImputer(strategy='mean')),
                ("select", SelectKBest(f_classif, k=min(k, x.shape[1]))),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        solver="liblinear",
                        C=1.0,
                        max_iter=5000,
                        random_state=SEED,
                    ),
                ),
            ]
        )

    oof = np.full(len(y), np.nan)
    for train, test in cv.split(x, y):
        model = pipeline(100)
        model.fit(x[train], y[train])
        oof[test] = model.predict_proba(x[test])[:, 1]
    oof_df = pd.DataFrame(
        {
            "matrix_sample": samples,
            "group": group.to_numpy(),
            "observed_AFR": y,
            "oof_probability_AFR": oof,
            "predicted_AFR_at_0.5": (oof >= 0.5).astype(int),
        }
    )
    observed = {
        "roc_auc": float(roc_auc_score(y, oof)),
        "average_precision": float(average_precision_score(y, oof)),
        "balanced_accuracy": float(balanced_accuracy_score(y, oof >= 0.5)),
    }

    clinical_x=clinical_design(clin,include_group=False).drop(columns='intercept').to_numpy()
    clinical_oof=np.full(len(y),np.nan)
    for train,test in cv.split(clinical_x,y):
        model=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(class_weight='balanced',solver='liblinear',C=1.0,max_iter=5000,random_state=SEED))])
        model.fit(clinical_x[train],y[train]);clinical_oof[test]=model.predict_proba(clinical_x[test])[:,1]
    oof_df['clinical_oof_probability_AFR']=clinical_oof
    observed['clinical_roc_auc']=float(roc_auc_score(y,clinical_oof))
    observed['clinical_average_precision']=float(average_precision_score(y,clinical_oof))

    k_rows = []
    for k in [10, 25, 50, 100, 250, 500]:
        scores = cross_validate(
            pipeline(k),
            x,
            y,
            cv=cv,
            scoring={"roc_auc": "roc_auc", "average_precision": "average_precision", "balanced_accuracy": "balanced_accuracy"},
            n_jobs=1,
        )
        for fold in range(5):
            k_rows.append(
                {
                    "selected_features": k,
                    "fold": fold + 1,
                    "roc_auc": scores["test_roc_auc"][fold],
                    "average_precision": scores["test_average_precision"][fold],
                    "balanced_accuracy": scores["test_balanced_accuracy"][fold],
                }
            )
    k_df = pd.DataFrame(k_rows)

    rng = np.random.default_rng(SEED)
    null_rows = []
    for i in range(permutations):
        yp = rng.permutation(y)
        pred = np.full(len(y), np.nan)
        for train, test in cv.split(x, yp):
            model = pipeline(100)
            model.fit(x[train], yp[train])
            pred[test] = model.predict_proba(x[test])[:, 1]
        null_rows.append({"permutation": i + 1, "roc_auc": roc_auc_score(yp, pred), "average_precision": average_precision_score(yp, pred)})
    null_df = pd.DataFrame(null_rows)
    observed["permutation_p_roc_auc"] = float((1 + (null_df.roc_auc >= observed["roc_auc"]).sum()) / (1 + len(null_df)))
    return oof_df, k_df, null_df, observed


def pca_covariate_associations(pca: pd.DataFrame, clin: pd.DataFrame) -> pd.DataFrame:
    """Quantify how leading methylome PCs relate to measured cohort features."""
    d = pca.merge(
        clin[["matrix_sample", "group", "age", "male", "hpv_positive", "hpv_unknown", "stage_iv", "site"]],
        on="matrix_sample",
        how="left",
        suffixes=("", "_clinical"),
        validate="one_to_one",
    )
    d["AFR group"] = (d.group == AFR).astype(float)
    d["Age"] = pd.to_numeric(d.age, errors="coerce")
    d["Male sex"] = pd.to_numeric(d.male, errors="coerce")
    d["HPV positive"] = pd.to_numeric(d.hpv_positive, errors="coerce")
    d["HPV unknown"] = d.hpv_unknown.astype(float)
    d["Stage IV"] = pd.to_numeric(d.stage_iv, errors="coerce")
    d['Oropharynx'] = d.site.eq('Oropharynx').astype(float)
    d['Larynx'] = d.site.eq('Larynx').astype(float)
    rows = []
    for covariate in ["AFR group", "Age", "Male sex", "HPV positive", "HPV unknown", "Stage IV", "Oropharynx", "Larynx"]:
        for pc in [f"PC{i}" for i in range(1, 7)]:
            valid = d[[covariate, pc]].dropna()
            r, p = stats.pearsonr(valid[covariate], valid[pc])
            rows.append({"covariate": covariate, "principal_component": pc, "pearson_r": r, "p_value": p, "n": len(valid)})
    return pd.DataFrame(rows)


def make_figure1(outdir: Path, clin: pd.DataFrame, balance: pd.DataFrame, pca: pd.DataFrame, pca_model: PCA) -> pd.DataFrame:
    associations = pca_covariate_associations(pca, clin)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.3), constrained_layout=True)
    ax = axes[0, 0]
    heat = associations.pivot(index="covariate", columns="principal_component", values="pearson_r").loc[
        ["AFR group", "Age", "Male sex", "HPV positive", "HPV unknown", "Stage IV", "Oropharynx", "Larynx"],
        [f"PC{i}" for i in range(1, 7)],
    ]
    sns.heatmap(
        heat,
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 6.5},
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Pearson r", "shrink": 0.78},
    )
    ax.set(xlabel="Methylome principal component", ylabel="")
    ax.set_title("Clinical features and methylome components", loc="left")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    panel_label(ax, "a")

    ax = axes[0, 1]
    counts = clin.group.value_counts().reindex([AFR, EUR])
    bars = ax.bar([0, 1], counts.values, color=[AFR_COLOR, EUR_COLOR], width=0.58)
    for b, n in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(counts) * 0.02, f"{n}", ha="center", va="bottom", fontweight="bold")
    ax.set_xticks([0, 1], ["AFR", "EUR"])
    ax.set_ylabel("Patients")
    ax.set_title("Cohort composition", loc="left")
    ax.text(0.03, 0.94, f"AFR prevalence = {counts[AFR] / counts.sum():.1%}", transform=ax.transAxes, ha="left", va="top", color=AFR_COLOR, fontweight="bold")
    sns.despine(ax=ax)
    panel_label(ax, "b")

    ax = axes[1, 0]
    y = np.arange(len(balance))
    colors = [AFR_COLOR if v > 0 else EUR_COLOR for v in balance.smd]
    ax.axvspan(-0.1, 0.1, color="#F1F5F9", zorder=0)
    ax.axvline(0, color="#55606A", lw=0.9)
    ax.hlines(y, 0, balance.smd, color=colors, lw=2)
    ax.scatter(balance.smd, y, color=colors, s=45, zorder=3, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y, balance.feature)
    ax.invert_yaxis()
    ax.set_xlabel("Standardized mean difference (AFR - EUR)")
    ax.set_title("Clinical covariate distributions", loc="left")
    ax.text(0.01, -0.18, "|SMD| > 0.10 indicates material imbalance; adjusted methylation models include these covariates.", transform=ax.transAxes, color=MID, fontsize=7.1)
    sns.despine(ax=ax)
    panel_label(ax, "c")

    ax = axes[1, 1]
    for group, color in [(EUR, EUR_COLOR), (AFR, AFR_COLOR)]:
        d = pca[pca.group == group]
        ax.scatter(d.PC1, d.PC2, s=23 if group == EUR else 34, alpha=0.58 if group == EUR else 0.85, c=color, edgecolor="white", linewidth=0.35, label=f"{group} (n={len(d)})")
    ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("Global tumour methylome structure", loc="left")
    ax.legend(frameon=False, loc="best")
    sns.despine(ax=ax)
    panel_label(ax, "d")
    save_figure(fig, outdir / "figure1_cohort_and_methylome_structure")
    return associations


def make_figure2(outdir: Path, diff: pd.DataFrame, beta: np.ndarray, samples: list[str], clin: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(11.2, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.08])

    ax = fig.add_subplot(gs[0, 0])
    d = diff.copy()
    y = -np.log10(np.maximum(d.q_adjusted.to_numpy(), 1e-300))
    base = ~d.reportable_DMP
    ax.scatter(d.loc[base, "delta_beta_AFR_minus_EUR"], y[base], s=7, c="#B8C0C7", alpha=0.42, linewidth=0)
    for direction, color in [("AFR hypermethylated", AFR_COLOR), ("EUR hypermethylated", EUR_COLOR)]:
        m = d.reportable_DMP & d.direction.eq(direction)
        ax.scatter(d.loc[m, "delta_beta_AFR_minus_EUR"], y[m], s=12, c=color, alpha=0.78, linewidth=0, label=f"{direction} (n={m.sum()})")
    ax.axvline(-0.05, color="#7A838B", ls="--", lw=0.8)
    ax.axvline(0.05, color="#7A838B", ls="--", lw=0.8)
    ax.axhline(-np.log10(0.05), color="#7A838B", ls="--", lw=0.8)
    top = d[d.reportable_DMP].nsmallest(5, "q_adjusted")
    offsets = [(5, -18), (-18, 22), (8, 4), (5, 12), (-46, -18)]
    for (_, r), offset in zip(top.iterrows(), offsets):
        ax.annotate(
            f"{r.HugoSymbol}\n{r.Entity_Stable_Id}",
            (r.delta_beta_AFR_minus_EUR, -np.log10(max(r.q_adjusted, 1e-300))),
            xytext=offset,
            textcoords="offset points",
            fontsize=5.7,
            color=INK,
            arrowprops=dict(arrowstyle="-", color="#9AA2A9", lw=0.5),
        )
    ax.set_xlabel("Delta beta (AFR - EUR)")
    ax.set_ylabel("-log10 adjusted FDR")
    ax.set_title("Covariate-adjusted differential methylation", loc="left")
    ax.legend(frameon=False, fontsize=6.4, loc="upper right")
    sns.despine(ax=ax)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    top = d.assign(abs_delta=d.delta_beta_AFR_minus_EUR.abs()).sort_values(["reportable_DMP", "abs_delta", "q_adjusted"], ascending=[False, False, True]).head(14).sort_values("delta_beta_AFR_minus_EUR")
    yy = np.arange(len(top))
    c = [AFR_COLOR if v > 0 else EUR_COLOR for v in top.delta_beta_AFR_minus_EUR]
    ax.hlines(yy, top.delta_beta_ci_low, top.delta_beta_ci_high, color=c, lw=1.5)
    ax.scatter(top.delta_beta_AFR_minus_EUR, yy, color=c, s=34, edgecolor="white", linewidth=0.5, zorder=3)
    labels = [f"{g} | {loc}\n{probe}" for g, loc, probe in zip(top.primary_gene, top.primary_location, top.Entity_Stable_Id)]
    ax.set_yticks(yy, labels)
    ax.axvline(0, color="#606A73", lw=0.8)
    ax.set_xlabel("Delta beta with approximate 95% CI")
    ax.set_title("Leading methylation differences", loc="left")
    sns.despine(ax=ax)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    sig = d[d.reportable_DMP].copy()
    if sig.empty:
        sig = d[d.q_adjusted < 0.05].copy()
    comp = sig.groupby(["primary_location", "direction"]).size().unstack(fill_value=0)
    comp = comp.reindex(["TSS1500", "TSS200", "5'UTR", "1stExon", "Body"]).dropna(how="all").fillna(0)
    left = np.zeros(len(comp))
    for direction, color in [("EUR hypermethylated", EUR_COLOR), ("AFR hypermethylated", AFR_COLOR)]:
        vals = comp.get(direction, pd.Series(0, index=comp.index)).to_numpy()
        ax.barh(comp.index, vals, left=left, color=color, label=direction)
        left += vals
    ax.set_xlabel("Differential CpGs")
    ax.set_title("Genomic annotations of differential CpGs", loc="left")
    ax.legend(frameon=False, fontsize=6.5)
    sns.despine(ax=ax)
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    top_idx = d.assign(rank_score=-np.log10(np.maximum(d.q_adjusted, 1e-300)) * d.delta_beta_AFR_minus_EUR.abs()).nlargest(25, "rank_score").index.to_numpy()
    mat = beta[top_idx]
    row_mean = np.nanmean(mat, axis=1, keepdims=True)
    z = (mat - row_mean) / np.maximum(np.nanstd(mat, axis=1, keepdims=True), 1e-8)
    groups = clin.set_index("matrix_sample").loc[samples, "group"].to_numpy()
    order = np.argsort(groups == AFR)  # EUR first, AFR second
    z = z[:, order]
    im = ax.imshow(z, aspect="auto", interpolation="nearest", cmap="vlag", vmin=-2.5, vmax=2.5)
    boundary = int(np.sum(groups[order] == EUR))
    ax.axvline(boundary - 0.5, color="white", lw=1.3)
    ax.set_xticks([boundary / 2, boundary + (len(samples) - boundary) / 2], [f"EUR\nn={boundary}", f"AFR\nn={len(samples)-boundary}"])
    heat_labels = [f"{d.loc[i, 'primary_gene']} | {d.loc[i, 'Entity_Stable_Id']}" for i in top_idx]
    ax.set_yticks(np.arange(len(top_idx)), heat_labels, fontsize=5.7)
    ax.set_title("Methylation at leading CpGs", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Probe-wise z score")
    panel_label(ax, "d")
    save_figure(fig, outdir / "figure2_differential_methylation_landscape")


def make_figure3(outdir: Path, oof: pd.DataFrame, k_df: pd.DataFrame, null_df: pd.DataFrame, metrics: dict, robustness: pd.DataFrame | None = None, robustness_predictions: pd.DataFrame | None = None, source_dir: Path | None = None) -> None:
    y = oof.observed_AFR.to_numpy()
    p = oof.oof_probability_AFR.to_numpy()
    has_robustness = robustness is not None and robustness_predictions is not None
    fig, axes = plt.subplots(3 if has_robustness else 2, 2, figsize=(10.8,11.6 if has_robustness else 8.4), constrained_layout=True)

    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, color=PURPLE, lw=2.2, label=f"5-fold OOF AUC = {metrics['roc_auc']:.3f}")
    cfpr,ctpr,_=roc_curve(y,oof.clinical_oof_probability_AFR)
    ax.plot(cfpr,ctpr,color=MID,ls=':',lw=1.4,label=f"Clinical covariates AUC = {metrics['clinical_roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="#9AA2A9", lw=1)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.02))
    ax.set_title("Held-out receiver operating characteristic", loc="left")
    ax.legend(frameon=False, loc="lower right")
    sns.despine(ax=ax)
    panel_label(ax, "a")

    ax = axes[0, 1]
    precision, recall, _ = precision_recall_curve(y, p)
    baseline = y.mean()
    ax.plot(recall, precision, color=TEAL, lw=2.2, label=f"Average precision = {metrics['average_precision']:.3f}")
    cp,cr,_=precision_recall_curve(y,oof.clinical_oof_probability_AFR)
    ax.plot(cr,cp,color=MID,ls=':',lw=1.4,label=f"Clinical covariates AP = {metrics['clinical_average_precision']:.3f}")
    ax.axhline(baseline, color="#9AA2A9", ls="--", lw=1, label=f"Prevalence baseline = {baseline:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.set_title("Held-out precision and recall", loc="left")
    ax.legend(frameon=False, loc="best")
    sns.despine(ax=ax)
    panel_label(ax, "b")

    ax = axes[1, 0]
    summary = k_df.groupby("selected_features").agg(roc=("roc_auc", "mean"), roc_sd=("roc_auc", "std"), ap=("average_precision", "mean"), ap_sd=("average_precision", "std")).reset_index()
    ax.errorbar(summary.selected_features, summary.roc, yerr=summary.roc_sd, color=PURPLE, marker="o", capsize=3, label="ROC AUC")
    ax.errorbar(summary.selected_features, summary.ap, yerr=summary.ap_sd, color=TEAL, marker="s", capsize=3, label="Average precision")
    ax.set_xscale("log")
    ax.set_xticks(summary.selected_features, summary.selected_features.astype(str))
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Features selected within each training fold")
    ax.set_ylabel("Cross-validated performance")
    ax.set_title("Sensitivity to feature count", loc="left")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    panel_label(ax, "c")

    ax = axes[1, 1]
    sns.histplot(null_df.roc_auc, bins=18, color="#B8C0C7", edgecolor="white", ax=ax)
    ax.axvline(metrics["roc_auc"], color=PURPLE, lw=2.2, label=f"Observed = {metrics['roc_auc']:.3f}")
    ax.set_xlabel("OOF ROC AUC after group permutation")
    ax.set_ylabel("Permutations")
    ax.set_title("Group-label permutation test", loc="left")
    ax.legend(frameon=False)
    ax.text(0.45, 0.78, f"Permutation P = {metrics['permutation_p_roc_auc']:.3g}\nBalanced accuracy = {metrics['balanced_accuracy']:.3f}", transform=ax.transAxes, ha="left", va="top", color=INK, fontsize=8,bbox={'facecolor':'white','edgecolor':'none','alpha':.9})
    sns.despine(ax=ax)
    panel_label(ax, "d")
    if has_robustness:
        centre=robustness_predictions[robustness_predictions.model.eq('Held-out collection centres')]
        ax=axes[2,0]
        sns.boxplot(data=centre,x='group',y='probability_AFR',hue='group',order=[EUR,AFR],palette={AFR:AFR_COLOR,EUR:EUR_COLOR},width=.45,showfliers=False,legend=False,ax=ax)
        sns.stripplot(data=centre,x='group',y='probability_AFR',hue='group',order=[EUR,AFR],palette={AFR:AFR_COLOR,EUR:EUR_COLOR},jitter=.14,size=2.5,alpha=.35,legend=False,ax=ax)
        ax.axhline(.5,ls='--',color=MID,lw=.8)
        ax.set(xlabel='Clinical group',ylabel='Held-out probability of AFR membership',ylim=(0,1.02),title='Predictions in held-out collection centres')
        sns.despine(ax=ax);panel_label(ax,'e')
        ax=axes[2,1]
        comparisons=pd.concat([pd.DataFrame([{'model':'Clinical covariates','roc_auc':metrics['clinical_roc_auc'],'average_precision':metrics['clinical_average_precision']},{'model':'Primary methylation','roc_auc':metrics['roc_auc'],'average_precision':metrics['average_precision']}]),robustness],ignore_index=True)
        yy=np.arange(len(comparisons))
        ax.barh(yy-.16,comparisons.roc_auc,height=.30,color=PURPLE,label='ROC AUC')
        ax.barh(yy+.16,comparisons.average_precision,height=.30,color=TEAL,label='Average precision')
        ax.set_yticks(yy,['Clinical covariates','Primary methylation','High completeness','Held-out centres'],fontsize=8)
        ax.set(xlabel='Pooled held-out performance',xlim=(0,1.10),title='Performance across validation settings')
        for yi,row in comparisons.iterrows():
            ax.text(row.roc_auc+.015,yi-.16,f'{row.roc_auc:.3f}',va='center',fontsize=7)
            ax.text(row.average_precision+.015,yi+.16,f'{row.average_precision:.3f}',va='center',fontsize=7)
        ax.legend(frameon=False,fontsize=7,loc='lower right');sns.despine(ax=ax);panel_label(ax,'f')
        if source_dir is not None:
            comparisons.to_csv(source_dir/'figure3_panel_f_validation_metrics.tsv',sep='\t',index=False)
    save_figure(fig, outdir / "figure3_cross_validated_group_separability")


def make_figure4(outdir: Path, corr: pd.DataFrame) -> pd.DataFrame:
    c = corr.copy()
    c["pair"] = c.HugoSymbol.astype(str) + " | " + c.Entity_Stable_Id.astype(str)
    c['primary_location']=c.Location.str.split(';').str[0]
    c["regulatory_signal"] = (c.q_value < 0.05) & (c.spearman_r.abs() >= 0.30)
    allc = c[c.Group == "All"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.6), constrained_layout=True)

    ax = axes[0, 0]
    q = np.maximum(allc.q_value.to_numpy(dtype=float), 1e-300)
    sig = allc.regulatory_signal
    ax.scatter(allc.loc[~sig, "spearman_r"], -np.log10(q[~sig]), s=7, c="#B8C0C7", alpha=0.35, linewidth=0)
    pos = sig & (allc.spearman_r > 0)
    neg = sig & (allc.spearman_r < 0)
    ax.scatter(allc.loc[neg, "spearman_r"], -np.log10(q[neg]), s=9, c=EUR_COLOR, alpha=0.68, linewidth=0, label=f"Inverse (n={neg.sum()})")
    ax.scatter(allc.loc[pos, "spearman_r"], -np.log10(q[pos]), s=9, c=AFR_COLOR, alpha=0.68, linewidth=0, label=f"Positive (n={pos.sum()})")
    ax.axvline(-0.3, color="#7A838B", ls="--", lw=0.8)
    ax.axvline(0.3, color="#7A838B", ls="--", lw=0.8)
    ax.set_xlabel("Spearman correlation (methylation vs mRNA)")
    ax.set_ylabel("-log10 FDR")
    ax.set_title("Methylation-expression coupling is predominantly inverse", loc="left")
    ax.legend(frameon=False, fontsize=6.6)
    sns.despine(ax=ax)
    panel_label(ax, "a")

    ax = axes[0, 1]
    neg_top = allc.sort_values("spearman_r").drop_duplicates("pair").head(8)
    pos_top = allc.sort_values("spearman_r", ascending=False).drop_duplicates("pair").head(8)
    top = pd.concat([neg_top, pos_top]).sort_values("spearman_r")
    yy = np.arange(len(top))
    colors = [EUR_COLOR if v < 0 else AFR_COLOR for v in top.spearman_r]
    ax.hlines(yy, 0, top.spearman_r, color=colors, lw=2)
    ax.scatter(top.spearman_r, yy, color=colors, s=34, edgecolor="white", linewidth=0.5)
    ax.set_yticks(yy, top.pair, fontsize=6.4)
    ax.axvline(0, color="#65717A", lw=0.8)
    ax.set_xlabel("Spearman rho")
    ax.set_title("Largest inverse and positive correlations", loc="left")
    sns.despine(ax=ax)
    panel_label(ax, "b")

    ax = axes[1, 0]
    order = [x for x in ["TSS1500", "TSS200", "5'UTR", "1stExon", "Body"] if x in allc.primary_location.unique()]
    sns.violinplot(data=allc, x="primary_location", y="spearman_r", order=order, color="#D8E3EC", inner="quartile", linewidth=0.7, cut=0, ax=ax)
    ax.axhline(0, color="#66717A", lw=0.8)
    ax.set_xlabel("CpG annotation")
    ax.set_ylabel("Spearman rho")
    ax.set_title("Correlation distributions by annotation", loc="left")
    sns.despine(ax=ax)
    panel_label(ax, "c")

    ax = axes[1, 1]
    afr = c[c.Group == "African"][["pair", "spearman_r", "q_value"]].rename(columns={"spearman_r": "r_AFR", "q_value": "q_AFR"})
    eur = c[c.Group == "White"][["pair", "spearman_r", "q_value"]].rename(columns={"spearman_r": "r_EUR", "q_value": "q_EUR"})
    joined = afr.merge(eur, on="pair", how="inner").dropna()
    joined["concordant"] = np.sign(joined.r_AFR) == np.sign(joined.r_EUR)
    ax.scatter(joined.r_EUR, joined.r_AFR, s=7, c=np.where(joined.concordant, "#8FA6B7", "#D4A72C"), alpha=0.35, linewidth=0)
    lim = max(abs(joined[["r_AFR", "r_EUR"]].to_numpy()).max(), 0.8)
    ax.plot([-lim, lim], [-lim, lim], ls="--", color="#6B747C", lw=0.9)
    rho, rp = stats.spearmanr(joined.r_EUR, joined.r_AFR)
    ax.set(xlabel="EUR-group Spearman rho", ylabel="AFR-group Spearman rho", xlim=(-lim, lim), ylim=(-lim, lim))
    ax.set_title("Group-specific correlation estimates", loc="left")
    p_text = "<1e-300" if rp == 0 else f"{rp:.2g}"
    ax.text(0.03, 0.96, f"Spearman rho = {rho:.2f}\nP {p_text}", transform=ax.transAxes, ha="left", va="top", color=INK)
    discord = joined.assign(diff=lambda x: (x.r_AFR - x.r_EUR).abs()).nlargest(3, "diff")
    offsets = [(4, 6), (5, -12), (-34, 6)]
    for (_, r), offset in zip(discord.iterrows(), offsets):
        ax.annotate(r.pair.split(" | ")[0], (r.r_EUR, r.r_AFR), xytext=offset, textcoords="offset points", fontsize=6, arrowprops=dict(arrowstyle="-", color="#8B949C", lw=0.45))
    sns.despine(ax=ax)
    panel_label(ax, "d")
    save_figure(fig, outdir / "figure4_methylation_expression_coupling")
    return c


def make_supplementary(outdir: Path, diff: pd.DataFrame, beta: np.ndarray, samples: list[str], clin: pd.DataFrame, corr_clean: pd.DataFrame) -> None:
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.3), constrained_layout=True)
    groups = clin.set_index("matrix_sample").loc[samples, "group"].to_numpy()
    ax = axes[0, 0]
    for grp, color in [(EUR, EUR_COLOR), (AFR, AFR_COLOR)]:
        idx = np.where(groups == grp)[0]
        vals = beta[:, idx].ravel()
        vals = vals[np.isfinite(vals)]
        vals = rng.choice(vals, size=min(250000, vals.size), replace=False)
        sns.kdeplot(vals, color=color, lw=1.7, label=grp, ax=ax, clip=(0, 1))
    ax.set(xlabel="Methylation beta value", ylabel="Density", title="Global beta-value distributions")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    panel_label(ax, "a")

    ax = axes[0, 1]
    sample_missing = np.mean(~np.isfinite(beta), axis=0)
    q = pd.DataFrame({"group": groups, "missing": sample_missing})
    sns.boxplot(data=q, x="group", y="missing", order=[AFR, EUR], palette=[AFR_COLOR, EUR_COLOR], width=0.55, ax=ax)
    ax.set(xlabel="", ylabel="Missing probe fraction", title="Per-sample methylation completeness")
    sns.despine(ax=ax)
    panel_label(ax, "b")

    ax = axes[1, 0]
    x = -np.log10(np.maximum(diff.p_unadjusted, 1e-300))
    y = -np.log10(np.maximum(diff.p_adjusted, 1e-300))
    ax.hexbin(x, y, gridsize=55, mincnt=1, cmap="Blues", bins="log")
    lim = min(max(x.max(), y.max()), 320)
    ax.plot([0, lim], [0, lim], ls="--", color="#6B747C", lw=0.8)
    rho = stats.spearmanr(x, y).statistic
    ax.set(xlabel="-log10 unadjusted P", ylabel="-log10 adjusted P", title="Adjusted and unadjusted statistical evidence")
    ax.text(0.04, 0.94, f"Spearman rho = {rho:.3f}", transform=ax.transAxes, va="top")
    sns.despine(ax=ax)
    panel_label(ax, "c")

    ax = axes[1, 1]
    thresholds = np.linspace(0, 0.20, 21)
    counts = [(diff.q_adjusted.lt(0.05) & diff.delta_beta_AFR_minus_EUR.abs().ge(t)).sum() for t in thresholds]
    ax.plot(thresholds, counts, color=PURPLE, marker="o", ms=3.5)
    ax.axvline(0.05, color="#6B747C", ls="--", lw=0.9)
    ax.set(xlabel="Minimum absolute delta beta", ylabel="CpG probes at FDR < 0.05", title="Effect-size threshold sensitivity")
    sns.despine(ax=ax)
    panel_label(ax, "d")
    save_figure(fig, outdir / "supplementary_figureS1_methylation_qc_and_robustness")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    allc = corr_clean[corr_clean.Group == "All"].copy()
    ax = axes[0]
    summary = allc[allc.q_value<.05].groupby("primary_location").agg(n=("spearman_r", "size"), median_r=("spearman_r", "median"), inverse=("spearman_r", lambda s: (s <= -0.3).sum()), positive=("spearman_r", lambda s: (s >= 0.3).sum())).reset_index()
    xx = np.arange(len(summary))
    ax.bar(xx - 0.18, summary.inverse, width=0.36, color=EUR_COLOR, label="r < -0.30")
    ax.bar(xx + 0.18, summary.positive, width=0.36, color=AFR_COLOR, label="r > 0.30")
    ax.set_xticks(xx, summary.primary_location, rotation=25, ha="right")
    ax.set_ylabel("Gene-location pairs")
    ax.set_title("Large regulatory correlations by annotation", loc="left")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    panel_label(ax, "a")
    ax = axes[1]
    piv = corr_clean.pivot_table(index="pair", columns="Group", values="spearman_r", aggfunc="first").dropna(subset=["African", "White"])
    piv["difference"] = piv.African - piv.White
    top = piv.reindex(piv.difference.abs().sort_values(ascending=False).head(18).index).sort_values("difference")
    colors = [AFR_COLOR if x > 0 else EUR_COLOR for x in top.difference]
    ax.barh(top.index, top.difference, color=colors)
    ax.axvline(0, color="#6B747C", lw=0.8)
    ax.set_xlabel("Spearman rho difference (AFR - EUR)")
    ax.set_title("Largest differences in correlation estimates", loc="left")
    ax.tick_params(axis="y", labelsize=6)
    sns.despine(ax=ax)
    panel_label(ax, "b")
    save_figure(fig, outdir / "supplementary_figureS2_regulatory_coupling_robustness")


def write_panel_sources(source_dir: Path, diff: pd.DataFrame, balance: pd.DataFrame, pca: pd.DataFrame, pca_associations: pd.DataFrame, oof: pd.DataFrame, k_df: pd.DataFrame, null_df: pd.DataFrame, corr: pd.DataFrame) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    mappings = {
        "figure1_panel_a_pc_covariate_associations.tsv": pca_associations,
        "figure1_panel_b_cohort_counts.tsv": pd.DataFrame({"group": [AFR, EUR], "n": [(pca.group == AFR).sum(), (pca.group == EUR).sum()]}),
        "figure1_panel_c_clinical_balance.tsv": balance,
        "figure1_panel_d_pca_scores.tsv": pca,
        "figure2_panel_a_differential_all_probes.tsv": diff,
        "figure2_panel_b_top_effects.tsv": diff[diff.reportable_DMP].assign(abs_delta=diff.loc[diff.reportable_DMP,'delta_beta_AFR_minus_EUR'].abs()).nlargest(14, "abs_delta"),
        "figure2_panel_c_annotation_counts.tsv": diff[diff.reportable_DMP].groupby(["primary_location", "direction"]).size().rename("n").reset_index(),
        "figure3_panels_a_b_oof_predictions.tsv": oof,
        "figure3_panel_c_feature_count_sensitivity.tsv": k_df,
        "figure3_panel_d_permutation_null.tsv": null_df,
        "figure4_all_correlation_panels.tsv": corr,
    }
    for name, frame in mappings.items():
        frame.to_csv(source_dir / name, sep="\t", index=False)
