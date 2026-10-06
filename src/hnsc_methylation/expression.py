"""Patient-matched exact CpG–gene expression correlations from measured data."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .core import bh_fdr


def patient_id(sample: str) -> str:
    return sample.replace('_', '-')[:12]


def clinical_source_fields(clinical: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    """Recover clinical fields by explicit one-to-one patient-ID matching."""
    assert matched.patient_id.is_unique
    assert clinical.PATIENT_ID.is_unique
    fields = matched[['matrix_sample', 'patient_id', 'group']].merge(
        clinical.rename(columns={'PATIENT_ID': 'patient_id'}), on='patient_id',
        how='left', validate='one_to_one', indicator=True,
    )
    assert fields['_merge'].eq('both').all()
    fields = fields.drop(columns='_merge')
    fields['hpv_status'] = fields.SUBTYPE.map(
        {'HNSC_HPV+': 'Positive', 'HNSC_HPV-': 'Negative'}
    ).fillna('Unknown')
    return fields


def expression_analysis(
    methyl: pd.DataFrame,
    expression: pd.DataFrame | Path,
    matched: pd.DataFrame,
    eligible_probes: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Preserve the reviewed two-sided Spearman and finite-test BH procedure.

    Use the original annotation rows, not slash-joined primary annotations.
    Providing eligible_probes limits the test universe to the primary quality
    mask before per-group BH correction. Omitting it returns the original
    full-source exact-correlation table.
    """
    assert matched.patient_id.is_unique
    annotation = methyl[['Entity_Stable_Id', 'HugoSymbol', 'Location']].dropna(
        subset=['Entity_Stable_Id', 'HugoSymbol']
    ).copy()
    annotation = annotation[~annotation.HugoSymbol.astype(str).isin(['NA', 'nan', ''])]
    annotation['Location'] = annotation.Location.fillna('Unannotated').astype(str)
    annotation = annotation.groupby(['Entity_Stable_Id', 'HugoSymbol'], sort=False).Location.agg(
        lambda s: ';'.join(sorted(set(s)))
    ).reset_index()
    expr = pd.read_csv(expression, sep='\t') if isinstance(expression, (str, Path)) else expression.copy()
    expr = expr.dropna(subset=['Hugo_Symbol'])
    duplicate_expression_symbols = int(expr.Hugo_Symbol.duplicated().sum())
    expr = expr.drop_duplicates('Hugo_Symbol', keep='first').set_index('Hugo_Symbol')
    expression_samples = {patient_id(c): c for c in expr.columns if str(c).startswith('TCGA-')}
    assert len(expression_samples) == sum(str(c).startswith('TCGA-') for c in expr.columns)
    paired = matched[matched.patient_id.isin(expression_samples)].copy()
    methyl_columns = paired.matrix_sample.tolist()
    expression_columns = [expression_samples[p] for p in paired.patient_id]
    assert [patient_id(c) for c in methyl_columns] == [patient_id(c) for c in expression_columns]
    methyl_unique = methyl.drop_duplicates('Entity_Stable_Id').set_index('Entity_Stable_Id')
    annotation = annotation[annotation.HugoSymbol.isin(expr.index)].reset_index(drop=True)
    if eligible_probes is not None:
        annotation = annotation[annotation.Entity_Stable_Id.isin(eligible_probes)].reset_index(drop=True)
    mvals = methyl_unique[methyl_columns].to_numpy(dtype=float)
    rvals = expr[expression_columns].to_numpy(dtype=float)
    midx = {probe: i for i, probe in enumerate(methyl_unique.index)}
    ridx = {gene: i for i, gene in enumerate(expr.index)}
    rows = []
    group_indices = {
        'All': np.ones(len(paired), dtype=bool),
        'African': paired.group.eq('AFR').to_numpy(),
        'White': paired.group.eq('EUR').to_numpy(),
    }
    for group, use in group_indices.items():
        for row in annotation.itertuples(index=False):
            x = mvals[midx[row.Entity_Stable_Id], use]
            y = rvals[ridx[row.HugoSymbol], use]
            finite = np.isfinite(x) & np.isfinite(y)
            x, y = x[finite], y[finite]
            n = len(x)
            rho, p = np.nan, np.nan
            if n >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
                result = spearmanr(x, y)
                rho, p = float(result.statistic), float(result.pvalue)
            rows.append({
                'Entity_Stable_Id': row.Entity_Stable_Id,
                'HugoSymbol': row.HugoSymbol, 'Location': row.Location,
                'Group': group, 'spearman_r': rho, 'p_value': p, 'n': n,
                'methylMean': float(np.mean(x)) if n else np.nan,
                'mrnaMean': float(np.mean(y)) if n else np.nan,
            })
    result = pd.DataFrame(rows)
    result['q_value'] = np.nan
    for _, idx in result.groupby('Group').groups.items():
        result.loc[idx, 'q_value'] = bh_fdr(result.loc[idx, 'p_value'].to_numpy())
    result['regulatory_signal'] = result.spearman_r.abs().ge(0.3) & result.q_value.lt(0.05)
    notes = {
        'expression_units': 'Batch-normalized RSEM from Illumina HiSeq RNASeqV2, according to supplied meta_mrna_seq_v2_rsem.txt',
        'paired_n_by_group': {g: int(v.sum()) for g, v in group_indices.items()},
        'unique_probe_gene_pairs': len(annotation),
        'duplicate_expression_gene_symbols_first_record_retained': duplicate_expression_symbols,
        'tested_pairs_by_group': result.groupby('Group').p_value.count().astype(int).to_dict(),
        'regulatory_signal_pairs_by_group': result.groupby('Group').regulatory_signal.sum().astype(int).to_dict(),
        'correlation': 'Two-sided Spearman rank correlation; pairwise complete measured beta and RNA values; no methylation imputation; n is pair-specific.',
        'multiple_testing': 'Benjamini–Hochberg over finite P values for all unique CpG–gene pairs separately within All, African, and White groups.',
        'alignment': 'Patients matched by normalized explicit TCGA patient IDs; All restricted to the manuscript cohort with available RNA. Multiple genomic locations for the same CpG–gene pair are joined and not retested.',
        'probe_universe': 'primary quality-eligible probes' if eligible_probes is not None else 'full supplied methylation matrix',
    }
    return result, paired[['matrix_sample', 'patient_id', 'group']], notes
