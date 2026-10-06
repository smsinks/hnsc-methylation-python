"""Regenerate the final reviewed supplementary panels from explicit results."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from . import core as m


def make_robustness_figures(figdir: Path, source_dir: Path, diff: pd.DataFrame, hpv_negative: pd.DataFrame, corr: pd.DataFrame, clin: pd.DataFrame) -> None:
    """Render final Figures S1 and S2 and their panel-source tables."""
    f, s = Path(figdir), Path(source_dir)
    f.mkdir(parents=True, exist_ok=True)
    s.mkdir(parents=True, exist_ok=True)
    m.set_style()
    d, neg = diff, hpv_negative
    allc = corr[corr.Group.eq('All')]
    fig,axes=plt.subplots(2,2,figsize=(10.5,8.2),constrained_layout=True)
    ax=axes[0,0]
    miss=d[d.missing_fraction.gt(0)].copy()
    ax.scatter(miss.missing_fraction_EUR*100,miss.missing_fraction_AFR*100,s=20,color=m.PURPLE,alpha=.55)
    for _,r in miss.nlargest(3,'missing_fraction').iterrows():
        ax.annotate(r.primary_gene,(r.missing_fraction_EUR*100,r.missing_fraction_AFR*100),xytext=(3,5),textcoords='offset points',fontsize=8)
    ax.plot([0,30],[0,30],ls='--',lw=.8,color=m.MID)
    ax.set(xlabel='Missing measurements in EUR (%)',ylabel='Missing measurements in AFR (%)',title='Probe-level measurement completeness')
    ax.text(.03,.96,f'{len(miss)} probes with any missing value',transform=ax.transAxes,va='top',fontsize=8)
    ax=axes[0,1]
    x=-np.log10(d.p_adjusted_ordinary_OLS.clip(lower=1e-100));y=-np.log10(d.p_adjusted.clip(lower=1e-100))
    ax.hexbin(x,y,gridsize=45,mincnt=1,cmap='Blues',bins='log')
    lim=max(x.max(),y.max());ax.plot([0,lim],[0,lim],ls='--',lw=.8,color=m.MID)
    ax.set(xlabel='−log10 P with conventional standard errors',ylabel='−log10 P with HC3 standard errors',title='Sensitivity to residual variance assumptions')
    ax=axes[1,0]
    z=d[d.reportable_DMP].merge(neg[['Entity_Stable_Id','delta_beta_AFR_minus_EUR','reportable_DMP']],on='Entity_Stable_Id',suffixes=('_primary','_negative'))
    for selected,color in [(True,m.TEAL),(False,m.MID)]:
        use=z.reportable_DMP_negative.eq(selected)
        ax.scatter(z.loc[use,'delta_beta_AFR_minus_EUR_primary'],z.loc[use,'delta_beta_AFR_minus_EUR_negative'],color=color,s=34,label=f'HPV-negative criteria {"met" if selected else "not met"} (n={int(use.sum())})')
    ax.plot([-.25,.16],[-.25,.16],ls='--',lw=.8,color=m.MID)
    ax.axhline(0,lw=.5,color=m.MID);ax.axvline(0,lw=.5,color=m.MID)
    ax.set(xlabel=f'Delta beta in all {len(clin)} patients',ylabel=f'Delta beta in {int((clin.hpv_positive.eq(0)&clin.hpv_unknown.eq(0)).sum())} HPV-negative patients',title='Differential CpGs in HPV-negative disease')
    ax.legend(frameon=False,fontsize=7)
    ax=axes[1,1]
    thresholds=np.linspace(0,.20,21);rows=[]
    for label,frame,color in [('All patients',d,m.PURPLE),('HPV-negative patients',neg,m.TEAL)]:
        counts=[int((frame.q_adjusted.lt(.05)&frame.delta_beta_AFR_minus_EUR.abs().ge(v)).sum()) for v in thresholds]
        ax.plot(thresholds,counts,color=color,marker='o',ms=3,label=label)
        rows.extend([{'cohort':label,'absolute_delta_beta_threshold':v,'DMP_count':n} for v,n in zip(thresholds,counts)])
    ax.axvline(.05,ls='--',lw=.8,color=m.MID)
    ax.set(xlabel='Minimum absolute delta beta',ylabel='CpGs at BH q < 0.05',title='Sensitivity to methylation effect size')
    ax.legend(frameon=False,fontsize=8)
    for ax,label in zip(axes.flat,'abcd'):
        m.panel_label(ax,label);sns.despine(ax=ax)
    m.save_figure(fig,f/'supplementary_figureS1_methylation_qc_and_robustness')
    miss.to_csv(s/'figureS1_panel_a_probe_missingness.tsv',sep='\t',index=False)
    d[['Entity_Stable_Id','p_adjusted_ordinary_OLS','p_adjusted']].to_csv(s/'figureS1_panel_b_variance_assumptions.tsv',sep='\t',index=False)
    z.to_csv(s/'figureS1_panel_c_HPV_negative_effects.tsv',sep='\t',index=False)
    pd.DataFrame(rows).to_csv(s/'figureS1_panel_d_effect_thresholds.tsv',sep='\t',index=False)

    fig,axes=plt.subplots(1,2,figsize=(10.8,4.6),constrained_layout=True)
    ax=axes[0]
    ann=allc[allc.q_value.lt(.05)].groupby('primary_location').spearman_r.agg(inverse=lambda v:v.le(-.30).sum(),positive=lambda v:v.ge(.30).sum()).reset_index()
    xx=np.arange(len(ann))
    ax.bar(xx-.18,ann.inverse,width=.36,color=m.EUR_COLOR,label='rho ≤ −0.30')
    ax.bar(xx+.18,ann.positive,width=.36,color=m.AFR_COLOR,label='rho ≥ 0.30')
    ax.set_xticks(xx,ann.primary_location,rotation=25,ha='right')
    ax.set(ylabel='CpG–gene pairs at BH q < 0.05',title='Correlation direction by annotation')
    ax.legend(frameon=False,fontsize=8)
    ax=axes[1]
    piv=corr.pivot(index='pair',columns='Group',values='spearman_r').dropna(subset=['African','White'])
    piv['difference']=piv.African-piv.White
    top=piv.loc[piv.difference.abs().nlargest(18).index].sort_values('difference')
    ax.barh(top.index,top.difference,color=[m.AFR_COLOR if v>0 else m.EUR_COLOR for v in top.difference])
    ax.set(xlabel='Spearman rho difference (AFR − EUR)',title='Largest group-specific coefficient differences')
    ax.tick_params(axis='y',labelsize=7)
    ax.axvline(0,color=m.MID,lw=.7)
    for ax,label in zip(axes,'ab'):
        m.panel_label(ax,label);sns.despine(ax=ax)
    m.save_figure(fig,f/'supplementary_figureS2_regulatory_coupling_robustness')
    ann.to_csv(s/'figureS2_panel_a_annotation_counts.tsv',sep='\t',index=False)
    top.to_csv(s/'figureS2_panel_b_group_correlations.tsv',sep='\t')
