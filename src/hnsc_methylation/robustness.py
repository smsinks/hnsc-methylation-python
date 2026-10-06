"""Independent numerical verification and HPV-negative sensitivity models."""
from pathlib import Path
import platform
import numpy as np
import pandas as pd
from scipy import stats
import scipy,sklearn,matplotlib,lifelines


from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score

from . import core as m


def differential_robustness(beta: np.ndarray, clin: pd.DataFrame, diff: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Repeat the reviewed independent HC3 and HPV-negative checks."""
    c = clin.reset_index(drop=True).copy()
    d = diff.reset_index(drop=True).copy()
    b = beta
    x=m.clinical_design(c).to_numpy(float)
    checks=[]
    for i in list(d.nsmallest(8,'q_adjusted').index)+[i for i in [0,100,1000,10000] if i < len(d)]:
        ok=np.isfinite(b[i]);v=np.clip(b[i,ok],1e-5,1-1e-5)
        xx=x[ok];yy=np.log2(v/(1-v))
        coef=np.linalg.lstsq(xx,yy,rcond=None)[0]
        pseudoinverse=np.linalg.pinv(xx)
        residual=yy-xx@coef
        leverage=np.diag(xx@pseudoinverse)
        hc_cov=(pseudoinverse*(residual/(1-leverage))**2)@pseudoinverse.T
        check_p=2*stats.t.sf(abs(coef[1]/np.sqrt(hc_cov[1,1])),len(yy)-np.linalg.matrix_rank(xx))
        assert np.isclose(coef[1],d.iloc[i].adjusted_race_coefficient_M_value,rtol=1e-8,atol=1e-10)
        assert np.isclose(check_p,d.iloc[i].p_adjusted,rtol=1e-7,atol=1e-100)
        checks.append({'probe':d.iloc[i].Entity_Stable_Id,'p_independent_SVD_HC3':check_p,'p_pipeline':d.iloc[i].p_adjusted})
    assert np.allclose(stats.false_discovery_control(d.p_adjusted,method='bh'),d.q_adjusted,rtol=1e-10,atol=1e-100)
    negative=(c.hpv_positive.eq(0)&c.hpv_unknown.eq(0)).to_numpy()
    meta=d[['Entity_Stable_Id','HugoSymbol','Location']].copy()
    meta.attrs['samples']=c.loc[negative,'matrix_sample'].tolist()
    neg=m.differential_analysis(meta,b[:,negative],c.loc[negative].copy())
    selected=d[d.reportable_DMP].copy()
    joined=selected[['Entity_Stable_Id','q_adjusted','delta_beta_AFR_minus_EUR','adjusted_race_coefficient_M_value','high_completeness']].merge(neg[['Entity_Stable_Id','q_adjusted','delta_beta_AFR_minus_EUR','adjusted_race_coefficient_M_value','reportable_DMP']],on='Entity_Stable_Id',suffixes=('_primary','_hpv_negative'))
    summary={
        'independently_verified_HC3_models':len(checks),
        'BH_agrees_with_SciPy':True,
        'n_primary':len(c),'n_hpv_negative':int(negative.sum()),
        'hpv_negative_groups':c.loc[negative,'group'].value_counts().to_dict(),
        'sites_by_group':pd.crosstab(c.site,c.group).to_dict(),
        'stage_by_group':pd.crosstab(c.stage,c.group).to_dict(),
        'hpv_unknown_by_group':c.groupby('group').hpv_unknown.sum().to_dict(),
        'primary_DMPs':int(d.reportable_DMP.sum()),
        'primary_DMPs_high_completeness':int(selected.high_completeness.sum()),
        'primary_DMPs_HC3_and_ordinary_OLS_q_lt_005':int(selected.q_adjusted_ordinary_OLS.lt(.05).sum()),
        'hpv_negative_DMPs':int(neg.reportable_DMP.sum()),
        'primary_DMPs_retained_HPV_negative_BH_and_effect':int(joined.reportable_DMP.sum()),
        'primary_DMPs_HPV_negative_same_coefficient_direction':int((np.sign(joined.adjusted_race_coefficient_M_value_primary)==np.sign(joined.adjusted_race_coefficient_M_value_hpv_negative)).sum()),
        'primary_missingness_flags_BH005':selected.loc[selected.q_group_missingness<.05,['Entity_Stable_Id','missing_fraction_AFR','missing_fraction_EUR','q_group_missingness']].to_dict('records'),
        'adjusted_unadjusted_spearman_p_ranking':stats.spearmanr(d.p_adjusted,d.p_unadjusted).statistic,
        'versions':{'Python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__,'lifelines':lifelines.__version__,'matplotlib':matplotlib.__version__},
    }
    return pd.DataFrame(checks), neg, joined, summary


class TrainingCompletenessFilter(BaseEstimator,TransformerMixin):
    def fit(self,X,y):
        self.keep_=np.isfinite(X[y==0]).mean(axis=0)>=.95
        self.keep_&=np.isfinite(X[y==1]).mean(axis=0)>=.95
        return self
    def transform(self,X):
        return X[:,self.keep_]



def classifier_robustness(beta: np.ndarray, clin: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Preserve fixed-parameter completeness and collection-centre validation."""
    c = clin.reset_index(drop=True).copy()
    b = beta.T
    y=c.group.eq('AFR').to_numpy(int);centres=c.patient_id.str.split('-').str[1].to_numpy()
    predictions=[];metrics=[];folds=[]
    for name,complete,cv in [('High completeness',True,StratifiedKFold(5,shuffle=True,random_state=42)),('Held-out collection centres',False,StratifiedGroupKFold(5,shuffle=True,random_state=42))]:
        pred=np.full(len(y),np.nan);fold_ids=np.zeros(len(y),int)
        for f,(train,test) in enumerate(cv.split(b,y,groups=centres) if isinstance(cv,StratifiedGroupKFold) else cv.split(b,y),1):
            if isinstance(cv,StratifiedGroupKFold):
                assert not set(centres[train])&set(centres[test])
            steps=[('complete',TrainingCompletenessFilter())] if complete else []
            steps += [('variance',m.TrainingVarianceFilter(5000)),('impute',SimpleImputer(strategy='mean')),('select',SelectKBest(f_classif,k=100)),('scale',StandardScaler()),('model',LogisticRegression(class_weight='balanced',solver='liblinear',C=1,max_iter=5000,random_state=42))]
            model=Pipeline(steps).fit(b[train],y[train])
            pred[test]=model.predict_proba(b[test])[:,1];fold_ids[test]=f
            folds.append({'model':name,'fold':f,'n_train':len(train),'n_test':len(test),'AFR_test':int(y[test].sum()),'EUR_test':int((1-y[test]).sum()),'train_centres':len(set(centres[train])),'test_centres':len(set(centres[test]))})
        metrics.append({'model':name,'roc_auc':roc_auc_score(y,pred),'average_precision':average_precision_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred>=.5)})
        predictions.append(pd.DataFrame({'matrix_sample':c.matrix_sample,'group':c.group,'observed_AFR':y,'model':name,'fold':fold_ids,'probability_AFR':pred,'collection_centre':centres}))

    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True), pd.DataFrame(folds)
