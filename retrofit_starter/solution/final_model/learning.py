"""Healthy conditional models and supervised multi-label fault ensembles."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, ExtraTreesClassifier
from threadpoolctl import threadpool_limits
from features import CHANNELS

FAULTS=['BEARING_WEAR','SUCTION_STRAINER_CLOGGING','CAVITATION','DISCHARGE_RESTRICTION',
 'MISALIGNMENT','IMBALANCE','IMPELLER_WEAR','AIR_ENTRAINMENT','SENSOR_DRIFT','SENSOR_STUCK',
 'SENSOR_CALIBRATION_SHIFT','SENSOR_NOISE_SPIKES','COMMUNICATION_LOSS']
SENSORS=set(FAULTS[8:])


def expert_columns(fault,columns):
    """Route measured evidence by physical mechanism, without equipment identity.

    These are input families, not diagnostic thresholds. Every decision boundary
    is fitted on labelled training examples including competing fault classes.
    """
    if fault=='COMMUNICATION_LOSS':
        return [c for c in columns if c in ['obs__gap','obs__gap_minutes','obs__row_fraction']]
    if fault in SENSORS:
        return [c for c in columns if not c.startswith(('value__','rating__','profile__','control__'))]
    mechanical=fault in ['BEARING_WEAR','MISALIGNMENT','IMBALANCE']
    channels=(['bearing_temp_c','motor_winding_temp_c','vibration_rms_mm_s',
        'vibration_accel_rms_g','vibration_crest_factor'] if mechanical else
        ['motor_current_a','motor_power_kw','suction_pressure_kpa','discharge_pressure_kpa',
        'flow_m3_h','vibration_rms_mm_s','vibration_accel_rms_g','vibration_crest_factor'])
    physics=(['bearing_','winding_','vibration_','acceleration_','crest_','velocity_'] if mechanical else
        ['current_','power_','electrical_','speed_command_','flow_','curve_','head_',
         'suction_','discharge_','hydraulic_','system_','npsh','pressure_',
         'vibration_','acceleration_','crest_'])
    selected=[]
    for c in columns:
        if c.startswith('snapshot__') and c!='snapshot__age_min':selected.append(c)
        elif 'phys__' in c and any(c.split('phys__',1)[1].startswith(p) for p in physics):selected.append(c)
        elif '__' in c:
            prefix,channel=c.split('__',1)
            if channel in channels and prefix in ['level','std','range','stdmean','mean10','mean60',
                'span10','span60','delta10','delta60','stdmean10','stdmean60',
                'residual','resid_mean10','resid_mean60','resid_delta10','resid_delta60']:
                selected.append(c)
    return selected


def make_targets(meta,events):
    """Unknown/invisible labels are -1; concurrent visible faults are multi-hot.

    The returned weights balance individual events, rather than letting a week
    of one pre-existing condition dominate short incidents. Labels never enter X.
    """
    n=len(meta);y=pd.DataFrame(0,index=range(n),columns=FAULTS,dtype=np.int8)
    w=pd.DataFrame(1.,index=range(n),columns=FAULTS)
    healthy=np.ones(n,bool)
    t=pd.to_datetime(meta.available_utc,utc=True)
    for _,e in events.iterrows():
        if e.fault_type not in FAULTS:continue
        onset=pd.to_datetime(e.onset_utc,utc=True)
        end=pd.to_datetime(e.end_utc,utc=True)
        detected=pd.to_datetime(e.detectable_utc,utc=True)
        active=(meta.asset_id.eq(e.asset_id)&t.ge(onset)&(t.lt(end) if pd.notna(end) else True)).to_numpy()
        healthy[active]=False
        observable=str(e.observable).lower() in ('true','1')
        if not observable:
            y.loc[active,e.fault_type]=-1
            continue
        begin=onset if pd.isna(detected) or e.fault_type in ('CAVITATION','AIR_ENTRAINMENT','COMMUNICATION_LOSS') else max(onset,detected)
        positive=active&t.ge(begin).to_numpy()
        y.loc[active&~positive,e.fault_type]=-1
        y.loc[positive,e.fault_type]=1
        if positive.any():w.loc[positive,e.fault_type]=1000./positive.sum()
    if 'steady' in meta:
        for fault in FAULTS:
            if fault not in SENSORS:y.loc[~meta.steady.to_numpy(),fault]=-1
    return y,w,healthy


@dataclass
class LearnedModel:
    kind: str='hist'
    seed: int=20261107
    regressors: dict=field(default_factory=dict)
    classifiers: dict=field(default_factory=dict)
    columns: list=field(default_factory=list)
    include_waveforms: bool=True
    include_residuals: bool=True
    training_info: dict=field(default_factory=dict)
    routed: bool=False
    classifier_columns: dict=field(default_factory=dict)

    def fit_reference(self,x,meta,healthy):
        context=[c for c in x if c.startswith(('ctx__','rating__','control__','profile__'))]
        context += ['level__motor_speed_rpm','level__fluid_temp_c','level__motor_power_kw','obs__run_minutes','obs__running']
        rng=np.random.default_rng(self.seed)
        clean=healthy&meta.steady.to_numpy()
        with threadpool_limits(limits=3):
            for c in CHANNELS:
                target='level__'+c
                cols=[v for v in context if v!=target]
                good=np.flatnonzero(clean&x[target].notna().to_numpy()&x['fresh__'+c].gt(0).to_numpy())
                if len(good)<80:continue
                if len(good)>18000:good=rng.choice(good,18000,replace=False)
                reg=HistGradientBoostingRegressor(loss='absolute_error',max_iter=100,max_leaf_nodes=15,
                    min_samples_leaf=35,l2_regularization=1.,early_stopping=False,random_state=self.seed)
                # Explicit missing-value code also covers all-missing predictor
                # columns in sklearn1.9's histogram binning edge case. It is
                # never used as an observed healthy sensor value.
                reg.fit(x.iloc[good][cols].fillna(-999),x.iloc[good][target])
                residual=x.iloc[good][target].to_numpy()-reg.predict(x.iloc[good][cols].fillna(-999))
                # Robust fitted spread with a nonzero floor. These are features,
                # not standalone alarms or calibrated anomaly probabilities.
                scale=max(float(np.median(np.abs(residual-np.median(residual))))*1.4826,.003)
                self.regressors[c]=(reg,cols,scale)
        return self

    def transform(self,x,meta):
        result=x.copy()
        if not self.include_waveforms:
            result=result.drop(columns=[c for c in result if c.startswith('snapshot__')])
        if not self.include_residuals:return result
        added={}
        groups=list(meta.groupby('asset_id',sort=False).indices.values())
        with threadpool_limits(limits=3):
            for channel,(reg,cols,scale) in self.regressors.items():
                resid=(x['level__'+channel].to_numpy()-reg.predict(x[cols].fillna(-999)))/scale
                resid=np.clip(resid,-100.,100.)
                added['residual__'+channel]=resid
                for window in [10,60]:
                    mean=np.full(len(x),np.nan,dtype=np.float32)
                    delta=np.full(len(x),np.nan,dtype=np.float32)
                    for ids in groups:
                        s=pd.Series(resid[ids])
                        mean[ids]=s.rolling(window,min_periods=2).mean()
                        delta[ids]=s-s.shift(window)
                    added[f'resid_mean{window}__'+channel]=mean
                    added[f'resid_delta{window}__'+channel]=delta
        if added:
            add=pd.DataFrame(added,index=x.index,dtype=np.float32)
            result=pd.concat([result,add],axis=1)
            instrument=[c for c in add if c.startswith('residual__') and not any(s in c for s in ['vibration','speed','ambient'])]
            if instrument:
                absolute=add[instrument].abs()
                result['residual_sensor_max']=absolute.max(axis=1)
                result['residual_sensor_mean']=absolute.mean(axis=1)
                result['residual_sensor_dominance']=absolute.max(axis=1)/(absolute.sum(axis=1)+1e-6)
        return result.replace([np.inf,-np.inf],np.nan).astype(np.float32)

    def fit_classifier(self,transformed,meta,y,event_weights):
        # Raw dimensional values are redundant with rating-scaled observations;
        # omit them to discourage learning a particular pump's numeric range.
        self.columns=[c for c in transformed if not c.startswith('value__')]
        x=transformed[self.columns].fillna(-999)
        rng=np.random.default_rng(self.seed)
        diagnostics={}
        with threadpool_limits(limits=3):
            for fault in FAULTS:
                cols=expert_columns(fault,self.columns) if self.routed else self.columns
                self.classifier_columns[fault]=cols
                positive=np.flatnonzero(y[fault].eq(1))
                negative=np.flatnonzero(y[fault].eq(0))
                # Positives are sampled with event balancing. Include many
                # other-fault negatives so that the model learns diagnosis.
                balanced_resample=len(positive)>6500
                if balanced_resample:
                    pw=event_weights.iloc[positive][fault].to_numpy()
                    positive=rng.choice(positive,6500,replace=True,p=pw/pw.sum())
                if len(negative)>18000:
                    negative=rng.choice(negative,18000,replace=False)
                ids=np.r_[positive,negative]
                if len(positive)<5:
                    self.classifiers[fault]=0.;continue
                if not len(negative):
                    self.classifiers[fault]=1.;continue
                labels=y.iloc[ids][fault].to_numpy()
                weights=np.ones(len(ids),dtype=float)
                posw=np.ones(len(positive)) if balanced_resample else event_weights.iloc[positive][fault].to_numpy()
                posw=posw/posw.mean()
                weights[:len(positive)]=posw*len(negative)/max(len(positive),1)
                if self.kind=='hist':
                    model=HistGradientBoostingClassifier(max_iter=140,learning_rate=.07,
                        max_leaf_nodes=15,min_samples_leaf=25,l2_regularization=2.,
                        early_stopping=False,random_state=self.seed)
                    model.fit(x.iloc[ids][cols],labels,sample_weight=weights)
                else:
                    model=ExtraTreesClassifier(n_estimators=180,min_samples_leaf=4,
                        max_features=.65,max_depth=24,n_jobs=3,random_state=self.seed)
                    model.fit(x.iloc[ids][cols],labels,sample_weight=weights)
                self.classifiers[fault]=model
                diagnostics[fault]={'positive_samples':len(positive),'negative_samples':len(negative),'features':len(cols)}
                print('fit',self.kind,fault,diagnostics[fault],flush=True)
        self.training_info=diagnostics
        return self

    def probabilities_from_transformed(self,transformed):
        x=transformed.reindex(columns=self.columns).fillna(-999)
        probabilities={}
        with threadpool_limits(limits=3):
            for fault,model in self.classifiers.items():
                cols=getattr(self,'classifier_columns',{}).get(fault,self.columns)
                probabilities[fault]=np.full(len(x),float(model)) if isinstance(model,(int,float)) else model.predict_proba(x[cols])[:,1]
        return pd.DataFrame(probabilities,index=transformed.index).reindex(columns=FAULTS,fill_value=0.)

    def predict_proba(self,x,meta):
        return self.probabilities_from_transformed(self.transform(x,meta))
