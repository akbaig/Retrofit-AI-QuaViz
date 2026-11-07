"""Equal-weight, development-tested multi-view ensemble; no labels read here."""
import argparse
import joblib
from composite import ViewEnsemble
from sensor_expert import SENSOR_TYPES
from run import HERE,write_json,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--views',nargs='+',required=True)
    p.add_argument('--sensor')
    p.add_argument('--sensor-types',nargs='+',choices=SENSOR_TYPES,default=['SENSOR_NOISE_SPIKES'])
    p.add_argument('--name',required=True)
    p.add_argument('--prune-unused',action='store_true',help='Drop sensor classifiers replaced by the shared expert')
    a=p.parse_args();out=HERE/'runs'/a.name
    if out.exists():raise FileExistsError('Preserve existing runs: select a new name')
    paths=[HERE/'runs'/n/'model.joblib' for n in a.views]
    views=[joblib.load(path) for path in paths]
    assert views[0].include_residuals,'First view supplies common healthy residuals'
    reference_hash=joblib.hash(views[0].regressors)
    for model in views:
        if model.include_residuals:
            assert joblib.hash(model.regressors)==reference_hash
            model.regressors=views[0].regressors
    sensorpath=HERE/'runs'/a.sensor/'model.joblib' if a.sensor else None
    sensor=joblib.load(sensorpath) if sensorpath else None
    if a.prune_unused and sensor:
        for model in views:
            model.classifiers={k:v for k,v in model.classifiers.items() if k not in a.sensor_types}
    model=ViewEnsemble(views,sensor,tuple(a.sensor_types) if sensor else ())
    out.mkdir(parents=True)
    joblib.dump(model,out/'model.joblib',compress=3)
    write_json(out/'composition.json',dict(views=a.views,weights=[1/len(views)]*len(views),
        sensor=a.sensor,sensor_types=list(model.sensor_faults),
        pruned_unused=a.prune_unused,
        source_models={name:sha(path) for name,path in zip(a.views,paths)},
        sensor_sha256=sha(sensorpath) if sensor else None,
        source_sha256={p.name:sha(p) for p in [HERE/'composite.py',HERE/'sensor_expert.py',HERE/'make_ensemble.py']}))
    print('saved',out/'model.joblib')


if __name__=='__main__':main()
