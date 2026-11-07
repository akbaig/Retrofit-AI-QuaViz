"""Assemble already-fitted experts; no training or evaluation labels are read."""
import argparse
import joblib
from composite import CompositeModel
from sensor_expert import SENSOR_TYPES
from run import HERE,write_json,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True)
    p.add_argument('--sensor',required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--sensor-types',nargs='+',choices=SENSOR_TYPES,default=list(SENSOR_TYPES))
    a=p.parse_args();out=HERE/'runs'/a.name
    if out.exists():raise FileExistsError('Preserve existing runs: select a new name')
    basepath=HERE/'runs'/a.base/'model.joblib';sensorpath=HERE/'runs'/a.sensor/'model.joblib'
    base=joblib.load(basepath);sensor=joblib.load(sensorpath)
    # Remove overridden classifiers so the artifact contains only active models.
    base.classifiers={k:v for k,v in base.classifiers.items() if k not in a.sensor_types}
    model=CompositeModel(base,sensor,tuple(a.sensor_types))
    out.mkdir(parents=True)
    joblib.dump(model,out/'model.joblib',compress=3)
    write_json(out/'composition.json',dict(base=a.base,sensor=a.sensor,sensor_types=a.sensor_types,
        base_sha256=sha(basepath),sensor_sha256=sha(sensorpath),
        active_base_classifiers=list(base.classifiers),
        source_sha256={p.name:sha(p) for p in [HERE/'composite.py',HERE/'sensor_expert.py',HERE/'make_composite.py']}))
    print('saved',out/'model.joblib')


if __name__=='__main__':main()
