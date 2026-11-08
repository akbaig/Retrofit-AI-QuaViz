"""Combine fitted machine/asset experts with shared per-channel sensor experts."""
from dataclasses import dataclass


@dataclass
class ViewEnsemble:
    """Equal votes from independently fitted measurement views/training cohorts.

    All residual-using views share the same frozen healthy reference. Direct
    physics views select only their own columns from the common transformation.
    """
    views: list
    sensor: object=None
    sensor_faults: tuple=()
    is_composite: bool=True

    def transform(self,x,meta):
        return self.views[0].transform(x,meta)

    def probabilities_from_transformed(self,x,meta):
        scores=sum(model.probabilities_from_transformed(x) for model in self.views)/len(self.views)
        if self.sensor is not None and self.sensor_faults:
            shared=self.sensor.predict(x,meta)
            for fault in self.sensor_faults:scores[fault]=shared[fault]
        return scores

    def predict_proba(self,x,meta):
        return self.probabilities_from_transformed(self.transform(x,meta),meta)
