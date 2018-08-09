import os
from pathlib import Path

serialized_models = map(
    lambda path: str(path),
    Path('../final-models/with-simple-fcn-regressor').rglob('*.hdf5')
)


def get_serialized_models(use_only_cpu_compatible=True):
    if (use_only_cpu_compatible == True):
        return list(filter(lambda name: 'resnet-inspired' in name, serialized_models))
    else:
        return list(serialized_models)
