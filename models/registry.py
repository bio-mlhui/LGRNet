_model_entrypoints = {}

def register_model(fn):
    model_name = fn.__name__
    if model_name in _model_entrypoints:
        raise ValueError(f'model name {model_name} has been registered')
    _model_entrypoints[model_name] = fn

    return fn

def model_entrypoint(model_name):
    try:
        return _model_entrypoints[model_name]
    except KeyError as e:
        print(f'Model Name {model_name} not found')

from detectron2.utils.registry import Registry
MODELITY_INPUT_MAPPER_REGISTRY = Registry("MODELITY_INPUT_MAPPER")
