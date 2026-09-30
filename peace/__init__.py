from importlib import import_module

__version__ = "0.1.0"
__all__ = [
    "PEACEModel", "canonicalize_model_config", "default_config",
    "model_from_config", "load_model_artifacts", "Calculator", "load_calculator", "__version__",
]

_MODEL_EXPORTS = {
    "Calculator": ".calculator.api",
    "load_calculator": ".calculator.api",
    "PEACEModel": ".nn.model",
    "canonicalize_model_config": ".nn.model",
    "default_config": ".nn.model",
    "model_from_config": ".nn.model",
    "load_model_artifacts": ".nn.io",
}

def __getattr__(name):
    module_name = _MODEL_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value

def __dir__():
    return sorted(set(globals()) | set(__all__))
