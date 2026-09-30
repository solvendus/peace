from functools import partial
import re
import jax
import jaxlib

def model_compiler_options(backend=None):
    release = tuple(int(x) for x in re.findall(r'\d+', jaxlib.__version__)[:2])
    platform = backend or jax.default_backend()
    if platform in ('gpu', 'cuda') and release >= (0, 9):
        return {
            'xla_gpu_experimental_enable_fusion_autotuner': False,
            'xla_gpu_enable_triton_gemm': False,
        }
    return {}

def model_jit(fun=None, /, **kwargs):
    if fun is None:
        return partial(model_jit, **kwargs)
    options = dict(kwargs.pop('compiler_options', None) or {})
    options.update(model_compiler_options(kwargs.get('backend')))
    if options:
        kwargs['compiler_options'] = options
    return jax.jit(fun, **kwargs)

def execution_config():
    return {
        'policy': 'model_derivative_compilation_v1',
        'jax_version': jax.__version__,
        'jaxlib_version': jaxlib.__version__,
        'backend': jax.default_backend(),
        'devices': [device.device_kind for device in jax.devices()],
        'jax_enable_x64': bool(jax.config.jax_enable_x64),
        'matmul_precision': str(jax.config.jax_default_matmul_precision),
        'compiler_options': model_compiler_options(),
    }