from collections.abc import Mapping
from pathlib import Path
import pickle
from typing import Any, Dict, List, Tuple
from peace.elements import chemical_symbols
import flax.linen as nn
import jax
import jax.numpy as jnp
import yaml
from .model import canonicalize_model_config, model_from_config

def promote_parameters_to_x64(params: Any) -> Any:
    if not jax.config.jax_enable_x64:
        return params
    def promote(value):
        dtype = getattr(value, 'dtype', None)
        if dtype is None:
            if isinstance(value, float):
                return jnp.asarray(value, dtype=jnp.float64)
            if isinstance(value, complex):
                return jnp.asarray(value, dtype=jnp.complex128)
            return value
        if jnp.issubdtype(dtype, jnp.floating):
            return jnp.asarray(value, dtype=jnp.float64)
        if jnp.issubdtype(dtype, jnp.complexfloating):
            return jnp.asarray(value, dtype=jnp.complex128)
        return value
    return jax.tree_util.tree_map(promote, params)

def validate_checkpoint_architecture(params, architecture, latent_n_states=None, *, include_soc=False):
    root = params.get('params', params)
    if include_soc:
        if architecture != 'peace' or not {'singlet', 'triplet', 'soc_head'}.issubset(root):
            raise ValueError('SOC checkpoint requires singlet, triplet and soc_head parameters')
        for sector in ('singlet', 'triplet'):
            validate_checkpoint_architecture(root[sector], architecture)
        dimension = sum(root[k]['diagonal_output']['bias'].shape[0] for k in ('singlet', 'triplet'))
        if latent_n_states is not None and dimension != latent_n_states:
            raise ValueError('SOC checkpoint spin-free dimension does not match model config')
        return
    if architecture != 'peace' or not {
        'species_encoder', 'diagonal_output', 'internal_connection_head',
        'rigid_connection_head'}.issubset(root):
        raise ValueError('Checkpoint is not compatible with the PEACE model architecture.')
    if latent_n_states is not None:
        dimension = root['diagonal_output']['bias'].shape
        if dimension != (latent_n_states,):
            raise ValueError(
                f'Checkpoint Hamiltonian dimension {dimension} does not match '
                f'latent_n_states={latent_n_states}; use the matching model config.')

def read_model_config(path):
    """Read the exported model config; retain historical SOC basis conventions."""
    with Path(path).open() as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("config must be a YAML/JSON mapping of model settings")
    raw = dict(raw)
    if raw.get("include_soc") is True and "soc_spin_convention" not in raw:
        raw["soc_spin_convention"] = "legacy_cartesian_v1"
    return canonicalize_model_config(raw)

def load_model_artifacts(
    model_path: str,
    params_path: str,
) -> Tuple[nn.Module, Any, Dict[str, Any]]:
    """Load compatible model defaults and honor x64 for all floating parameters."""
    config = read_model_config(model_path)
    model = model_from_config(config)
    with Path(params_path).open("rb") as handle:
        params = pickle.load(handle)
    if not isinstance(params, Mapping):
        raise ValueError("params must contain a Flax parameter mapping")
    if "params" not in params:
        params = {"params": params}
    validate_checkpoint_architecture(params, config.architecture, config.latent_n_states,
                                     include_soc=config.include_soc)
    if config.include_soc:
        root = params.get('params', params)
        for label, count in (('singlet', config.n_singlets), ('triplet', config.n_triplets)):
            if root[label]['diagonal_output']['bias'].shape != (count,):
                raise ValueError(f'SOC checkpoint {label} dimension disagrees with config')
    params = promote_parameters_to_x64(params)

    return model, params, config.to_dict()

def atomic_energy_offset(config: Dict[str, Any], atomic_numbers) -> float:
    """Return the composition-dependent reference energy stored in a model config."""
    e0s = config.get("E0s") or {}
    total = 0.0
    for z_value in atomic_numbers:
        z = int(z_value)
        symbol = chemical_symbols[z]
        total += float(e0s.get(symbol, e0s.get(z, e0s.get(str(z), 0.0))))
    return total