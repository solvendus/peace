"""O(3)-equivariant, padding-safe multihead attention for one molecular graph.
"""
import math
import e3nn_jax as e3nn
import flax.linen as nn
import jax.numpy as jnp
from .layers import Linear

def invariant_features(features):
    pieces = [features.filter('0e').array]
    for (mul, ir), chunk in zip(features.irreps, features.chunks):
        if ir == e3nn.Irrep('0e'):
            continue
        if chunk is None:
            pieces.append(jnp.zeros(features.shape[:-1] + (mul,), dtype=features.dtype))
        else:
            pieces.append(jnp.log1p(jnp.sum(chunk * chunk, axis=-1) / ir.dim))
    return jnp.concatenate(pieces, axis=-1)

def masked_attention(logits, node_mask):
    valid = jnp.asarray(node_mask).reshape(-1) > .5
    logits = jnp.where(valid[None, None, :], logits, jnp.finfo(logits.dtype).min)
    shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
    numerator = jnp.exp(shifted) * valid[None, None, :]
    denominator = jnp.maximum(jnp.sum(numerator, axis=-1, keepdims=True), 1.)
    return numerator / denominator * valid[None, :, None]

class EquivariantSelfAttention(nn.Module):
    num_heads: int = 4
    head_dim: int = 16
    distance_scale: float = 8.0
    residual_scale: float = 0.1

    @nn.compact
    def __call__(self, features, positions, node_mask):
        mask = jnp.asarray(node_mask, dtype=features.dtype).reshape(-1, 1)
        scalar = invariant_features(features * mask)
        scalar = nn.LayerNorm(param_dtype=scalar.dtype, name='invariant_norm')(scalar)
        width = self.num_heads * self.head_dim
        query = nn.Dense(width, use_bias=False, param_dtype=scalar.dtype, name='query')(scalar)
        key = nn.Dense(width, use_bias=False, param_dtype=scalar.dtype, name='key')(scalar)
        query = query.reshape(-1, self.num_heads, self.head_dim)
        key = key.reshape(-1, self.num_heads, self.head_dim)
        logits = jnp.einsum('ihd,jhd->hij', query, key) / math.sqrt(self.head_dim)
        squared_distance = jnp.sum((positions[:, None] - positions[None, :]) ** 2, axis=-1)
        decay = self.param('distance_decay', nn.initializers.zeros, (self.num_heads,), scalar.dtype)
        logits = logits - nn.softplus(decay)[:, None, None] * (
            squared_distance[None] / self.distance_scale ** 2)
        weights = masked_attention(logits, mask)
        self.sow('intermediates', 'weights', weights)

        value_irreps = e3nn.Irreps([
            (self.num_heads * math.ceil(mul / self.num_heads), ir)
            for mul, ir in features.irreps])
        values = Linear(irreps_out=value_irreps, name='value')(features * mask)
        chunks = []
        for (mul, ir), chunk in zip(values.irreps, values.chunks):
            if chunk is None:
                chunks.append(None)
                continue
            chunk = chunk.reshape(-1, self.num_heads, mul // self.num_heads, ir.dim)
            mixed = jnp.einsum('hij,jhcd->ihcd', weights, chunk)
            chunks.append(mixed.reshape(-1, mul, ir.dim))
        mixed = e3nn.from_chunks(values.irreps, chunks, values.shape[:-1], dtype=values.dtype)
        update = Linear(irreps_out=features.irreps, name='output')(mixed)
        return (features + self.residual_scale * update) * mask