import e3nn_jax as e3nn
import flax.linen as nn
import jax.numpy as jnp
from peace.nn.layers import Linear
from peace.nn.parity import ElectronicParity

class StructuredConnectionHead(nn.Module):
    signature: tuple
    hidden_dim: int = 128
    gated: bool = True
    zero_init_output: bool = False

    @nn.compact
    def __call__(self, features, node_mask):
        layout = ElectronicParity(self.signature)
        output = jnp.zeros((features.shape[0], len(layout.pairs), 3), dtype=features.dtype)
        if self.gated:
            hidden = nn.silu(nn.Dense(self.hidden_dim, name='scalar_hidden')(
                features.filter('0e').array))
        for name, irrep, indices in (('polar', '1o', layout.even_pairs),
                                    ('axial', '1e', layout.odd_pairs)):
            if not indices:
                continue
            if self.gated:
                basis = features.filter(irrep).array
                channels = basis.shape[-1] // 3
                if not channels:
                    raise ValueError(f'Electronic parity requires {irrep} connection channels')
                basis = basis.reshape(features.shape[0], channels, 3)
                init = nn.initializers.zeros_init() if self.zero_init_output else nn.initializers.lecun_normal()
                gates = nn.Dense(len(indices) * channels, kernel_init=init,
                                 name=f'{name}_gates')(hidden)
                values = jnp.einsum('npc,ncd->npd',
                    gates.reshape(features.shape[0], len(indices), channels), basis)
            else:
                values = Linear(irreps_out=e3nn.Irreps(f'{len(indices)}x{irrep}'),
                    zero_init=self.zero_init_output, name=f'{name}_output')(features).array
                values = values.reshape(features.shape[0], len(indices), 3)
            output = output.at[:, jnp.asarray(indices, dtype=jnp.int32), :].set(values)
        return output * jnp.asarray(node_mask).reshape(-1, 1, 1)
