from functools import partial, reduce
import operator
import jax
from typing import Callable, Dict, Optional, Tuple, Union, List
import flax.linen as nn
from jax.nn import initializers
import jax.numpy as jnp
import e3nn_jax as e3nn
import jraph
from e3nn_jax.legacy import FunctionalTensorProduct, FunctionalFullyConnectedTensorProduct
from jraph import GraphsTuple

Array = jax.Array
FeaturizerFn = Callable[
    [GraphsTuple, Array, Array, Optional[Array]], GraphsTuple
]
Irrep = e3nn.Irrep
Irreps = e3nn.Irreps
IrrepsArray = e3nn.IrrepsArray
f32 = jnp.float32
normal = lambda var: initializers.variance_scaling(var, 'fan_in', 'normal')
UnaryFn = Callable[[Array], Array]
tree_map = partial(
    jax.tree_util.tree_map, is_leaf=lambda x: isinstance(x, e3nn.IrrepsArray)
)

def prod(xs):
  return reduce(operator.mul, xs, 1)

def tp_path_exists(arg_in1, arg_in2, arg_out):
  arg_in1 = Irreps(arg_in1).simplify()
  arg_in2 = Irreps(arg_in2).simplify()
  arg_out = Irrep(arg_out)
  for multiplicity_1, irreps_1 in arg_in1:
    for multiplicity_2, irreps_2 in arg_in2:
      if arg_out in irreps_1 * irreps_2:
        return True
  return False

class BetaSwish(nn.Module):

  @nn.compact
  def __call__(self, x):
    features = x.shape[-1]
    beta = self.param('Beta', nn.initializers.ones, (features,))
    return x * nn.sigmoid(beta * x)

NONLINEARITY = {
    'none': lambda x: x,
    'relu': nn.relu,
    'swish': BetaSwish(),
    'raw_swish': nn.swish,
    'tanh': nn.tanh,
    'sigmoid': nn.sigmoid,
    'silu': nn.silu,
}

def get_nonlinearity_by_name(name: str) -> UnaryFn:
  if name in NONLINEARITY:
    return NONLINEARITY[name]
  raise ValueError(f'Nonlinearity "{name}" not found.')

class MLP(nn.Module):
  """Multilayer Perceptron."""

  features: Tuple[int, ...]
  nonlinearity: str

  use_bias: bool = True
  scalar_mlp_std: Optional[float] = None

  @nn.compact
  def __call__(self, x):
    features = self.features

    dense = partial(nn.Dense, use_bias=self.use_bias)
    phi = get_nonlinearity_by_name(self.nonlinearity)

    kernel_init = normal(self.scalar_mlp_std)

    for h in features[:-1]:
      x = phi(dense(h, kernel_init=kernel_init)(x))

    return dense(features[-1], kernel_init=normal(1.0))(x)

def mlp(
    hidden_features: Union[int, Tuple[int, ...]], nonlinearity: str, **kwargs
) -> Callable[..., Array]:
  if isinstance(hidden_features, int):
    hidden_features = (hidden_features,)

  def mlp_fn(*args):
    fn = MLP(hidden_features, nonlinearity, **kwargs)
    return jraph.concatenated_args(fn)(*args)

  return mlp_fn

def bessel(r_c, frequencies, r):

  rp = jnp.where(r > f32(1e-5), r, f32(1000.0))
  b = 2 / r_c * jnp.sin(frequencies * rp / r_c) / rp

  return jnp.where(r > f32(1e-5), b, 0)

def smooth_cutoff_envelope(
    distances: Array,
    inner_cutoff: float,
    outer_cutoff: float,
) -> Array:
  r_c = outer_cutoff ** f32(2)
  r_o = inner_cutoff ** f32(2)
  r = distances ** f32(2)
  inner = jnp.where(
      distances < outer_cutoff,
      (r_c - r) ** 2 * (r_c + 2 * r - 3 * r_o) / (r_c - r_o) ** 3,
      0,
  )
  return jnp.where(distances < inner_cutoff, 1, inner) * jnp.ones_like(distances)

class BesselEmbedding(nn.Module):
  count: int
  inner_cutoff: float
  outer_cutoff: float

  @nn.compact
  def __call__(self, rs: Array) -> Array:
    def init_fn(key, shape):
      del key
      assert len(shape) == 1
      n = shape[0]
      return jnp.arange(1, n + 1) * jnp.pi

    frequencies = self.param('frequencies', init_fn, (self.count,))
    bessel_fn = jax.vmap(
        partial(bessel, self.outer_cutoff, frequencies))
    envelope = smooth_cutoff_envelope(
        rs, self.inner_cutoff, self.outer_cutoff)
    return envelope[:, None] * bessel_fn(rs)

class LocalChiralPseudoscalarSeed(nn.Module):
    n_channels: int = 16
    hidden_dim: int = 64
    eps: float = 1.0e-8
    init_gain: float = 5.0
    scalar_mlp_std: float = 4.0

    @nn.compact
    def __call__(
        self,
        edge_vec: Array,        # (E, 3), dR / shifts
        edge_dst: Array,        # (E,)
        edge_embedded: Array,   # (E, num_basis)
        edge_weight: Array,     # (E, 1), binary mask or smooth cutoff weight
        n_nodes: int,
    ) -> e3nn.IrrepsArray:

        valid = jnp.squeeze(edge_weight, axis=-1).astype(edge_vec.dtype)

        r = jnp.linalg.norm(edge_vec, axis=-1, keepdims=True)
        rhat = edge_vec / (r + self.eps)

        weights = MLP(
            (self.hidden_dim, 3 * self.n_channels),
            nonlinearity='raw_swish',
            scalar_mlp_std=self.scalar_mlp_std,
            name='chiral_radial_mlp',
        )(edge_embedded)

        weights = weights.reshape(-1, 3, self.n_channels)
        weights = weights * valid[:, None, None]

        A_edge = weights[:, 0, :, None] * rhat[:, None, :]
        B_edge = weights[:, 1, :, None] * rhat[:, None, :]
        C_edge = weights[:, 2, :, None] * rhat[:, None, :]

        A_node = jax.ops.segment_sum(A_edge, edge_dst, num_segments=n_nodes)
        B_node = jax.ops.segment_sum(B_edge, edge_dst, num_segments=n_nodes)
        C_node = jax.ops.segment_sum(C_edge, edge_dst, num_segments=n_nodes)

        degree = jax.ops.segment_sum(valid, edge_dst, num_segments=n_nodes)
        degree = 0.5 * (
            degree + 1.0 + jnp.sqrt((degree - 1.0) ** 2 + self.eps ** 2)
        )
        norm = degree[:, None, None]

        A_node = A_node / norm
        B_node = B_node / norm
        C_node = C_node / norm

        # p = A dot (B cross C), shape: (N, C)
        p = jnp.sum(A_node * jnp.cross(B_node, C_node, axis=-1), axis=-1)

        gain = self.param(
            'gain',
            nn.initializers.constant(self.init_gain),
            (self.n_channels,),
        )
        p = p * gain[None, :]

        return e3nn.IrrepsArray(
            e3nn.Irreps(f'{self.n_channels}x0o'),
            p,
        )

class LocalOutOfPlanePseudoscalar(nn.Module):
    n_channels: int = 16
    hidden_dim: int = 64
    eps: float = 1.0e-8
    scalar_mlp_std: float = 4.0
    soft_normalize: bool = False

    @nn.compact
    def __call__(
        self,
        edge_vec: Array,        # (E, 3), safe_dR / shifts
        edge_dst: Array,        # (E,)
        edge_embedded: Array,   # (E, num_basis)
        edge_weight: Array,     # (E, 1), binary mask or smooth cutoff weight
        n_nodes: int,
    ) -> Array:

        valid = jnp.squeeze(edge_weight, axis=-1).astype(edge_vec.dtype)

        r = jnp.linalg.norm(edge_vec, axis=-1, keepdims=True)
        rhat = edge_vec / (r + self.eps)

        weights = MLP(
            (self.hidden_dim, 3 * self.n_channels),
            nonlinearity='raw_swish',
            scalar_mlp_std=self.scalar_mlp_std,
            name='oop_radial_mlp',
        )(edge_embedded)
        weights = weights.reshape(-1, 3, self.n_channels)
        weights = weights * valid[:, None, None]

        A_edge = weights[:, 0, :, None] * rhat[:, None, :]
        B_edge = weights[:, 1, :, None] * rhat[:, None, :]
        C_edge = weights[:, 2, :, None] * rhat[:, None, :]

        A_node = jax.ops.segment_sum(A_edge, edge_dst, num_segments=n_nodes)
        B_node = jax.ops.segment_sum(B_edge, edge_dst, num_segments=n_nodes)
        C_node = jax.ops.segment_sum(C_edge, edge_dst, num_segments=n_nodes)

        degree = jax.ops.segment_sum(valid, edge_dst, num_segments=n_nodes)
        degree = 0.5 * (
            degree + 1.0 + jnp.sqrt((degree - 1.0) ** 2 + self.eps ** 2)
        )
        degree = degree[:, None, None]
        A_node = A_node / degree
        B_node = B_node / degree
        C_node = C_node / degree

        normal_node = jnp.cross(B_node, C_node, axis=-1)  # axial (1e)
        if self.soft_normalize:
            nn2 = jnp.sum(normal_node ** 2, axis=-1, keepdims=True)
            normal_node = normal_node / jnp.sqrt(nn2 + self.eps)

        q_perp = jnp.sum(A_node * normal_node, axis=-1)
        return q_perp

class FullyConnectedTensorProductE3nn(nn.Module):
  irreps_out: Irreps
  irreps_in1: Optional[Irreps] = None
  irreps_in2: Optional[Irreps] = None

  @nn.compact
  def __call__(self, x1: IrrepsArray, x2: IrrepsArray, **kwargs) -> IrrepsArray:
    irreps_out = Irreps(self.irreps_out)
    irreps_in1 = (
        Irreps(self.irreps_in1) if self.irreps_in1 is not None else None
    )
    irreps_in2 = (
        Irreps(self.irreps_in2) if self.irreps_in2 is not None else None
    )

    x1 = e3nn.as_irreps_array(x1)
    x2 = e3nn.as_irreps_array(x2)

    leading_shape = jnp.broadcast_shapes(x1.shape[:-1], x2.shape[:-1])
    x1 = x1.broadcast_to(leading_shape + (-1,))
    x2 = x2.broadcast_to(leading_shape + (-1,))

    if irreps_in1 is not None:
      x1 = x1.rechunk(irreps_in1)
    if irreps_in2 is not None:
      x2 = x2.rechunk(irreps_in2)

    x1 = x1.remove_zero_chunks().simplify()
    x2 = x2.remove_zero_chunks().simplify()

    tp = FunctionalFullyConnectedTensorProduct(
        x1.irreps, x2.irreps, irreps_out.simplify()
    )

    ws = [
        self.param(
            (
                f"w[{ins.i_in1},{ins.i_in2},{ins.i_out}] "
                f"{tp.irreps_in1[ins.i_in1]},"
                f"{tp.irreps_in2[ins.i_in2]},{tp.irreps_out[ins.i_out]}"
            ),
            nn.initializers.normal(stddev=ins.weight_std),
            ins.path_shape,
        )
        for ins in tp.instructions
    ]

    f = lambda x1, x2: tp.left_right(ws, x1, x2, **kwargs)

    for _ in range(len(leading_shape)):
      f = e3nn.utils.vmap(f)

    output = f(x1, x2)
    return output.rechunk(irreps_out)

class Linear(nn.Module):
  """Flax module of an equivariant linear layer."""

  irreps_out: Irreps
  irreps_in: Optional[Irreps] = None
  zero_init: bool = False

  @nn.compact
  def __call__(self, x: IrrepsArray) -> IrrepsArray:
    irreps_out = Irreps(self.irreps_out)
    irreps_in = Irreps(self.irreps_in) if self.irreps_in is not None else None

    if self.irreps_in is None and not isinstance(x, IrrepsArray):
      raise ValueError(
          "the input of Linear must be an IrrepsArray, or "
          "`irreps_in` must be specified"
      )

    if irreps_in is not None:
      x = IrrepsArray(irreps_in, x)

    x = x.remove_zero_chunks().simplify()

    lin = e3nn.FunctionalLinear(x.irreps, irreps_out, instructions=None, biases=None)
    parameter_initializer = (
        nn.initializers.zeros_init()
        if self.zero_init
        else None
    )

    w = [
        self.param(  # pylint:disable=g-long-ternary
            f"b[{ins.i_out}] {lin.irreps_out[ins.i_out]}",
            (
                parameter_initializer
                if parameter_initializer is not None
                else nn.initializers.normal(stddev=ins.weight_std)
            ),
            ins.path_shape,
        )
        if ins.i_in == -1
        else self.param(
            f"w[{ins.i_in},{ins.i_out}] {lin.irreps_in[ins.i_in]},"
            f"{lin.irreps_out[ins.i_out]}",
            (
                parameter_initializer
                if parameter_initializer is not None
                else nn.initializers.normal(stddev=ins.weight_std)
            ),
            ins.path_shape,
        )
        for ins in lin.instructions
    ]

    f = lambda x: lin(w, x)
    for _ in range(x.ndim - 1):
      f = e3nn.utils.vmap(f)
    return f(x)

def tp_out_irreps_with_instructions(irreps1: e3nn.Irreps, irreps2: e3nn.Irreps, target_irreps: e3nn.Irreps
) -> Tuple[e3nn.Irreps, List]:

    mode = 'uvu'
    trainable = 'True'
    irreps_after_tp = []
    instructions = []
    for i, (mul_in1, irreps_in1) in enumerate(irreps1):
      for j, (_, irreps_in2) in enumerate(irreps2):
        for curr_irreps_out in irreps_in1 * irreps_in2:
          if curr_irreps_out in target_irreps:
            k = len(irreps_after_tp)
            irreps_after_tp += [(mul_in1, curr_irreps_out)]
            instructions += [(i, j, k, mode, trainable)]

    irreps_after_tp, p, _ = e3nn.Irreps(irreps_after_tp).sort()
    sorted_instructions = []
    for irreps_in1, irreps_in2, irreps_out, mode, trainable in instructions:
        sorted_instructions += [(
          irreps_in1,
          irreps_in2,
          p[irreps_out],
          mode,
          trainable,
      )]
    return irreps_after_tp, sorted_instructions

class NequIPConvolution(nn.Module):
  hidden_irreps: Irreps
  use_sc: bool
  nonlinearities: Union[str, Dict[str, str]]
  radial_net_nonlinearity: str = 'raw_swish'
  radial_net_n_hidden: int = 64
  radial_net_n_layers: int = 2
  num_basis: int = 8
  n_neighbors: float = 1.0
  scalar_mlp_std: float = 4.0

  @nn.compact
  def __call__(
      self,
      node_features: IrrepsArray,
      node_attributes: IrrepsArray,
      edge_sh: Array,
      edge_src: Array,
      edge_dst: Array,
      edge_embedded: Array,
  ) -> IrrepsArray:
    irreps_scalars = []
    irreps_nonscalars = []
    irreps_gate_scalars = []

    for multiplicity, irrep in self.hidden_irreps:
      if Irrep(irrep).l == 0 and tp_path_exists(
          node_features.irreps, edge_sh.irreps, irrep
      ):
        irreps_scalars += [(multiplicity, irrep)]

    irreps_scalars = Irreps(irreps_scalars)

    for multiplicity, irrep in self.hidden_irreps:
      if Irrep(irrep).l > 0 and tp_path_exists(
          node_features.irreps, edge_sh.irreps, irrep
      ):
        irreps_nonscalars += [(multiplicity, irrep)]

    irreps_nonscalars = Irreps(irreps_nonscalars)

    if tp_path_exists(node_features.irreps, edge_sh.irreps, '0e'):
      gate_scalar_irreps_type = '0e'
    else:
      gate_scalar_irreps_type = '0o'

    for multiplicity, irreps in irreps_nonscalars:
      irreps_gate_scalars += [(multiplicity, gate_scalar_irreps_type)]

    irreps_gate_scalars = Irreps(irreps_gate_scalars)

    h_out_irreps = irreps_scalars + irreps_gate_scalars + irreps_nonscalars

    if self.use_sc:
      self_connection = FullyConnectedTensorProductE3nn(
          h_out_irreps,
      )(node_features, node_attributes)

    h = node_features

    h = Linear(node_features.irreps)(h)

    edge_features = tree_map(lambda x: x[edge_src], h)

    mode = 'uvu'
    trainable = 'True'
    irreps_after_tp = []
    instructions = []

    for i, (mul_in1, irreps_in1) in enumerate(node_features.irreps):
      for j, (_, irreps_in2) in enumerate(edge_sh.irreps):
        for curr_irreps_out in irreps_in1 * irreps_in2:
          if curr_irreps_out in h_out_irreps:
            k = len(irreps_after_tp)
            irreps_after_tp += [(mul_in1, curr_irreps_out)]
            instructions += [(i, j, k, mode, trainable)]

    irreps_after_tp, p, _ = Irreps(irreps_after_tp).sort()

    sorted_instructions = []

    for irreps_in1, irreps_in2, irreps_out, mode, trainable in instructions:
      sorted_instructions += [(
          irreps_in1,
          irreps_in2,
          p[irreps_out],
          mode,
          trainable,
      )]

    tp = FunctionalTensorProduct(
        irreps_in1=edge_features.irreps,
        irreps_in2=edge_sh.irreps,
        irreps_out=irreps_after_tp,
        instructions=sorted_instructions,
    )

    n_tp_weights = 0

    for ins in tp.instructions:
      if ins.has_weight:
        n_tp_weights += prod(ins.path_shape)

    fc = MLP(
        (self.radial_net_n_hidden,) * self.radial_net_n_layers
        + (n_tp_weights,),
        self.radial_net_nonlinearity,
        use_bias=False,
        scalar_mlp_std=self.scalar_mlp_std,
    )

    weight = fc(edge_embedded)

    edge_features = e3nn.utils.vmap(tp.left_right)(
        weight, edge_features, edge_sh
    )

    edge_features = tree_map(lambda x: x.astype(h.dtype), edge_features)

    h_type = h.dtype

    e = edge_features.remove_zero_chunks().simplify()
    h = e3nn.scatter_sum(e, dst=edge_dst, output_size=h.shape[0])
    h = h.astype(h_type)

    h = h / self.n_neighbors

    h = Linear(h_out_irreps)(h)

    if self.use_sc:
      h = h + self_connection

    gate_fn = partial(
        e3nn.gate,
        even_act=get_nonlinearity_by_name(self.nonlinearities['e']),
        odd_act=get_nonlinearity_by_name(self.nonlinearities['o']),
        even_gate_act=get_nonlinearity_by_name(self.nonlinearities['e']),
        odd_gate_act=get_nonlinearity_by_name(self.nonlinearities['o']),
    )

    h = gate_fn(h)
    h = tree_map(lambda x: x.astype(h_type), h)

    return h

class DiabaticCouplingLayer(nn.Module):
  n_states: int
  pair_hidden_dim: int = 128
  scalar_mlp_std: float = 4.0
  radial_net_n_hidden: int = 64
  pair_encoder_n_layers: int = 2
  reflection_signature: tuple = ()

  @nn.compact
  def __call__(self, h_node, edge_src, edge_dst, edge_embedded, edge_weight, edge_sh=None):
    scalar_irreps = Irreps(f'{self.pair_hidden_dim}x0e')
    scalar_src = Linear(irreps_out=scalar_irreps, name='src_scalar_project')(h_node[edge_src]).array
    scalar_dst = Linear(irreps_out=scalar_irreps, name='dst_scalar_project')(h_node[edge_dst]).array

    edge_feat_even = jnp.concatenate([scalar_src, scalar_dst, edge_embedded], axis=-1)

    q_0o = None
    h_0o = h_node.filter('0o')
    if h_0o.irreps.dim > 0:
        q_0o = Linear(
            irreps_out=Irreps(f'{self.pair_hidden_dim}x0o'),
            name='q0o_project',
        )(h_0o).array  # (N, C)
        edge_feat_even = jnp.concatenate(
            [edge_feat_even, (q_0o ** 2)[edge_src]], axis=-1)  # q² is even

    pair_idx = jnp.array(jnp.triu_indices(self.n_states, k=1)).T
    n_pairs = pair_idx.shape[0]
    state_emb = self.param(
        'state_embeddings',
        nn.initializers.normal(stddev=0.02),
        (self.n_states, self.pair_hidden_dim),
    )

    encoder_features = (self.radial_net_n_hidden,) * max(0, self.pair_encoder_n_layers - 1) + (self.pair_hidden_dim,)
    edge_latent_even = MLP(
        encoder_features,
        nonlinearity='raw_swish',
        scalar_mlp_std=self.scalar_mlp_std,
        name='edge_encoder',
    )(edge_feat_even)  # (E, H)

    pair_query = jnp.concatenate([state_emb[pair_idx[:, 0]], state_emb[pair_idx[:, 1]]], axis=-1)
    pair_query = MLP(
        (self.pair_hidden_dim,) * self.pair_encoder_n_layers,
        nonlinearity='raw_swish',
        scalar_mlp_std=self.scalar_mlp_std,
        name='pair_query',
    )(pair_query)  # (n_pairs, H)

    even_readout_features = (self.radial_net_n_hidden,) * max(0, self.pair_encoder_n_layers - 1) + (1,)
    pair_readout_even = MLP(
        even_readout_features,
        nonlinearity='raw_swish',
        scalar_mlp_std=self.scalar_mlp_std,
        name='pair_readout_even',
    )
    pair_readout_odd_coeff = None
    if q_0o is not None:
        odd_coeff_features = (self.radial_net_n_hidden,) * max(0, self.pair_encoder_n_layers - 1) + (q_0o.shape[-1],)
        pair_readout_odd_coeff = MLP(
            odd_coeff_features,
            nonlinearity='raw_swish',
            scalar_mlp_std=self.scalar_mlp_std,
            name='pair_readout_odd_coeff',
        )

    mask_1d = jnp.squeeze(edge_weight, axis=-1)  # (E,)

    if self.reflection_signature:
        from peace.nn.parity import ElectronicParity
        layout = ElectronicParity(self.reflection_signature)
        triu = jnp.triu_indices(self.n_states, k=1)
        even_params = jnp.zeros((n_pairs,), dtype=edge_latent_even.dtype)
        odd_params = jnp.zeros_like(even_params)

        def pair_features(query):
            return jnp.concatenate([edge_latent_even, jnp.broadcast_to(
                query, (edge_latent_even.shape[0], query.shape[0]))], axis=-1)

        if layout.even_pairs:
            idx = jnp.asarray(layout.even_pairs)
            values = jax.vmap(lambda query: jnp.sum(
                pair_readout_even(pair_features(query)).squeeze(-1) * mask_1d))(pair_query[idx])
            even_params = even_params.at[idx].set(values)
        if layout.odd_pairs and q_0o is not None:
            idx = jnp.asarray(layout.odd_pairs)
            values = jax.vmap(lambda query: jnp.sum(jnp.sum(
                q_0o[edge_src] * pair_readout_odd_coeff(pair_features(query)), axis=-1)
                * mask_1d))(pair_query[idx])
            odd_params = odd_params.at[idx].set(values)
        even = jnp.zeros((self.n_states, self.n_states), dtype=even_params.dtype).at[triu].set(even_params)
        odd = jnp.zeros_like(even).at[triu].set(odd_params)
        return even + even.T, odd + odd.T

    def readout_single(p, query):
        pair_edge_feat = jnp.concatenate(
            [edge_latent_even, jnp.broadcast_to(query, (edge_latent_even.shape[0], query.shape[0]))],
            axis=-1)
        H_even = pair_readout_even(pair_edge_feat).squeeze(-1)  # (E,)

        # Odd contribution: q_0o · coeff(even_feat).
        if q_0o is not None:
            q_0o_edge = q_0o[edge_src]  # (E, C)
            odd_coeff = pair_readout_odd_coeff(pair_edge_feat)  # (E, C)
            H_odd = jnp.sum(q_0o_edge * odd_coeff, axis=-1)  # (E,)
        else:
            H_odd = jnp.zeros_like(H_even)

        H_even = H_even * mask_1d
        H_odd = H_odd * mask_1d
        return jnp.stack([jnp.sum(H_even), jnp.sum(H_odd)])

    parity_components = jax.vmap(readout_single)(
        jnp.arange(n_pairs), pair_query)
    even_params = parity_components[:, 0]
    odd_params = parity_components[:, 1]

    triu = jnp.triu_indices(self.n_states, k=1)
    H_even = jnp.zeros(
        (self.n_states, self.n_states), dtype=even_params.dtype)
    H_odd = jnp.zeros_like(H_even)
    H_even = H_even.at[triu].set(even_params)
    H_odd = H_odd.at[triu].set(odd_params)
    return H_even + H_even.T, H_odd + H_odd.T