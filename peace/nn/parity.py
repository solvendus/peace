from dataclasses import dataclass
from numbers import Integral
from peace.state import validate_latent_n_states, resolve_state_dimensions

def resolve_parity(n_states, counts=None, signature=None, *, n_observed_states=None):
    n_states = validate_latent_n_states(n_states, context="Electronic parity")
    if n_observed_states is not None:
        n_observed_states, n_states = resolve_state_dimensions(n_observed_states, n_states)
    if counts is not None:
        counts = tuple(counts)
        if (len(counts) != 2 or any(isinstance(x, bool) or not isinstance(x, Integral)
                                   or x < 0 for x in counts)
                or sum(counts) != n_states):
            raise ValueError('electronic_parity must be two nonnegative integers summing to n_states')
    if signature is not None and len(signature):
        signature = tuple(signature)
        if (len(signature) != n_states or any(isinstance(x, bool)
                or not isinstance(x, Integral) or x not in (-1, 1) for x in signature)):
            raise ValueError('reflection_signature must contain exactly n_states integer +1/-1 values')
        observed = (signature.count(1), signature.count(-1))
        if counts is not None and counts != observed:
            raise ValueError('electronic_parity and reflection_signature disagree')
        return signature, observed
    if counts is None:
        counts = ((n_observed_states, n_states - n_observed_states)
                  if n_observed_states is not None and n_states > n_observed_states
                  else (n_states - 1, 1))
    return (1,) * counts[0] + (-1,) * counts[1], counts


@dataclass(frozen=True)
class ElectronicParity:
    signature: tuple

    def __post_init__(self):
        signature, _ = resolve_parity(len(self.signature), signature=self.signature)
        object.__setattr__(self, 'signature', signature)

    @property
    def pairs(self):
        return tuple((i, j) for i in range(len(self.signature))
                     for j in range(i + 1, len(self.signature)))

    @property
    def even_pairs(self):
        return tuple(k for k, (i, j) in enumerate(self.pairs)
                     if self.signature[i] == self.signature[j])

    @property
    def odd_pairs(self):
        return tuple(k for k, (i, j) in enumerate(self.pairs)
                     if self.signature[i] != self.signature[j])

    def summary(self):
        even, odd = len(self.even_pairs), len(self.odd_pairs)
        return {'signature': list(self.signature),
                'electronic_parity': [self.signature.count(1), self.signature.count(-1)],
                'hamiltonian_irreps': f'{len(self.signature) + even}x0e + {odd}x0o',
                'connection_irreps': f'{even}x1o + {odd}x1e'}