"""Small NumPy-facing prediction API, independent of a dynamics driver."""
import numpy as np
from peace import units
from peace.graph import validate_atomic_numbers
from peace.nn.io import read_model_config

def load_calculator(config, params, atomic_numbers, **kwargs):
    """Load the SHARC-facing backend selected by explicit include_soc."""
    from .calculator import PEACECalculator
    from .soc import SOCCalculator
    cfg = read_model_config(config)
    numbers = validate_atomic_numbers(atomic_numbers)
    cls = SOCCalculator if cfg.include_soc else PEACECalculator
    return cls(str(config), str(params), numbers, **kwargs)

class Calculator:
    def __init__(self, config, params):
        import jax
        jax.config.update("jax_enable_x64", True)
        self.config_path = str(config)
        self.params_path = str(params)
        self.config = read_model_config(config).to_dict()
        self._backend = None
        self._numbers = None

    def predict(self, atomic_numbers, positions):
        numbers = validate_atomic_numbers(atomic_numbers)
        xyz = np.asarray(positions, dtype=np.float64)
        if xyz.shape != (len(numbers), 3) or not np.isfinite(xyz).all():
            raise ValueError("positions must be a finite (n_atoms, 3) Angstrom array")
        if self._backend is None or not np.array_equal(numbers, self._numbers):
            self._backend = load_calculator(self.config_path, self.params_path, numbers,
                                            warn_on_state_mixing=False)
            self._numbers = numbers.copy()
        backend = self._backend
        backend.reset_phase_tracking()
        result = backend.calculate(xyz / units.Bohr)
        qm = backend.get_qm(result)
        pairs = (tuple(zip(*backend.nac_idx)) if self.config["include_soc"]
                 else tuple(zip(*np.triu_indices(backend.n_total_states, 1))))
        output = {
            "energies": np.asarray(result["energy"]) * units.Hartree,
            "forces": -np.asarray(result["gradients"]) * (units.Hartree / units.Bohr),
            "nacs": np.asarray(result["nacs"]) / units.Bohr,
            "snacs": np.asarray(result["snacs"]) * (units.Hartree / units.Bohr),
            "nac_pairs": np.asarray(pairs, dtype=np.int32).reshape(-1, 2),
            "hamiltonian": np.asarray(qm["h"]) * units.Hartree,
        }
        if self.config["include_soc"]:
            output["soc"] = np.asarray(result["soc"]) * units.Hartree
        else:
            output["hamiltonian"] = output["hamiltonian"].real
        return output

    __call__ = predict
