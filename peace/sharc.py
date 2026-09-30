#!/usr/bin/env python3

import datetime
import os
import shutil
import sys
from io import TextIOWrapper
from pathlib import Path
import numpy as np

_root = Path(__file__).resolve().parents[1]
if (_root / "peace").is_dir():
    sys.path.insert(0, str(_root))
if os.environ.get("SHARC"):
    _sharc = Path(os.environ["SHARC"]).expanduser().resolve()
    for _candidate in (_sharc, _sharc / "lib", _sharc.parent / "lib"):
        if (_candidate / "SHARC_FAST.py").is_file():
            sys.path.insert(0, str(_candidate))
            break
try:
    from SHARC_FAST import SHARC_FAST
    from utils import expand_path, link, question
except ModuleNotFoundError as exc:
    if exc.name in ("SHARC_FAST", "SHARC_INTERFACE", "utils", "qmin", "qmout"):
        raise ImportError("Set SHARC or PYTHONPATH to the SHARC 4 Python library containing SHARC_FAST.py.") from exc
    raise ImportError(
        f"SHARC 4 requires the missing dependency {exc.name!r}; use a working SHARC Python environment."
    ) from exc

from peace.elements import atomic_numbers
from peace.overlap import validate_state_overlap

class SHARC_PEACE(SHARC_FAST):
    _version = "1.0"
    _versiondate = datetime.datetime(2026, 9, 22)
    _authors = "R. Gao"
    _changelogstring = "SHARC 4 interface to PEACE electronic-property calculator"
    _name = "PEACE"
    _description = "PEACE electronic-property interface"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.QMin.template.update({"modelpath": "", "paramspath": "", "paddingstates": []})
        self.QMin.template.types.update({"modelpath": str, "paramspath": str, "paddingstates": list})
        self.peace_calc = None
        self._prediction = None
        self._overlap_diagnostics = None
        self._has_previous_geometry = False

    def get_features(self, KEYSTROKES=None):
        return {"h", "grad", "nacdr", "dm", "overlap", "soc"}

    def read_template(self, template_filename="PEACE.template"):
        super().read_template(template_filename, kw_whitelist={"modelpath", "paramspath", "paddingstates"})
        parent = Path(template_filename).resolve().parent
        for key in ("modelpath", "paramspath"):
            value = self.QMin.template[key]
            if not value:
                raise ValueError(f"Missing PEACE.template key: {key}")
            path = Path(os.path.expandvars(os.path.expanduser(value)))
            if not path.is_absolute():
                path = parent / path
            if not path.is_file():
                raise FileNotFoundError(f"{key} does not point to a file: {path}")
            self.QMin.template[key] = str(path.resolve())

    def read_resources(self, resources_filename="PEACE.resources"):
        self._read_resources = True

    def setup_interface(self):
        super().setup_interface()
        import jax
        jax.config.update("jax_enable_x64", True)
        from peace.calculator import load_calculator
        numbers = np.asarray([atomic_numbers[s] for s in self.QMin.molecule["elements"]], dtype=np.int32)
        self.peace_calc = load_calculator(self.QMin.template["modelpath"], self.QMin.template["paramspath"], numbers)
        cfg = self.peace_calc.model_configs
        roots = list(int(x) for x in self.QMin.molecule["states"])
        expected = [cfg["n_singlets"], 0, cfg["n_triplets"]] if cfg["include_soc"] else [cfg["n_states"]]
        roots += [0] * max(0, len(expected) - len(roots))
        expected += [0] * max(0, len(roots) - len(expected))
        if roots != expected:
            raise ValueError(f"SHARC roots {roots} do not match model roots {expected}")
        self._n_atoms = len(numbers)
        self._n_total_states = self.peace_calc.n_total_states
        self.peace_calc.reset_phase_tracking()
        self._has_previous_geometry = False

    def run(self):
        if self.peace_calc is None:
            raise RuntimeError("setup_interface() must run before evaluation")
        if self.QMin.requests.get("init"):
            self.peace_calc.reset_phase_tracking()
            self._has_previous_geometry = False
        if self.QMin.requests.get("overlap") and not self._has_previous_geometry:
            raise RuntimeError(
                "Electronic overlaps require the previous geometry in the same "
                "persistent SHARC_PEACE instance; process restarts cannot restore it."
            )
        self._prediction = self.peace_calc.get_qm(self.peace_calc.calculate(self.QMin.coords["coords"]))
        self._validate_prediction(self._prediction)
        self._has_previous_geometry = True

    def create_restart_files(self):
        pass

    @staticmethod
    def version() -> str:
        return SHARC_PEACE._version

    @staticmethod
    def versiondate() -> datetime.datetime:
        return SHARC_PEACE._versiondate

    @staticmethod
    def authors() -> str:
        return SHARC_PEACE._authors

    @staticmethod
    def changelogstring() -> str:
        return SHARC_PEACE._changelogstring

    @staticmethod
    def name() -> str:
        return SHARC_PEACE._name

    @staticmethod
    def description() -> str:
        return SHARC_PEACE._description

    @staticmethod
    def about() -> str:
        return f"{SHARC_PEACE._name}\n{SHARC_PEACE._description}"

    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'PEACE interface setup': ^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")
        if os.path.isfile("PEACE.template"):
            self.log.info("Found PEACE.template in current directory")
            if question(
                "Use this template file?",
                bool,
                KEYSTROKES=KEYSTROKES,
                default=True,
            ):
                self._template_file = "PEACE.template"
        else:
            self.log.info("Specify a path to a PEACE template file.")
            while not os.path.isfile(
                template_file := question(
                    "Template path:",
                    str,
                    KEYSTROKES=KEYSTROKES,
                )
            ):
                self.log.info(f"File {template_file} does not exist!")
            self._template_file = template_file

        return INFOS

    def prepare(self, INFOS: dict, dir_path: str):
        create_file = link if INFOS["link_files"] else shutil.copy
        if hasattr(self, "_template_file") and self._template_file:
            create_file(
                expand_path(self._template_file),
                os.path.join(dir_path, "PEACE.template"),
            )
        else:
            self.log.error("Template file not found during prepare!")
            raise ValueError("Missing template file")

    def _validate_prediction(self, prediction):
        nstates = self._n_total_states
        natoms = self._n_atoms
        expected_shapes = {
            "h": (nstates, nstates),
            "grad": (nstates, natoms, 3),
            "nacdr": (nstates, nstates, natoms, 3),
            "overlap": (nstates, nstates),
        }
        for key, expected_shape in expected_shapes.items():
            array = np.asarray(prediction[key])
            if array.shape != expected_shape:
                raise ValueError(
                    f"PEACE {key} has shape {array.shape}; "
                    f"expected {expected_shape}"
                )
            if not np.all(np.isfinite(array)):
                raise FloatingPointError(
                    f"PEACE returned non-finite values in {key}"
                )
        hamiltonian = np.asarray(prediction["h"])
        if not np.allclose(
            hamiltonian,
            hamiltonian.conj().T,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError("PEACE Hamiltonian is not Hermitian")
        nacdr = np.asarray(prediction["nacdr"])
        if not np.allclose(
            nacdr + np.swapaxes(nacdr, 0, 1),
            0.0,
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise ValueError("PEACE derivative couplings are not antisymmetric")
        requests = getattr(getattr(self, "QMin", None), "requests", {})
        self._overlap_diagnostics = validate_state_overlap(
            prediction["overlap"],
            require_full_rank=bool(requests.get("overlap", False)),
        )

    def getQMout(self):
        if self._prediction is None:
            raise RuntimeError("run() must be called before getQMout()")
        requests = set()
        for key, val in self.QMin.requests.items():
            if not val:
                continue
            requests.add(key)
        self.log.debug("Allocate space in QMout object")
        self.QMout.allocate(
            states=self.QMin.molecule["states"],
            natom=self.QMin.molecule["natom"],
            npc=self.QMin.molecule["npc"],
            requests=requests,
        )
        if self.QMin.requests["h"]:
            self.QMout["h"] = np.asarray(self._prediction["h"])
        if self.QMin.requests["grad"]:
            self.QMout["grad"] = np.asarray(self._prediction["grad"])
        if self.QMin.requests["nacdr"]:
            self.QMout["nacdr"] = np.asarray(self._prediction["nacdr"])
        if self.QMin.requests["overlap"]:
            self.QMout["overlap"] = np.asarray(
                self._prediction["overlap"], dtype=np.complex128
            )
        if self.QMin.requests["dm"]:
            nstates = self._n_total_states
            self.QMout["dm"] = np.zeros(
                (3, nstates, nstates),
                dtype=np.complex128,
            )
        return self.QMout

def main():
    SHARC_PEACE().main()

if __name__ == "__main__":
    main()