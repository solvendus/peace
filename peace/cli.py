import argparse
from pathlib import Path
import numpy as np
from peace.elements import atomic_numbers

def read_xyz(path):
    lines = Path(path).read_text().splitlines()
    if len(lines) < 2:
        raise ValueError("XYZ requires an atom count and comment line")
    try:
        count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError("XYZ first line must be an integer atom count") from exc
    if count < 1 or len(lines) < count + 2 or any(x.strip() for x in lines[count + 2:]):
        raise ValueError("Provide exactly one complete XYZ geometry")
    numbers, positions = [], []
    for line in lines[2:count+2]:
        fields = line.split()
        if len(fields) != 4 or fields[0] not in atomic_numbers:
            raise ValueError("Each XYZ row must be: element x y z")
        numbers.append(atomic_numbers[fields[0]])
        positions.append([float(x) for x in fields[1:]])
    xyz = np.asarray(positions, dtype=np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError("XYZ coordinates must be finite")
    return np.asarray(numbers, dtype=np.int32), xyz

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Exported model_config.yaml")
    parser.add_argument("--params", required=True, help="Trusted Flax parameter pickle")
    parser.add_argument("--xyz", required=True, help="One geometry, coordinates in Angstrom")
    parser.add_argument("--output", default="prediction.npz")
    args = parser.parse_args(argv)
    import jax
    jax.config.update("jax_enable_x64", True)
    from peace import Calculator
    numbers, positions = read_xyz(args.xyz)
    result = Calculator(args.config, args.params).predict(numbers, positions)
    out = Path(args.output)
    if out.suffix != ".npz":
        parser.error("--output must end in .npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **result)
    print(f"Saved {out}; energy=eV, force/SNAC=eV/Angstrom, NAC=1/Angstrom")
    return 0
