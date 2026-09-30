import numpy as np
from peace.elements import atomic_numbers as symbol_numbers

def free_boundary_neighbor_list(positions, cutoff):
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    return np.where((distances < cutoff) & (distances > 0.0))

def validate_atomic_numbers(values):
    raw = np.asarray(values)
    if raw.ndim != 1 or not raw.size:
        raise ValueError("atomic_numbers must be a nonempty one-dimensional array")
    if not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.bool_):
        raise ValueError("atomic_numbers must contain integers")
    if np.any((raw < 1) | (raw > 118)):
        raise ValueError("atomic_numbers must lie between 1 and 118")
    return raw.astype(np.int32)

class AtomicNumberTable:
    """Preserve the species-channel ordering stored in model_config.mapping."""
    def __init__(self, mapping):
        self.mapping_type = mapping
        self.zs = sorted(mapping)

    def __len__(self):
        return len(self.zs)

    def mapping(self, numbers):
        numbers = validate_atomic_numbers(numbers)
        missing = sorted(set(numbers.tolist()) - set(self.mapping_type))
        if missing:
            raise ValueError(f"Elements absent from model config: {missing}")
        return np.asarray([self.mapping_type[int(z)] for z in numbers], dtype=np.int32)

    @classmethod
    def from_dict(cls, mapping):
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("config.mapping must specify each element's species channel")
        converted = {}
        for key, channel in mapping.items():
            if isinstance(key, str) and key in symbol_numbers:
                z = symbol_numbers[key]
            else:
                try:
                    z = int(key)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Unknown element in config.mapping: {key!r}") from exc
                if str(key) != str(z) and not isinstance(key, (int, np.integer)):
                    raise ValueError(f"Invalid atomic number in config.mapping: {key!r}")
            if z < 1 or z > 118 or z in converted:
                raise ValueError("config.mapping contains an invalid or duplicated element")
            if type(channel) is not int:
                raise ValueError("Species channels must be integers")
            converted[z] = channel
        if sorted(converted.values()) != list(range(len(converted))):
            raise ValueError("Species channels must be unique and contiguous, starting at zero")
        return cls(converted)