import numpy as np

import base64
import struct


def numerical_encode(obj: int | float | np.ndarray):

    if isinstance(obj, int):
        return {"type": "int",
                "value": obj}

    elif isinstance(obj, float):
        return {"type": "float",
                "value": base64.b64encode(bytearray(struct.pack('d', obj)))}

    elif isinstance(obj, np.ndarray):
        return {
            "type": "numpy",
            "value": base64.b64encode(obj.tobytes()),
            "dtype": obj.dtype.str,
            "shape": list(obj.shape)
        }

    else:
        raise TypeError(f"Cannot serialise object of type: {type(obj)}")

def numerical_decode(data: dict[str, str | int | list[int]]) -> int | float | np.ndarray:
    match data["type"]:
        case "int":
            return int(data["value"])

        case "float":
            return struct.unpack('d', base64.b64decode(data["value"]))[0]

        case "numpy":
            value = base64.b64decode(data["value"])
            dtype = np.dtype(data["dtype"])
            shape = tuple(data["shape"])
            return np.frombuffer(value, dtype=dtype).reshape(*shape)
