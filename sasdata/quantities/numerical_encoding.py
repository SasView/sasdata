import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, coo_array, csr_array, csc_array

import base64
import struct


def numerical_encode(obj: int | float | np.ndarray | coo_matrix | coo_array | csr_matrix | csr_array | csc_matrix | csc_array):

    if isinstance(obj, int):
        return {"type": "int",
                "value": obj}

    elif isinstance(obj, float):
        return {"type": "float",
                "value": base64.b64encode(bytearray(struct.pack('d', obj))).decode("utf-8")}

    elif isinstance(obj, np.ndarray):
        return {
            "type": "numpy",
            "value": base64.b64encode(obj.tobytes()).decode("utf-8"),
            "dtype": obj.dtype.str,
            "shape": list(obj.shape)
        }

    elif isinstance(obj, (coo_matrix, coo_array, csr_matrix, csr_array, csc_matrix, csc_array)):

        output = {
            "type": obj.__class__.__name__, # not robust to name changes, but more concise
            "dtype": obj.dtype.str,
            "shape": list(obj.shape)
        }

        if isinstance(obj, (coo_array, coo_matrix)):

            output["data"] = numerical_encode(obj.data)
            output["coords"] = [numerical_encode(coord) for coord in obj.coords]


        elif isinstance(obj, (csr_array, csr_matrix)):
            pass


        elif isinstance(obj, (csc_array, csc_matrix)):

            pass


        return output

    else:
        raise TypeError(f"Cannot serialise object of type: {type(obj)}")

def numerical_decode(data: dict[str, str | int | list[int]]) -> int | float | np.ndarray | coo_matrix | coo_array | csr_matrix | csr_array | csc_matrix | csc_array:
    obj_type = data["type"]

    match obj_type:
        case "int":
            return int(data["value"])

        case "float":
            return struct.unpack('d', base64.b64decode(data["value"]))[0]

        case "numpy":
            value = base64.b64decode(data["value"])
            dtype = np.dtype(data["dtype"])
            shape = tuple(data["shape"])
            return np.frombuffer(value, dtype=dtype).reshape(*shape)

        case _:
            raise ValueError(f"Cannot decode objects of type '{obj_type}'")

