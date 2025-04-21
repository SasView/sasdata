import requests

session = {
    "title": "Example Session",
    "datasets": [
        {
            "name": "Dataset 1",
            "metadata": {
                "title": "Metadata 1",
                "run": 1,
                "description": "test",
                "instrument": {},
                "process": {},
                "sample": {},
            },
            "data_contents": [
                {
                    "value": 0,
                    "variance": 0,
                    "units": "no",
                    "hash": 0,
                    "label": "Quantity 1",
                    "history": {"operation_tree": {}, "references": []},
                }
            ],
        },
        {
            "name": "Dataset 2",
            "metadata": {
                "title": "Metadata 2",
                "run": 2,
                "description": "test",
                "instrument": {},
                "process": {},
                "sample": {},
            },
            "data_contents": [
                {
                    "label": "Quantity 2",
                    "value": {"array_contents": [0, 0, 0, 0], "shape": (2, 2)},
                    "variance": {"array_contents": [0, 0, 0, 0], "shape": (2, 2)},
                    "units": "none",
                    "hash": 0,
                    "history": {
                        "operation_tree": {
                            "operation": "neg",
                            "parameters": {
                                "a": {
                                    "operation": "mul",
                                    "parameters": {
                                        "a": {
                                            "operation": "constant",
                                            "parameters": {
                                                "value": {"type": "int", "value": 7}
                                            },
                                        },
                                        "b": {
                                            "operation": "variable",
                                            "parameters": {
                                                "hash_value": 111,
                                                "name": "x",
                                            },
                                        },
                                    },
                                },
                            },
                        },
                        "references": [
                            {
                                "value": 5,
                                "variance": 0,
                                "units": "none",
                                "hash": 111,
                                "history": {},
                            }
                        ],
                    },
                }
            ],
        },
    ],
    "is_public": True,
}

url = "http://127.0.0.1:8000/v1/data/session/"

requests.request("POST", url, json=session)
