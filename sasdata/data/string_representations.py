from typing import Dict, Any

def format_parameters(parameters: Dict[str, Any]) -> str:
    """ Formatting of parameters, abstracted so that we have a uniform way of rendering them"""
    return ",".join([f"{key}={parameters[key]}" for key in parameters])