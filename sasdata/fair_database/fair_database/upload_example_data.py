import logging
import os
from glob import glob

import requests

EXAMPLE_DATA_DIR = os.environ.get("EXAMPLE_DATA_DIR", "../../example_data")


def parse_1D():
    dir_1d = os.path.join(EXAMPLE_DATA_DIR, "1d_data")
    if not os.path.isdir(dir_1d):
        logging.error(f"1D Data directory not found at: {dir_1d}")
        return
    for file_path in glob(os.path.join(dir_1d, "*")):
        upload_file(file_path)


def parse_2D():
    dir_2d = os.path.join(EXAMPLE_DATA_DIR, "2d_data")
    if not os.path.isdir(dir_2d):
        logging.error(f"2D Data directory not found at: {dir_2d}")
        return
    for file_path in glob(os.path.join(dir_2d, "*")):
        upload_file(file_path)


def parse_sesans():
    sesans_dir = os.path.join(EXAMPLE_DATA_DIR, "sesans_data")
    if not os.path.isdir(sesans_dir):
        logging.error(f"Sesans Data directory not found at: {sesans_dir}")
        return
    for file_path in glob(os.path.join(sesans_dir, "*")):
        upload_file(file_path)


def upload_file(file_path):
    url = "http://localhost:8000/v1/data/file/"
    file = open(file_path, "rb")
    requests.request("POST", url, data={"is_public": True}, files={"file": file})


if __name__ == "__main__":
    parse_1D()
    parse_2D()
    parse_sesans()
