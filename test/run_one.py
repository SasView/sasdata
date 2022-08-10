#!/usr/bin/env python
import os
import sys
import xmlrunner
import unittest
import importlib.util
from os.path import abspath, split as splitpath

import logging
logger = logging.getLogger(__name__)
if not logger.root.handlers:
    import logging.config
    LOGGER_CONFIG_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'logging.ini')
    logging.config.fileConfig(LOGGER_CONFIG_FILE, disable_existing_loggers=False)

if len(sys.argv) < 2:
    logger.error("Use %s <filename to test>", sys.argv[0])
    sys.exit(-1)

test_path_abs = abspath(sys.argv[1])
test_path, f_name_full = splitpath(abspath(sys.argv[1]))
f_name = f_name_full.split('.')[0]
#print("=== testing:", sys.argv[1])
sys.argv = [sys.argv[0]]
os.chdir(test_path)
sys.path.insert(0, test_path)
#print("\n".join(sys.path))
foo = importlib.util.spec_from_file_location('tests', f_name_full)
test = importlib.util.module_from_spec(foo)
sys.modules['tests'] = test
foo.loader.exec_module(test)
#print(test)
unittest.main(test, testRunner=xmlrunner.XMLTestRunner(output='logs'))
