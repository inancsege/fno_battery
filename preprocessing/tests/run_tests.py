#!/usr/bin/env python

import os
import sys
import pytest

# Add the parent directory to the Python path to enable relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

if __name__ == "__main__":
    # Run the tests
    sys.exit(pytest.main(["-xvs", os.path.dirname(__file__)])) 