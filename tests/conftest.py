# tests/conftest.py
import sys
from pathlib import Path

# Append the directory containing server.py to sys.path
project_root = str(Path(__file__).parents[1])  # Adjust the number of parents based on actual path to project root
sys.path.insert(0, project_root)

from server import app  # Import the Flask app

import pytest

@pytest.fixture(scope='module')
def client():
    app.config.update({
        "TESTING": True,
    })
    with app.test_client() as testing_client:
        yield testing_client
