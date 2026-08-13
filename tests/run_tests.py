import unittest
import sys
import os

if __name__ == "__main__":
    # Ensure the project root is in the path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    # Unit tests build small synthetic CSV fixtures that will never match the
    # real-dataset integrity fingerprints in cissn/data/registry.py.
    os.environ.setdefault('CISSN_SKIP_DATA_VERIFY', '1')

    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
