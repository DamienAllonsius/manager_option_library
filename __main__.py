import sys

# only unit testing for the moment
if __name__ == "__main__":
    from tests.test_policy_manager import *
    import unittest

    del sys.argv[1:]
    unittest.main()