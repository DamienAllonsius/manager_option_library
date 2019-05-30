import sys

# only unit testing for the moment
if __name__ == "__main__":
    from tests.test_Q import *
    from tests.test_Option import *
    from tests.test_Tree import *
    from tests.test_Node import *
    import unittest

    del sys.argv[1:]
    unittest.main()