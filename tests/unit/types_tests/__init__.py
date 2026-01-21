# This directory unfortunately needs to be named "types_tests"
# Otherwise we hit import naming collisions with the "types" module
# See:
# ImportError: Failed to import test module: types.data_test
# Traceback (most recent call last):
#  File "/opt/conda/envs/gnn/lib/python3.9/unittest/loader.py", line 436, in _find_test_path
#    module = self._get_module_from_name(name)
#  File "/opt/conda/envs/gnn/lib/python3.9/unittest/loader.py", line 377, in _get_module_from_name
#    __import__(name)
# ModuleNotFoundError: No module named 'types.data_test'; 'types' is not a package
