import sys

import parfun.py_list as py_list
from parfun.about import __version__
from parfun.decorators import parallel, parfun
from parfun.entry_point import get_parallel_backend, set_parallel_backend, set_parallel_backend_context
from parfun.partition.api import all_arguments, multiple_arguments, per_argument


__all__ = (
    "__version__",
    "parallel", "parfun",
    "get_parallel_backend", "set_parallel_backend", "set_parallel_backend_context",
    "all_arguments", "multiple_arguments", "per_argument",
    "py_list",
)


def __getattr__(name: str):
    # Only load the dataframe and numpy modules when requested, as these have optional dependencies

    if name == "dataframe":
        import parfun.dataframe as dataframe
        sys.modules[__name__ + ".dataframe"] = dataframe
        return dataframe

    if name == "numpy":
        import parfun.numpy as numpy
        sys.modules[__name__ + ".numpy"] = numpy
        return numpy

    raise AttributeError(f"module {__name__} has no attribute {name}")
