Quick start
===========


When to use it
--------------

**Parfun** works well with tasks that are CPU-intensive and can be easily divided into *independent sub-tasks*.


.. image:: images/parallel_scatter_gather.png


Here are a few examples of tasks that can be easily parallelized:

* ✔ Filtering tasks (e.g. :py:func:`pandas.Dataframe.filter`)
* ✔ Operations that are associative or independent (e.g. matrix addition, sum ...)
* ✔ By-row data processing tasks (e.g. cleaning or normalizing input data).

Not all tasks can be easily parallelized:

* ✖ Computations on non-partitionable datasets (e.g. median computation, sorting)
* ✖ I/O intensive tasks (file loading, network communications)
* ✖ Very short (< 100ms) tasks.

  These tasks are too small for the parallelism gains to exceed the overhead of parallelization.
  Always keep in mind that parallelization is a trade-off between CPU speed and the time needed to transfer data
  between processes.


Setup and backend selection
---------------------------

First, install the ``opengris-parfun`` package from PyPI using any compatible package manager:

.. code:: bash

    pip install opengris-parfun


The above command will **only install the base package**. This is suitable for using the ``multiprocessing`` backend, however
if you wish to use an alternate computing backend, such as Scaler or Dask, or to enable Numpy's or Pandas' support,
install the `scaler`, `dask`, `numpy`, or `pandas` extras as required:

.. code:: bash

    pip install "opengris-parfun[dask,scaler,numpy,pandas]"

**Initializing the library**

Before using Parfun, the backend must be initialized. This can either be done with :py:func:`~parfun.set_parallel_backend`
or in a scoped context manager using :py:func:`~parfun.set_parallel_backend_context`.

.. literalinclude:: ../../../examples/api_usage/backend_setup.py
    :language: python
    :start-at: import parfun as pf

See :py:func:`~parfun.set_parallel_backend` for a description of the available backend options.


Your first parallel function
----------------------------

The following example does basic statistical analysis on a stock portfolio. By using Parfun, these metrics are
calculated in parallel, **splitting the computation by country**.

.. literalinclude:: ../../../examples/portfolio_metrics/main.py
    :language: python
    :start-at: from typing import List

In this example, the :py:func:`~parfun.parallel` **decorator** *parallelizes the execution of*
``relative_metrics()`` *by country*, and then reduces the results by concatenating them:

.. literalinclude:: ../../../examples/portfolio_metrics/main.py
    :language: python
    :start-at: @pf.parallel
    :end-at: def relative_metrics

We tell the parallel engine how to partition the data with the ``split`` parameter. Note that we can safely
split the calculation over countries, as the function itself processes these countries independently.

- We use :py:func:`~parfun.per_argument` to tell the parallel engine which arguments to partition on. In
  the example, we will only partition on the ``portfolio`` argument, and we don't touch the ``columns`` argument.

- We use :py:func:`~parfun.dataframe.by_group` to specify that the portfolio dataframe can be partitioned
  over its ``country`` column.

Finally, Parfun needs to know how to *combine the results* of the partitioned calls to
``relative_metrics()``. This is done with the ``combine_with`` parameter. In our example, we concatenate the
result dataframes with :py:func:`~parfun.dataframe.concat`.

Parfun handles executing the function in parallel and collection the results. We can represent this computation visually:

.. image:: images/parallel_function.png

This is a well-known parallelization architecture called **map/reduce** or **scatter/gather**.


Partitioning functions
----------------------

As seen in the example above, the ``@parallel`` decorator accepts a partitioning function (``split``).

Previously, we applied a single partitioning function (:py:func:`~parfun.dataframe.by_group`) on a
single argument. However, we can also use :py:func:`~parfun.per_argument` to apply partitioning
to multiple arguments.

.. literalinclude:: ../../../examples/api_usage/per_argument.py
    :language: python
    :start-at: from typing import List

Here we are using two partitioning functions, :py:func:`~parfun.py_list.by_chunk` and :py:func:`~parfun.dataframe.by_row`.
These split the arguments in to equally sized partitions. It's semantically equivalent to the following code:


.. code-block:: python

    size = min(len(factors), len(dataframe))
    for begin in range(0, size, PARTITION_SIZE):
        end = min(begin + PARTITION_SIZE, size)
        multiply_by_row(factors[begin:end], dataframe.iloc[begin:end])


Sometimes it might be desirable to partition all of the arguments the same way, this can be done with :py:func:`~parfun.all_arguments`:

.. literalinclude:: ../../../examples/api_usage/all_arguments.py
    :language: python
    :start-at: import pandas as pd


Combining functions
-------------------

In addition to the partitioning function, the ``@parallel`` decorator requires a combining function (``combine_with``) to
collect results and handle the reduction stage of map-reduce.

Parfun provides useful combining functions for handling :ref:`Python lists <section-lists>`,
:ref:`Numpy arrays <section-numpy>` and :ref:`Pandas dataframes <section-dataframes>`, such as:
:py:func:`~parfun.py_list.concat`, :py:func:`~parfun.numpy.concat` and :py:func:`~parfun.dataframe.concat`.


Custom partitioning and combining functions
--------------------------------------------

Parfun has many built-in partitioning and combining functions, but you can also define your own:

.. literalinclude:: ../../../examples/api_usage/custom_generators.py
    :language: python
    :start-at: from typing import Generator, Iterable, Tuple

Partitioning functions are implemented as generators, whereas combining functions accept an iterable of results and return the combined result.

Custom partitioning functions should:

1. **use the** ``yield`` **mechanism**, and not return a collection (e.g. a list). Returning a collection instead of
   using a generator will lead to deteriorated performances and higher memory usage.
2. **accept the parameters to partition, and yield these partitioned parameters as a tuple**, in the same order.


Partition size estimate
-----------------------

Parfun tries to automatically determine the optimal size for the parallelly distributed partitions.

:doc:`Read more </tutorials/implementation_details>` about how the library computes the optimal partition size.

**You can override how the library chooses the partition size to use by providing either the**
``initial_partition_size: int`` **or** ``fixed_partition_size: int`` **parameter:**

.. literalinclude:: ../../../examples/api_usage/partition_size.py
    :language: python
    :start-at: import numpy as np

.. note::

    Partition size estimation is disabled for custom partition generators.

Nested parallel function calls
------------------------------

Parfun functions can be safely called from other Parfun functions.


.. note::
    Currently, Scaler is the only backend that can run nested functions in parallel and other backends will execute the
    functions sequentially.


.. literalinclude:: ../../../examples/api_usage/nested_functions.py
    :language: python
    :start-at: import pprint


Profiling
---------

The easiest way to profile the speedup provided by a parallel function is to either use Python's `timeit` module, or the
IPython/Jupyter ``%timeit`` command.

In addition, **the ``@parallel`` decorator accepts a** ``profile: bool`` **and a** ``trace_export: Optional[str]`` **parameter** that
can be used get profiling metrics about the execution of a parallel function.

.. literalinclude:: ../../../examples/api_usage/profiling.py
    :language: python
    :start-at: from typing import List

Setting ``profile`` to ``True`` returns a performance summary with timings:

.. code-block:: console

   parallel_sum()
	total CPU execution time: 0:00:00.372508.
	compute time: 0:00:00.347168 (93.20%)
		min.: 0:00:00.001521
		max.: 0:00:00.006217
		avg.: 0:00:00.001973
	total parallel overhead: 0:00:00.025340 (6.80%)
		total partitioning: 0:00:00.024997 (6.71%)
		average partitioning: 0:00:00.000142
		total combining: 0:00:00.000343 (0.09%)
	maximum speedup (theoretical): 14.70x
	total partition count: 176
		estimator state: running
		estimated partition size: 3127


* **total CPU execution time** is the actual CPU time required to execute the parallel function.

  This duration is larger than the value returned by ``%timeit`` because **it sums the execution times for all
  the cores that processed the function**. It also includes the additional processing required to run Parfun (e.g. for
  partitioning the input and combining the results).

* **compute time** is how much CPU time was spent doing the actual computation.

  This value should roughly match the duration of the original sequential function if measured with ``%timeit``.
  The *min*, *max*, and *avg* values aggregate timing information across the partitioned parallel function calls.

* **total parallel overhead**, **total partitioning**, and **total combining** are the overheads related to the parallel
  execution of the function.

  These overheads limit the performance gains achievable through parallelization.

* **maximum speedup (theoretical)** estimates how much faster the function would run on a parallel machine with an
  unlimited number of cores.

  This value is calculated based on the input data size, function behavior, and the measured parallel overheads.
* **total partition count** and **current estimated partition size** describe how Parfun is splitting the input data.

  The library uses heuristics to estimate the optimal partition size. The library tries to find a partition size that
  provides significant parallel speedup without causing too much parallel overhead.
  :doc:`Read more </tutorials/implementation_details>` about how the optimal partition size is estimated.


.. note::
    As the library is constantly learning the optimal partition size, the first call to the parallelized function might
    not produce the most optimal timings. In these cases, it is recommended to call the function multiple times before
    analyzing the profiler output.


When setting the ``trace_export`` parameter, Parfun will dump the latest parallel function call metrics to the provided
CSV file. All durations in this file are measured in nanoseconds (10E–9).
