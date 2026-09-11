API
===

.. autofunction:: parfun.parallel


Backend setup
-------------

.. autofunction:: parfun.set_parallel_backend

.. autofunction:: parfun.set_parallel_backend_context

.. autofunction:: parfun.get_parallel_backend


Backend instances
~~~~~~~~~~~~~~~~~

.. automodule:: parfun.backend.mixins
    :members:

.. automodule:: parfun.backend.local_single_process
    :members:

.. automodule:: parfun.backend.local_multiprocessing
    :members:

.. automodule:: parfun.backend.dask
    :members:

.. automodule:: parfun.backend.scaler
    :members:


Partitioning
------------

.. automodule:: parfun.partition.object
    :members:

.. autofunction:: parfun.all_arguments

.. autofunction:: parfun.multiple_arguments

.. autofunction:: parfun.per_argument

.. autofunction:: parfun.partition.utility.with_partition_size


.. _section-lists:

Python lists
------------

.. automodule:: parfun.py_list
    :members:


.. _section-numpy:

Numpy arrays
------------

.. automodule:: parfun.numpy
    :members:


.. _section-dataframes:

Pandas dataframes
-----------------

.. automodule:: parfun.dataframe
    :members:
