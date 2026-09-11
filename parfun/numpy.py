"""
A collection of pre-define APIs to partition and combine Numpy arrays.
"""

from typing import Iterable, Tuple

try:
    import numpy as np
except ImportError:
    raise ImportError("Numpy dependency missing. Use `pip install 'opengris-parfun[numpy]'` to install Numpy.")

from parfun.partition.object import PartitionGenerator


def concat(arrays: Iterable[np.ndarray]) -> np.ndarray:
    """
    Similar to :py:func:`numpy.concatenate`.

    .. code:: python

        array_1 = np.array([1, 2, 3])
        array_2 = np.array([4, 5, 6])

        print(concat([array_1, array_2]))
        # [1 2 3 4 5 6]

    """

    return np.concatenate(tuple(arrays))


def by_axis(*arrays: np.ndarray, axis: int = 0) -> PartitionGenerator[Tuple[np.ndarray, ...]]:
    """
    Partitions one or multiple Numpy arrays along the given axis.

    If multiple arrays are given, the returned partitions will have an identical size on the partitioned axis.

    .. code:: python

        array_1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

        with_partition_size(by_axis(array_1, axis=1), partition_size=2)

        # [(array([[1, 2],
        #          [5, 6]]),),
        #  (array([[3, 4],
        #          [7, 8]]),)]

    :param arrays: the arrays to partition
    :param axis: the axis to partition on. Negative values are not supported.
    """

    __validate_arrays_parameter(axis, *arrays)

    chunk_size = yield None

    def arrays_chunk(rng_start: int, rng_end: int) -> Tuple[np.ndarray, ...]:
        chunk_slice = (slice(None),) * axis + (slice(rng_start, rng_end),)
        return tuple(array[chunk_slice] for array in arrays)

    total_size = arrays[0].shape[axis]
    range_start = 0
    range_end = chunk_size
    while range_end < total_size:
        chunk_size = yield chunk_size, arrays_chunk(range_start, range_end)

        range_start = range_end
        range_end += chunk_size

    if range_start < total_size:
        yield total_size - range_start, arrays_chunk(range_start, total_size)


def by_row(*arrays: np.ndarray) -> PartitionGenerator[Tuple[np.ndarray, ...]]:
    """
    Partitions one or multiple Numpy arrays by rows (i.e. on their first axis).

    Equivalent to :py:func:`by_axis` with ``axis=0``.

    .. code:: python

        array_1 = np.array([1, 2, 3, 4, 5])
        array_2 = array_1 ** 2

        with_partition_size(by_row(array_1, array_2), partition_size=2)

        # [(array([1, 2]), array([1, 4])),
        #  (array([3, 4]), array([ 9, 16])),
        #  (array([5]), array([25]))]

    """

    return by_axis(*arrays, axis=0)


def by_col(*arrays: np.ndarray) -> PartitionGenerator[Tuple[np.ndarray, ...]]:
    """
    Partitions one or multiple Numpy arrays by columns (i.e. on their second axis).

    Equivalent to :py:func:`by_axis` with ``axis=1``.

    .. code:: python

        array_1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

        with_partition_size(by_col(array_1), partition_size=2)

        # [(array([[1, 2],
        #          [5, 6]]),),
        #  (array([[3, 4],
        #          [7, 8]]),)]

    """

    return by_axis(*arrays, axis=1)


def __validate_arrays_parameter(axis: int, *arrays: np.ndarray) -> None:
    if len(arrays) < 1:
        raise ValueError("missing `arrays` parameter.")

    if any(not isinstance(array, np.ndarray) for array in arrays):
        raise ValueError("all `arrays` values should be Numpy array instances.")

    if axis < 0:
        raise ValueError(f"negative `axis` values are not supported ({axis}).")

    if any(axis >= array.ndim for array in arrays):
        raise ValueError(f"`axis` ({axis}) out of bounds for at least one of the provided arrays.")

    total_size = arrays[0].shape[axis]
    if any(array.shape[axis] != total_size for array in arrays[1:]):
        raise ValueError(f"all arrays should have the same size on axis {axis}.")
