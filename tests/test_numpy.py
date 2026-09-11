import math
import unittest
from typing import List

try:
    import numpy as np
except ImportError:
    raise ImportError("Numpy dependency missing. Use `pip install 'opengris-parfun[numpy]'` to install Numpy.")

import parfun as pf
from parfun.partition.utility import with_partition_size


class TestNumpy(unittest.TestCase):
    def test_concat(self):
        array_1 = np.array([1, 2, 3])
        array_2 = np.array([4, 5, 6])

        self.assertTrue(np.array_equal(pf.numpy.concat([array_1, array_2]), np.arange(1, 7)))

        array_3 = np.array([[1, 2], [3, 4]])
        array_4 = np.array([[5, 6], [7, 8]])

        self.assertTrue(
            np.array_equal(pf.numpy.concat([array_3, array_4]), np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
        )

    def test_by_row(self):
        def test_with_params(input_arrays: List[np.ndarray], partition_size: int):
            n_rows = input_arrays[0].shape[0]

            partitions = list(with_partition_size(pf.numpy.by_row(*input_arrays), partition_size=partition_size))

            self.assertEqual(len(partitions), math.ceil(n_rows / partition_size))

            # Validates the partition shapes.
            for partitioned_arrays in partitions:
                self.assertEqual(len(partitioned_arrays), len(input_arrays))

                for partition_array, input_array in zip(partitioned_arrays, input_arrays):
                    self.assertEqual(partition_array.shape[1:], input_array.shape[1:])
                    self.assertLessEqual(partition_array.shape[0], partition_size)

            # Validates the partition values.
            for input_array_i, input_array in enumerate(input_arrays):
                output_array = np.concatenate([partition[input_array_i] for partition in partitions])
                self.assertTrue(np.array_equal(input_array, output_array))

        test_with_params([np.arange(1)], partition_size=1)
        test_with_params([np.arange(13)], partition_size=1)
        test_with_params([np.arange(13 * 23).reshape(13, 23), np.arange(13 * 3).reshape(13, 3)], partition_size=3)
        test_with_params([np.arange(2 * 3 * 4).reshape(2, 3, 4)], partition_size=5)
        test_with_params([np.array(list("hello")), np.arange(5)], partition_size=2)

        with self.assertRaises(ValueError):
            test_with_params([np.arange(10), np.arange(6)], partition_size=5)

    def test_by_col(self):
        input_array = np.arange(2 * 4).reshape(2, 4)

        partitions = list(with_partition_size(pf.numpy.by_col(input_array), partition_size=2))

        self.assertEqual(len(partitions), 2)
        self.assertTrue(np.array_equal(partitions[0][0], np.array([[0, 1], [4, 5]])))
        self.assertTrue(np.array_equal(partitions[1][0], np.array([[2, 3], [6, 7]])))

        # A 1-dimensional array has no column axis.
        with self.assertRaises(ValueError):
            list(with_partition_size(pf.numpy.by_col(np.arange(4)), partition_size=2))

    def test_parallel_function(self):
        @pf.parallel(split=pf.per_argument(values=pf.numpy.by_row), combine_with=pf.numpy.concat)
        def multiply_by_constant(values: np.ndarray, constant: int) -> np.ndarray:
            return values * constant

        values = np.arange(1000)

        with pf.set_parallel_backend_context("local_single_process"):
            self.assertTrue(np.array_equal(multiply_by_constant(values, 3), values * 3))


if __name__ == "__main__":
    unittest.main()
