import unittest

from parfun.backend.mixins import BackendEngine

try:
    from parfun.backend.scaler import ScalerLocalBackend, ScalerRemoteBackend

    scaler_installed = True
except ImportError:
    scaler_installed = False

from tests.backend.mixins import BackendEngineTestCase
from tests.backend.utility import no_op_task, warmup_workers

# A valid but unbound address. Any client trying to connect to it will fail to submit tasks.
UNREACHABLE_SCHEDULER_ADDRESS = "tcp://127.0.0.1:1"


def nested_session_task(backend: BackendEngine) -> None:
    """Submits a nested task from inside a worker, using the deserialized backend instance."""

    with backend.session() as session:
        return session.submit(no_op_task).result()


@unittest.skipUnless(scaler_installed, "Scaler backend not installed")
class TestScalerBackend(unittest.TestCase, BackendEngineTestCase):
    N_WORKERS = 4

    def setUp(self) -> None:
        self._backend = ScalerLocalBackend(n_workers=TestScalerBackend.N_WORKERS, per_worker_task_queue_size=1)

        warmup_workers(self._backend, self.n_workers())

    def tearDown(self) -> None:
        self.backend().shutdown()

    def n_workers(self) -> int:
        return TestScalerBackend.N_WORKERS

    def backend(self) -> BackendEngine:
        return self._backend

    def test_is_backend_scaler_remote(self):
        # ScalerLocalBackend is a special case of ScalerRemoteBackend, so no need to test ScalerRemoteBackend
        self.assertIsInstance(self._backend, (ScalerRemoteBackend, ScalerLocalBackend))

    def test_use_worker_client_ignores_parent_scheduler_address(self):
        """
        Nested tasks should connect to their worker's scheduler instead of the parent's address.

        The backend instance the workers receive advertises an unreachable scheduler address. Nested tasks will only
        succeed if they ignore it in favor of their worker's scheduler.
        """

        nested_backend = ScalerRemoteBackend(
            scheduler_address=UNREACHABLE_SCHEDULER_ADDRESS, n_workers=self.n_workers(), use_worker_client=True
        )

        with self.backend().session() as session:
            self.assertIsNone(session.submit(nested_session_task, nested_backend).result())

    def test_use_worker_client_disabled_uses_parent_scheduler_address(self):
        """With `use_worker_client=False`, the parent's unreachable address should be used, and thus fail."""

        CLIENT_TIMEOUT_SECONDS = 3

        nested_backend = ScalerRemoteBackend(
            scheduler_address=UNREACHABLE_SCHEDULER_ADDRESS,
            n_workers=self.n_workers(),
            use_worker_client=False,
            timeout_seconds=CLIENT_TIMEOUT_SECONDS,
        )

        with self.backend().session() as session:
            with self.assertRaises(Exception):
                session.submit(nested_session_task, nested_backend).result()


if __name__ == "__main__":
    unittest.main()
