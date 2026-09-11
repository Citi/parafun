import inspect
import itertools
import threading
from collections import deque
from threading import BoundedSemaphore
from typing import Any, Deque, Dict, Optional, Set, Tuple

try:
    from scaler import Client, SchedulerClusterCombo
    from scaler.client.future import ScalerFuture
    from scaler.client.object_reference import ObjectReference
    from scaler.worker.agent.processor.processor import Processor
except ImportError:
    raise ImportError("Scaler dependency missing. Use `pip install 'opengris-parfun[scaler]'` to install Scaler.")

import psutil

from parfun.backend.mixins import BackendEngine, BackendSession
from parfun.backend.profiled_future import ProfiledFuture
from parfun.profiler.functions import profile


class ScalerSession(BackendSession):
    # Additional constant scheduling overhead that cannot be accounted for when measuring the task execution duration.
    CONSTANT_SCHEDULING_OVERHEAD = 8_000_000  # 8ms

    def __init__(self, scheduler_address: str, n_workers: int, client_kwargs: Dict, use_worker_client: bool = False):
        """
        :param use_worker_client: if ``True`` and the session is created inside a Scaler worker, connects to the
            scheduler the worker is currently connected to, instead of using ``scheduler_address``.
        """

        self._concurrent_task_guard = BoundedSemaphore(n_workers)
        self._client = self.__get_client_instance(scheduler_address, client_kwargs, use_worker_client)

    def __enter__(self) -> "ScalerSession":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._client.disconnect()

    def preload_value(self, value: Any) -> ObjectReference:
        return self._client.send_object(value)

    def submit(self, fn, *args, **kwargs) -> Optional[ProfiledFuture]:
        with profile() as submit_duration:
            future = ProfiledFuture()

            acquired = self._concurrent_task_guard.acquire()
            if not acquired:
                return None

            underlying_future = self._client.submit(fn, *args, **kwargs)

        def on_done_callback(underlying_future: ScalerFuture):
            assert submit_duration.value is not None

            if underlying_future.cancelled():
                future.cancel()
                return

            with profile() as release_duration:
                exception = underlying_future.exception()

                if exception is None:
                    result = underlying_future.result()
                    function_duration = int(underlying_future.profiling_info().cpu_time_s * 1_000_000_000)
                else:
                    function_duration = 0
                    result = None

                self._concurrent_task_guard.release()

            task_duration = (
                self.CONSTANT_SCHEDULING_OVERHEAD + submit_duration.value + function_duration + release_duration.value
            )

            if exception is None:
                future.set_result(result, duration=task_duration)
            else:
                future.set_exception(exception, duration=task_duration)

        underlying_future.add_done_callback(on_done_callback)

        return future

    @staticmethod
    def __get_client_instance(scheduler_address: str, client_kwargs: Dict, use_worker_client: bool) -> Client:
        """
        Instantiates the Scaler client to submit the session's tasks with.

        When ``use_worker_client`` is set and we are currently running inside a Scaler worker, lets Scaler resolve the
        scheduler's address from the running worker instead of reusing the (possibly unreachable) address from the
        parent task.
        """

        is_inside_worker = Processor.get_current_processor() is not None

        if use_worker_client and is_inside_worker:
            worker_client_kwargs = {
                kwarg: value
                for kwarg, value in client_kwargs.items()
                if kwarg not in {"address", "object_storage_address"}
            }

            return Client(address=None, profiling=True, **worker_client_kwargs)

        return Client(address=scheduler_address, profiling=True, **client_kwargs)


class ScalerClientPool:
    def __init__(self, scheduler_address: str, client_kwargs: Dict, max_unused_clients: int = 1):
        self._scheduler_address = scheduler_address
        self._client_kwargs = client_kwargs
        self._max_unused_clients = max_unused_clients

        self._lock = threading.Lock()
        self._assigned_clients: Dict[bytes, Client] = {}
        self._unassigned_clients: Deque[Client] = deque()  # a FIFO poll of up to `max_unused_clients`.

    def acquire(self) -> Client:
        with self._lock:
            if len(self._unassigned_clients) > 0:
                client = self._unassigned_clients.popleft()
            else:
                client = Client(address=self._scheduler_address, profiling=True, **self._client_kwargs)

            self._assigned_clients[client.identity] = client

            return client

    def release(self, client: Client) -> None:
        with self._lock:
            self._assigned_clients.pop(client.identity)

            if len(self._unassigned_clients) < self._max_unused_clients:
                self._unassigned_clients.append(client)
            else:
                client.disconnect()

    def disconnect_all(self) -> None:
        with self._lock:
            for client in itertools.chain(self._unassigned_clients, self._assigned_clients.values()):
                client.disconnect()

            self._unassigned_clients.clear()
            self._assigned_clients.clear()


class ScalerRemoteBackend(BackendEngine):
    """Connects to a previously instantiated Scaler instance as a backend engine."""

    def __init__(
        self,
        scheduler_address: str,
        n_workers: int = psutil.cpu_count(logical=False) - 1,
        allows_nested_tasks: bool = True,
        use_worker_client: bool = False,
        **client_kwargs,
    ):
        """
        :param use_worker_client: if ``True``, nested tasks connect to the scheduler their worker is currently
            connected to, instead of reusing the address and the network settings of the process that submitted the
            parent task.

            Required when the workers cannot reach the scheduler at the same address as the submitting process
            (e.g. NAT-ed, Docker or Kubernetes deployments).
        """

        self._scheduler_address = scheduler_address
        self._n_workers = n_workers
        self._allows_nested_tasks = allows_nested_tasks
        self._use_worker_client = use_worker_client
        self._client_kwargs = client_kwargs

    def __getstate__(self) -> dict:
        return {
            "scheduler_address": self._scheduler_address,
            "n_workers": self._n_workers,
            "allows_nested_tasks": self._allows_nested_tasks,
            "use_worker_client": self._use_worker_client,
            "client_kwargs": self._client_kwargs,
        }

    def __setstate__(self, state: dict) -> None:
        self._scheduler_address = state["scheduler_address"]
        self._n_workers = state["n_workers"]
        self._allows_nested_tasks = state["allows_nested_tasks"]
        self._use_worker_client = state["use_worker_client"]
        self._client_kwargs = state["client_kwargs"]

    def session(self) -> ScalerSession:
        return ScalerSession(self._scheduler_address, self._n_workers, self._client_kwargs, self._use_worker_client)

    def get_scheduler_address(self) -> str:
        return self._scheduler_address

    def disconnect(self):
        pass

    def shutdown(self):
        pass

    def allows_nested_tasks(self) -> bool:
        return self._allows_nested_tasks


class ScalerLocalBackend(ScalerRemoteBackend):
    """Creates a Scaler cluster on the local machine and uses it as a backend engine."""

    def __init__(
        self,
        scheduler_address: Optional[str] = None,
        n_workers: int = psutil.cpu_count(logical=False) - 1,
        per_worker_task_queue_size: int = 1000,
        allows_nested_tasks: bool = True,
        use_worker_client: bool = False,
        logging_paths: Tuple[str, ...] = ("/dev/stdout",),
        logging_level: str = "INFO",
        logging_config_file: Optional[str] = None,
        **kwargs,
    ):
        """
        :param scheduler_address the ``tcp://host:port`` tuple to use as a cluster address. If ``None``, listen to the
        local host on an available TCP port.
        :param use_worker_client: see :class:`ScalerRemoteBackend`.
        """

        client_kwargs = self.__get_constructor_arg_names(Client)
        scheduler_cluster_combo_kwargs = self.__get_constructor_arg_names(SchedulerClusterCombo)

        self._cluster = SchedulerClusterCombo(
            address=scheduler_address,
            n_workers=n_workers,
            logging_paths=logging_paths,
            logging_level=logging_level,
            logging_config_file=logging_config_file,
            per_worker_task_queue_size=per_worker_task_queue_size,
            **{kwarg: value for kwarg, value in kwargs.items() if kwarg in scheduler_cluster_combo_kwargs},
        )
        scheduler_address = self._cluster.get_address()

        super().__init__(
            scheduler_address=scheduler_address,
            allows_nested_tasks=allows_nested_tasks,
            use_worker_client=use_worker_client,
            n_workers=n_workers,
            **{kwarg: value for kwarg, value in kwargs.items() if kwarg in client_kwargs},
        )

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        self._cluster = None  # Unserialized instances have no cluster reference.

    @property
    def cluster(self) -> SchedulerClusterCombo:
        if self._cluster is None:
            raise AttributeError("cluster is undefined for serialized instances.")

        return self._cluster

    def shutdown(self):
        super().shutdown()

        if self._cluster is not None:
            self._cluster.shutdown()
            self._cluster = None

    @staticmethod
    def __get_constructor_arg_names(class_: type) -> Set:
        return set(inspect.signature(class_).parameters.keys())
