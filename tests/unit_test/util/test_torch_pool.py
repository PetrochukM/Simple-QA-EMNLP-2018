import unittest
import random
import math

from unittest.mock import patch

from seq2seq.util.torch_pool import pool
from seq2seq.util.torch_pool import TaskWorker


class MockTaskWorker(TaskWorker):

    def set_runtime_static_data(self):
        self.static_data = 'static_data'

    def execute_task(self, task):
        # Random failures are kept low for the sake of test speed.
        # Raise the error during actual testing.
        if random.random() < 0.25:
            raise RuntimeError()
        return task, self.static_data, self.init


class TorchPoolTest(unittest.TestCase):

    def setUp(self):
        # Avoid CUDA errors
        self.torch_mock = patch('seq2seq.util.torch_pool.torch').start()
        self.torch_mock.cuda.set_device.return_value = 1

        # Random Args
        self.n_devices = random.randint(1, 3)
        self.devices = list(range(self.n_devices))
        self.n_tasks = random.randint(2, 5)
        self.task = 'task'
        self.initial_tasks = [self.task]
        self.tasks_left = self.n_tasks - 1  # One task was sent as an initial_tasks
        self.results = []

    def mock_task_manager(self, task, static_data, init, *args):
        self.results.append(tuple([task, static_data, init]))
        if self.tasks_left >= 1:
            tasks = [self.task] * self.tasks_left
            self.tasks_left = 0
            return tasks
        else:
            return None

    def test_torch_pool_random_failure(self):
        pool(
            self.devices,
            MockTaskWorker,
            self.mock_task_manager,
            self.initial_tasks,
            n_retries=math.inf,
            task_worker_init={'init': 'init'})
        self.assertEqual(self.n_tasks, len(self.results))
        for result in self.results:
            self.assertEqual(result, tuple([self.task, 'static_data', 'init']))
