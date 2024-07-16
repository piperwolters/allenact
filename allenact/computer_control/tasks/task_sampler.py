import abc
import gym
from typing import List, Optional, Union

from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task, TaskSampler
from allenact.computer_control.tasks.task import ComputerControlTask


class ComputerControlTaskSampler(TaskSampler):
    def __init__(
        self,
        sensors: List[Sensor],
        max_steps: int,
        max_tasks: int,
        make_env_fn,
        #action_space: gym.Space,
        **task_init_kwargs
    ) -> None:

        self.env = make_env_fn()
        print("SELF>ENV:", self.env)

        self.sensors = sensors
        self.max_steps = max_steps
        self.max_tasks = max_tasks
        #self.action_space = action_space

        # Temporarily hardcode the tasks.
        train_task_file = open('/home/piperw/digirl/digirl/environment/android/assets/task_set/general_train.txt')
        self.train_tasks = train_task_file.readlines()
        test_task_file = open('/home/piperw/digirl/digirl/environment/android/assets/task_set/general_test.txt')
        self.test_tasks = test_task_file.readlines()
        print("Loaded in tasks...", len(self.train_tasks), " and ", len(self.test_tasks))

        self.counter = 0
        self.reset_tasks = max_tasks
        self._last_sampled_task = None

    @property
    def length(self) -> Union[int, float]:
        return 10

    @property
    def last_sampled_task(self) -> Optional[Task]:
        return self._last_sampled_task

    @property
    def all_observation_spaces_equal(self) -> bool:
        """
        @return: True if all Tasks that can be sampled by this sampler have the
            same observation space. Otherwise False.
        """
        return True

    def next_task(self, force_advance_scene: bool = False) -> Optional[Task]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.env is not None:
            self.env.refresh_driver()
        else:
            print("no existing env?")
            self.env.refresh_driver()

        if self.max_tasks is not None:
            self.max_tasks -= 1

        next_task = ComputerControlTask(env=self.env, sensors=self.sensors, task_info=self.train_tasks[self.counter], max_steps=self.max_steps)

        self.counter += 1
        self._last_sampled_task = next_task
        print("returning task...", self._last_sampled_task)
        return self._last_sampled_task

    def close(self) -> None:
        if self.env is not None:
            self.env.terminate()

    def reset(self) -> None:
        self.max_tasks = self.reset_tasks
        self.counter = 0

    def set_seed(self, seed: int):
        self.seed = seed
