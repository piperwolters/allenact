import abc
import gym
import numpy as np
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar, Union

from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.base_abstractions.task import Task


EnvType = TypeVar("EnvType")

class ActionType(Enum):
    Idle=0
    DualPoint=1
    Type=2
    GoBack=3
    GoHome=4
    Enter=5
    TaskComplete=6
    TaskImpossible=7

class ComputerControlTask(Task):

    def __init__(
        self,
        env: EnvType,
        sensors: Union[SensorSuite, Sequence[Sensor]],
        task_info: Dict[str, Any],
        max_steps: int,
        **kwargs
    ) -> None:
        self.env = env
        self.sensor_suite = (
            SensorSuite(sensors) if not isinstance(sensors, SensorSuite) else sensors
        )
        self.task_info = task_info
        self.max_steps = max_steps
        self.observation_space = self.sensor_suite.observation_spaces
        self._num_steps_taken = 0
        self._total_reward: Union[float, List[float]] = 0.0

    @property
    def action_space(self) -> gym.Space:
        action_types = gym.spaces.Discrete(len(ActionType))

        touch_point_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=float)
        lift_point_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=float)
        typed_text_space = gym.spaces.Text(max_length=256)

        return gym.spaces.Dict({
            "action_type": action_types,
            "touch_point": touch_point_space,
            "lift_point": lift_point_space,
            "typed_text": typed_text_space
        })

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    def _step(self, action: Any) -> RLStepResult:
        print("in _step(action....")
        self.env.step(action)

    def reached_terminal_state(self) -> bool:
        raise NotImplementedError()
