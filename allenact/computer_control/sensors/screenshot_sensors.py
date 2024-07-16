import gym
from typing import Optional, Any

from allenact.base_abstractions.misc import EnvType
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import SubTaskType


class ScreenshotSensor(Sensor[EnvType, SubTaskType]):
    def __init__(self, uuid: str, observation_space: gym.Space):
        super(ScreenshotSensor, self).__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any) -> Any:
        all_obs = env.get_obs()
        return all_obs['image_feature']
