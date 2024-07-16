"""
Script to invoke an experiment for the computer control project.

PYTHONPATH=. python allenact/main.py computer_control -b . -m 8 -o results/computer_control_output -s 12345
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from typing import Any, Dict, List, Optional 

from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.base_abstractions.experiment_config import ExperimentConfig, TaskSampler
from allenact.base_abstractions.sensor import SensorSuite
from allenact.computer_control.envs.android_env import get_env
from allenact.computer_control.models.actor_critic import ComputerControlActorCritic
from allenact.computer_control.sensors.screenshot_sensors import ScreenshotSensor
from allenact.computer_control.tasks.task_sampler import ComputerControlTaskSampler
from allenact.computer_control.tasks.task import ActionType, ComputerControlTask
from allenact.utils.experiment_utils import Builder, LinearDecay, PipelineStage, TrainingPipeline

class ComputerControlExperimentConfig(ExperimentConfig):

    @classmethod
    def tag(cls) -> str:
        return "ComputerControl"

    SENSORS = [
        ScreenshotSensor(uuid='screenshot', observation_space=gym.spaces.Box(low=1.0, high=2.0))
    ]

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:

        action_types = gym.spaces.Discrete(len(ActionType))

        touch_point_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=float)
        lift_point_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=float)
        typed_text_space = gym.spaces.Text(max_length=256)

        action_space = gym.spaces.Dict({
            "action_type": action_types,
            "touch_point": touch_point_space,
            "lift_point": lift_point_space,
            "typed_text": typed_text_space
        })

        observation_space=gym.spaces.Box(low=1.0, high=2.0)
        return ComputerControlActorCritic(action_space=action_space, observation_space=observation_space, **kwargs)

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return ComputerControlTaskSampler(**kwargs)

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(process_ind=process_ind, mode="train")

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(process_ind=process_ind, mode="valid")

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(process_ind=process_ind, mode="test")

    def _get_sampler_args(self, process_ind: int, mode: str) -> Dict[str, Any]:
        """Generate initialization arguments for train, valid, and test
        TaskSamplers.

        # Parameters
        process_ind : index of the current task sampler
        mode:  one of `train`, `valid`, or `test`
        """
        if mode == "train":
            max_tasks = None  # infinite training tasks
            task_seeds_list = None  # no predefined random seeds for training
            deterministic_sampling = False  # randomly sample tasks in training
        else:
            max_tasks = 20 + 20 * (mode == "test")  # 20 tasks for valid, 40 for test

            # one seed for each task to sample:
            # - ensures different seeds for each sampler, and
            # - ensures a deterministic set of sampled tasks.
            task_seeds_list = list(
                range(process_ind * max_tasks, (process_ind + 1) * max_tasks)
            )

            deterministic_sampling = (
                True  # deterministically sample task in validation/testing
            )

        return dict(
            max_tasks=max_tasks,  # see above
            max_steps=10,  # hardcoded for now
            make_env_fn=self.make_env,  # builder for third-party environment (defined below)
            sensors=self.SENSORS,  # sensors used to return observations to the agent
            env_info=dict(),  # parameters for environment builder (none for now)
            task_seeds_list=task_seeds_list,  # see above
            deterministic_sampling=deterministic_sampling,  # see above
        ) 

    @staticmethod
    def make_env(*args, **kwargs):
        return get_env()

    @classmethod
    def machine_params(cls, mode="train", **kwargs) -> Dict[str, Any]:
        return {
            "nprocesses": 1 if mode == "train" else 16,
            "devices": [],
        }

    @classmethod
    def training_pipeline(cls, **kwargs) -> TrainingPipeline:
        named_losses = {"ppo_loss": (PPO(**PPOConfig, normalize_advantage=True), 1.0)}
        return TrainingPipeline(
		save_interval=5000000,
		metric_accumulate_interval=10000 if torch.cuda.is_available() else 1,
		optimizer_builder=Builder(optim.Adam, dict(lr=3e-4)),
		num_mini_batch=1,
		update_repeats=4,
		max_grad_norm=0.5,
		num_steps=128,
		named_losses=named_losses,
		gamma=0.99,
		use_gae=True,
		gae_lambda=0.95,
		advance_scene_rollout_period=None,
		pipeline_stages=[
		    PipelineStage(
			loss_names=list(named_losses.keys()),
			max_stage_steps=int(75000000),
			loss_weights=[val[1] for val in named_losses.values()],
		    )
		],
		lr_scheduler_builder=Builder(
		    LambdaLR, {"lr_lambda": LinearDecay(steps=int(75000000))}
		),
	    )
