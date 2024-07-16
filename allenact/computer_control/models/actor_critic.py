import gym
import torch
import torch.nn as nn

from allenact.base_abstractions.misc import ActorCriticOutput
from allenact.algorithms.onpolicy_sync.policy import ActorCriticModel, LinearActorHead, LinearCriticHead


class ComputerControlActorCritic(ActorCriticModel):
    def __init__(
        self, 
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.dict,
        **kwargs
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.action_space = action_space
        self.observation_space = observation_space

        self.hidden_size = 128
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, self.hidden_size),
            nn.ReLU()
        )
        
        self.actor_head = LinearActorHead(self.hidden_size, 8)  # there's 8 possible actions ? 
        self.critic_head = LinearCriticHead(self.hidden_size)

    def _recurrent_memory_specification(self):
	# uhhh???
        return {
            "rnn": (
                (("layer", 1), ("sampler", None), ("hidden", self.hidden_size)),
                torch.float32
            )
        }

    def forward(self, observations, memory, prev_actions, masks):

        x = observations["image_feature"]
        x = x.permute(0, 1, 4, 2, 3)  # [steps, samplers, channels, height, width]
        x = x.view(-1, *x.shape[2:])  # flatten [steps, samplers] into batch dimension
        features = self.feature_extractor(x)
        
        # Reshape features back to [steps, samplers, ...]
        features = features.view(*observations["image_feature"].shape[:2], -1)
        
        actor_out = self.actor_head(features)
        critic_out = self.critic_head(features)
        
        distribution = actor_out
        values = critic_out
        
        output = ActorCriticOutput(distribution=distribution, values=values, extras={})
        
        return output, memory
