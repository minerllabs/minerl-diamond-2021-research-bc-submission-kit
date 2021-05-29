import random
import numpy as np
import gym
import torch as th
from torch import nn

MAX_TEST_EPISODE_LEN = 18000  # 18k is the default for MineRLObtainDiamondVectorObf.
TEST_MODEL_NAME = './train/research_potato.pth'  # name to use when loading the trained agent.
TEST_KMEANS_MODEL_NAME = './train/centroids_for_research_potato.npy'  # name to use when loading the KMeans model.


# !!! Do not change this! This is part of the submission kit !!!
class EpisodeDone(Exception):
    pass


# !!! Do not change this! This is part of the submission kit !!!
class Episode(gym.Env):
    """A class for a single episode."""
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._done = False

    def reset(self):
        if not self._done:
            return self.env.reset()

    def step(self, action):
        s, r, d, i = self.env.step(action)
        if d:
            self._done = True
            raise EpisodeDone()
        else:
            return s, r, d, i


class NatureCNN(nn.Module):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    Nicked from stable-baselines3:
        https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py
    :param input_shape: A three-item tuple telling image dimensions in (C, H, W)
    :param output_dim: Dimensionality of the output vector
    """

    def __init__(self, input_shape, output_dim):
        super().__init__()
        n_input_channels = input_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.zeros(1, *input_shape)).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class MineRLAgent():
    """
    To compete in the competition, you are required to implement the two
    functions in this class:
        - load_agent: a function that loads e.g. network models
        - run_agent_on_episode: a function that plays one game of MineRL

    By default this agent behaves like a random agent: pick random action on
    each step.

    NOTE:
        This class enables the evaluator to run your agent in parallel in Threads,
        which means anything loaded in load_agent will be shared among parallel
        agents. Take care when tracking e.g. hidden state (this should go to run_agent_on_episode).
    """

    def load_agent(self):
        """
        This method is called at the beginning of the evaluation.
        You should load your model and do any preprocessing here.
        THIS METHOD IS ONLY CALLED ONCE AT THE BEGINNING OF THE EVALUATION.
        DO NOT LOAD YOUR MODEL ANYWHERE ELSE.
        """
        # Load up the behavioural cloning model.
        self.action_centroids = np.load(TEST_KMEANS_MODEL_NAME)
        self.num_actions = self.action_centroids.shape[0]
        self.network = NatureCNN((3, 64, 64), self.num_actions).cuda()
        self.network.load_state_dict(th.load(TEST_MODEL_NAME))

    def run_agent_on_episode(self, single_episode_env: Episode):
        """This method runs your agent on a SINGLE episode.

        You should just implement the standard environment interaction loop here:
            obs  = env.reset()
            while not done:
                env.step(self.agent.act(obs))
                ...

        NOTE:
            This method will be called in PARALLEL during evaluation.
            So, only store state in LOCAL variables.
            For example, if using an LSTM, don't store the hidden state in the class
            but as a local variable to the method.

        Args:
            env (gym.Env): The env your agent should interact with.
        """
        env = single_episode_env

        action_list = np.arange(self.num_actions)

        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            # Process the action:
            #   - Add/remove batch dimensions
            #   - Transpose image (needs to be channels-last)
            #   - Normalize image
            obs = th.from_numpy(obs['pov'].transpose(2, 0, 1)[None].astype(np.float32) / 255).cuda()
            # Turn logits into probabilities
            probabilities = th.softmax(self.network(obs), dim=1)[0]
            # Into numpy
            probabilities = probabilities.detach().cpu().numpy()
            # Sample action according to the probabilities
            discrete_action = np.random.choice(action_list, p=probabilities)

            # Map the discrete action to the corresponding action centroid (vector)
            action = self.action_centroids[discrete_action]
            minerl_action = {"vector": action}

            obs, reward, done, info = env.step(minerl_action)
            total_reward += reward
            steps += 1
            if steps >= MAX_TEST_EPISODE_LEN:
                break
