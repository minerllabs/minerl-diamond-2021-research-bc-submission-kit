import json
import select
import time
import logging
import os
import random

import numpy as np
import aicrowd_helper
import gym
import minerl
import torch as th
from torch import nn
from sklearn.cluster import KMeans
from utility.parser import Parser

from test_submission_code import NatureCNN

import coloredlogs
coloredlogs.install(logging.DEBUG)

# --- NOTE ---
# This code is only used for "Research" track submissions
# ------------

EPOCHS = 2  # how many times we train over dataset.
LEARNING_RATE = 0.0001  # learning rate for the neural network.
BATCH_SIZE = 32
NUM_ACTION_CENTROIDS = 100  # number of KMeans centroids used to cluster the data.
DATA_SAMPLES = 1000000
TRAIN_MODEL_NAME = './train/research_potato.pth'  # name to use when saving the trained agent.
TRAIN_KMEANS_MODEL_NAME = './train/centroids_for_research_potato.npy'  # name to use when saving the KMeans model.


# All research-tracks evaluations will be ran on the MineRLObtainDiamondVectorObf-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamondVectorObf-v0')
# You need to ensure that your submission is trained in under MINERL_TRAINING_MAX_STEPS steps
MINERL_TRAINING_MAX_STEPS = int(os.getenv('MINERL_TRAINING_MAX_STEPS', 8000000))
# You need to ensure that your submission is trained by launching less than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 5))
# You need to ensure that your submission is trained within allowed training time.
# Round 1: Training timeout is 5 minutes
# Round 2: Training timeout is 4 days
MINERL_TRAINING_TIMEOUT = int(os.getenv('MINERL_TRAINING_TIMEOUT_MINUTES', 4 * 24 * 60))
# The dataset is available in data/ directory from repository root.
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')

# Optional: You can view best effort status of your instances with the help of parser.py
# This will give you current state like number of steps completed, instances launched and so on. Make your you keep a tap on the numbers to avoid breaching any limits.
parser = Parser(
    'performance/',
    allowed_environment=MINERL_GYM_ENV,
    maximum_instances=MINERL_TRAINING_MAX_INSTANCES,
    maximum_steps=MINERL_TRAINING_MAX_STEPS,
    raise_on_error=False,
    no_entry_poll_timeout=600,
    submission_timeout=MINERL_TRAINING_TIMEOUT * 60,
    initial_poll_timeout=600
)


def train():
    # For demonstration purposes, we will only use ObtainPickaxe data which is smaller,
    # but has the similar steps as ObtainDiamond in the beginning.
    # "VectorObf" stands for vectorized (vector observation and action), where there is no
    # clear mapping between original actions and the vectors (i.e. you need to learn it)
    data = minerl.data.make("MineRLObtainIronPickaxeVectorObf-v0",  data_dir=MINERL_DATA_ROOT, num_workers=1)

    # First, use k-means to find actions that represent most of them.
    # This proved to be a strong approach in the MineRL 2020 competition.
    # See the following for more analysis:
    # https://github.com/GJuceviciute/MineRL-2020

    # Go over the dataset once and collect all actions and the observations (the "pov" image).
    # We do this to later on have uniform sampling of the dataset and to avoid high memory use spikes.
    all_actions = []
    all_pov_obs = []

    trajectory_names = data.get_trajectory_names()
    random.shuffle(trajectory_names)

    # Add trajectories to the data until we reach the required DATA_SAMPLES.
    for trajectory_name in trajectory_names:
        trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
        for dataset_observation, dataset_action, _, _, _ in trajectory:
            all_actions.append(dataset_action["vector"])
            all_pov_obs.append(dataset_observation["pov"])
        if len(all_actions) >= DATA_SAMPLES:
            break

    all_actions = np.array(all_actions)
    all_pov_obs = np.array(all_pov_obs)

    # Run k-means clustering using scikit-learn.
    kmeans = KMeans(n_clusters=NUM_ACTION_CENTROIDS)
    kmeans.fit(all_actions)
    action_centroids = kmeans.cluster_centers_

    # Now onto behavioural cloning itself.
    # Much like with intro track, we do behavioural cloning on the discrete actions,
    # where we turn the original vectors into discrete choices by mapping them to the closest
    # centroid (based on Euclidian distance).

    network = NatureCNN((3, 64, 64), NUM_ACTION_CENTROIDS).cuda()
    optimizer = th.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    num_samples = all_actions.shape[0]
    update_count = 0
    # We have the data loaded up already in all_actions and all_pov_obs arrays.
    # Let's do a manual training loop
    for _ in range(EPOCHS):
        # Randomize the order in which we go over the samples
        epoch_indices = np.arange(num_samples)
        np.random.shuffle(epoch_indices)
        for batch_i in range(0, num_samples, BATCH_SIZE):
            # NOTE: this will cut off incomplete batches from end of the random indices
            batch_indices = epoch_indices[batch_i:batch_i + BATCH_SIZE]

            # Load the inputs and preprocess
            obs = all_pov_obs[batch_indices].astype(np.float32)
            # Transpose observations to be channel-first (BCHW instead of BHWC)
            obs = obs.transpose(0, 3, 1, 2)
            # Normalize observations. Do this here to avoid using too much memory (images are uint8 by default)
            obs /= 255.0

            # Map actions to their closest centroids
            action_vectors = all_actions[batch_indices]
            # Use numpy broadcasting to compute the distance between all
            # actions and centroids at once.
            # "None" in indexing adds a new dimension that allows the broadcasting
            distances = np.sum((action_vectors - action_centroids[:, None]) ** 2, axis=2)
            # Get the index of the closest centroid to each action.
            # This is an array of (batch_size,)
            actions = np.argmin(distances, axis=0)

            # Obtain logits of each action
            logits = network(th.from_numpy(obs).float().cuda())

            # Minimize cross-entropy with target labels.
            # We could also compute the probability of demonstration actions and
            # maximize them.
            loss = loss_function(logits, th.from_numpy(actions).long().cuda())

            # Standard PyTorch update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_count += 1

    # Save network and the centroids into separate files
    np.save(TRAIN_KMEANS_MODEL_NAME, action_centroids)
    th.save(network.state_dict(), TRAIN_MODEL_NAME)
    del data


def main():
    """
    This function will be called for training phase.
    """
    train()


if __name__ == "__main__":
    main()
