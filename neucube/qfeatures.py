import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import random


def extract_features(spike_counts, w_in):
    """
    Combines neuron pooling and weight-based contribution for each input channel.

    Parameters:
        spike_counts: Tensor of shape (n_trials, n_neurons)
        w_in: Tensor of shape (n_neurons, n_channels)

    Returns:
        features: np.array of shape (n_trials, n_channels)
    """
    # Using .cpu().numpy() for calculations if inputs are torch tensors
    w_in_np = w_in.cpu().numpy()
    spike_counts_np = spike_counts.cpu().numpy()

    n_trials, n_neurons = spike_counts_np.shape
    n_channels = w_in_np.shape[1]

    features = np.zeros((n_trials, n_channels))

    # For each channel, get neurons that are most strongly connected to it
    strongest_channel = np.argmax(w_in_np, axis=1)
    
    for ch in range(n_channels):
        # Get neurons whose max connection is with this channel
        pool_indices = np.where(strongest_channel == ch)[0]
        
        if len(pool_indices) == 0:
            continue

        # Extract weights for this channel from selected neurons
        weights = w_in_np[pool_indices, ch]

        # Ensure weights sum is not zero to avoid division by zero
        if weights.sum() == 0:
            continue

        # For each trial, take the spike counts of these neurons and apply weighted average
        for t in range(n_trials):
            spikes_in_pool = spike_counts_np[t, pool_indices]
            weighted_avg = np.sum(spikes_in_pool * weights) / np.sum(weights)
            features[t, ch] = weighted_avg

    return features