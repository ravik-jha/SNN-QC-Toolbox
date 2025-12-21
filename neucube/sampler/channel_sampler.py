# neucube/sampler/channel_sampler.py

import torch

class ChannelContributionSampler:
    def __init__(self):
        self.channel_map = None

    def setup(self, w_in):
        # Normalize w_in transpose so each row sums to 1
        self.channel_map = w_in.T
        self.channel_map = self.channel_map / self.channel_map.sum(dim=1, keepdim=True)

    def sample(self, spike_rec):
        """
        spike_rec: Tensor of shape (n_trials, n_time, n_neurons)
        returns: Tensor of shape (n_trials, n_channels)
        """
        spike_counts = spike_rec.sum(dim=1)  # sum over time
        features = torch.matmul(spike_counts, self.channel_map.T)  # weighted sum
        return features