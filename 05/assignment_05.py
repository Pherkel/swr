# ======
# Task 1
# ======

# Task 1: Proportion of the peel compared to the fruit in an n-dimensional
# orange. Out 1% of the radius is peel. Calculate the volume ratio between peel
# and fruit for the following dimensions 2, 3, 5, 10, 170. How does a
# 1-dimensional orange looks like? The diameter of the orange is 8 cm. Bonus:
# calculate the proportion for dimensions 300, 4096 as well. For these larger
# numbers you get an overflow error with the standard gamma function and need
# to reformulate the problem in log-space, i. e. taking the logarithm of
# everything and use the lgamma function.


# for two dimensions the formula is 
# =================================

# "Volume" of the peel: pi * r **2 (full orange) - pi * (0.99 * r) ** 2 (fruit)
# "Volume" of the fruit: pi * (0.99 * r) ** 2

# Ratio fruit / peel:  pi * (0.99 * r) ** 2 / (pi * r **2 - pi * (0.99 * r) ** 2)
# Proportion of fruit:  pi * (0.99 * r) ** 2 / (pi * r **2)

from math import pi, gamma

# r = 4cm
rr = 4.0

pi * (0.99 * rr) ** 2 / (pi * rr ** 2 - pi * (0.99 * rr) ** 2)
# 49 times more fruit than peel

pi * (0.99 * rr) ** 2 / (pi * rr ** 2)
# 98% fruit

# for three dimensions the formula is of the volume of a sphere is: 4/3 pi r ** 3

# the general formula is: https://en.wikipedia.org/wiki/Ball_(mathematics)

def volume_orange(rr, *, dimension=3):
    """Calculate the volume of a ball."""
    return pi ** (dimension / 2) / gamma(dimension / 2 + 1) * rr ** dimension

# The ratio now is:
volume_orange(0.99 * rr, dimension=2) / (volume_orange(rr, dimension=2) - volume_orange(0.99 * rr, dimension=2))

# The proportion of fruit is:
volume_orange(0.99 * rr, dimension=2) / volume_orange(rr, dimension=2)


# plot the proportion of fruit against the dimension:


dimensions = (2, 3, 5, 10, 170)
proportions = [ ???? # TODO
        for dd in dimensions]

from matplotlib import pyplot as plt

plt.plot(dimensions, proportions, label="proportion selected")
plt.plot((2, 170), (0.1, 0.1), "k:", label="10% line")
plt.plot((2, 170), (0.01, 0.01), "r--", label="1% line")
plt.legend()
plt.xlabel('number of dimensions')
plt.ylabel('proportion orange fruit')
plt.tight_layout()
plt.show()


# BONUS
dimensions = (2, 3, 5, 10, 170, 300, 4096)
proportions = [volume_orange(0.99 * rr, dimension=dd) / volume_orange(rr, dimension=dd)
        for dd in dimensions]
# Overflow error for 300

from math import log, lgamma, exp

# TODO


# ======
# Task 2
# ======

# What the following video (preferably with 2x speed)
# Random walks in 2D and 3D are fundamentally different (Markov chains approach)
# https://www.youtube.com/watch?v=iH2kATv49rc


# ======
# Task 3
# ======

# Task 3: Learn a linear mapping, LSTM mapping, and a Transformer for
# data0.npz, data1.npz, and data2.npz. Which model has the lowest training loss
# after training?

import numpy as np
import torch

npz_files = np.load('data0.npz')
print(npz_files.files)
xx0s = npz_files['xx0s']
yy0s = npz_files['yy0s']

class LinearModel(torch.nn.Module):
    """
    This is a sample classe for SWR2.
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_mapping = torch.nn.Linear(input_size, output_size)
        
    def forward(self, input_, *args):
        output = self.linear_mapping(input_)
        return output


class LSTMModel(torch.nn.Module):
    """
    This is a sample classe for SWR2.
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.lstm_layer = torch.nn.LSTM(input_size, hidden_size=16)
        self.linear = torch.nn.Linear(16, output_size)
        
    def forward(self, input_, *args):
        output, (h_n, c_n) = self.lstm_layer(input_)
        output = self.linear(output)
        return output


class TransformerModel(torch.nn.Module):
    """
    This is a sample classe for SWR2.
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=input_size, nhead=1, dim_feedforward=64)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.linear = torch.nn.Linear(input_size, output_size)
        
    def forward(self, input_, *args):
        output = self.encoder(input_)
        output = self.linear(output)
        return output


npz_files = np.load('data1.npz')
print(npz_files.files)
xx1s = npz_files['xx1s']
yy1s = npz_files['yy1s']


linear = LinearModel(2, 3)
lstm = LSTMModel(2, 3)
transformer_enc = TransformerModel(2, 3)

# test a single forward pass to see if the model is specified correctly
xx = torch.tensor(xx1s[0])

yy_linear = linear.forward(xx)
yy_linear.shape  # [3]

# insert the batch and sequence dimension with .view
yy_lstm = lstm.forward(xx.view(1, 1, 2))
yy_lstm.shape  # [1, 1, 3]
yy_lstm = yy_lstm.view(3)  # remove the batch and sequence dimension again


yy_transformer = transformer_enc.forward(xx.view(1, 1, 2))
yy_transformer.shape  # [1, 1, 3]
yy_transformer = yy_transformer.view(3)  # remove the batch and sequence dimension again


# do the learning
learning_rate = 0.001
optimizer = torch.optim.SGD(transformer_enc.parameters(), lr=learning_rate)

for epoch in range(100):
    losses = list()
    for index in range(len(yy1s)):
        xx = torch.tensor(xx1s[index])
        yy_true = torch.tensor(yy1s[index])

        # reset gradients
        optimizer.zero_grad()

        # forward pass
        yy_transformer = transformer_enc.forward(xx.view(1, 1, 2))
        yy_transformer = yy_transformer.view(3)  # remove the batch and sequence dimension again

        # loss computation
        loss = torch.mean( (yy_true - yy_transformer) ** 2 )
        losses.append(float(loss.item()))

        # backwards pass
        loss.backward()

        # stochastic gradient decent
        optimizer.step()
    print(f"Loss after epoch {epoch} is: {np.mean(losses)}")

# TODO do it for the other two models as well

# Which one has the lowest loss at the end?


# Now for the next data set:
npz_files = np.load('data2.npz')
print(npz_files.files)
xx2s = npz_files['xx2s']
yy2s = npz_files['yy2s']

linear = LinearModel(????)
lstm = LSTMModel(????)
transformer_enc = TransformerModel(????)


# TODO


# And for the third data set:

# TODO


# ======
# Task 4
# ======

# Task 4: Train an LSTM mapping an a Transformer on sequence data
# seq_data1.pkl. Plot the sequences and the predictions together. Which one
# converges faster?

import pickle

with open('seq_data0.pkl', 'rb') as pfile:
    seq_datas = pickle.load(pfile)


# test forward pass on first sequence
xx, yy = seq_datas[0]

xx.shape  # (19, 3)
yy.shape  # (19, 1)

# sequence length: 19
# channel input size: 3
# channel output size: 1

lstm = LSTMModel(3, 1)
transformer_enc = TransformerModel(3, 1)


xx = torch.tensor(xx)
xx = xx.view(19, 1, 3)

yy = torch.tensor(yy)
yy = yy.view(19, 1, 1)

yy_hat = lstm(xx)
torch.mean((yy - yy_hat) ** 2)

yy_hat = transformer_enc(xx)
torch.mean((yy - yy_hat) ** 2)


# TODO: Now do the training. Be aware that the sequence length is not always 19!

# Which model converges fast? Create a plot where the epoch is on the x-axis
# and the mean loss over all training samples is on the y-axis.


# NOTE: No batching is done in this example therefore no padding is required.




# =====
# Bonus
# =====

# simulate the random ball
# randomly sample 10,000 points from n-ball

# code adapted from https://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
# Uniformly sampling a d-ball: Method 20. Muller


from random import random

import numpy as np
from matplotlib import pyplot as plt

dd = 2 

xxs = list()
for _ in range(10000):
    uu = np.random.normal(0, 1, dd)  # an array of d normally distributed random variables
    norm = np.sum(uu**2) **(0.5)
    rr = random() ** (1.0 / dd)
    xx = rr * uu / norm
    xxs.append(xx)

xxs = np.array(xxs)
plt.rcParams['figure.figsize'] = [4, 4]
plt.plot(xxs[:, 0], xxs[:, 1], 'b.')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.title(f"{dd}-ball dimension 0 and 1")
plt.savefig(f"random_{dd}ball.png")
plt.show()

# 2-dimensional: rrs = (xxs[:, 0] ** 2 + xxs[:, 1] ** 2) ** (1/2)
# n-dimensional:
rrs = np.sum([xxs[:, col] ** 2 for col in range(xxs.shape[1])], axis=0) ** (1/2)
print(f"proportion fruit: {np.sum(rrs < .99) / 10000}")

