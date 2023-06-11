import torch


class IntroModel(torch.nn.Module):
    """
    This is a sample classe for SWR2.
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_mapping = torch.nn.Linear(input_size, output_size)
        
    def forward(self, input_, *args):
        output = self.linear_mapping(input_)
        return output


# 0  skalars
# 1  vectors
# 2  matrix
# 3  tensor of order 3
# 4  tensor of order 4


intro_model = IntroModel(1, 2)
xx = torch.tensor([1.0])

intro_model.forward(xx)


xx = torch.tensor([1.0, 2.0, 3.0, 4.0])

#intro_model.forward(xx)  # RuntimeError

xx.shape

xx = xx.view(4, 1)

intro_model.forward(xx)

# now it works and gives the four results in parrallel

yy = intro_model.forward(xx)

yy.shape  # [4, 2]


# The first dimension is interpreted as batch dimension and all computatoins
# are done in parralel along the first dimension.


# The linear_mapping maps one number on two numbers. This is done by a linear
# mapping with four paramerters. The four parameters are two weights and two
# biases.

intro_model.linear_mapping.weight

intro_model.linear_mapping.bias


xx = torch.tensor([3.1415])

# The linear mapping multiplies the weights and adds the bias.

yy0 = intro_model.linear_mapping.weight[0] * xx + intro_model.linear_mapping.bias[0]
yy1 = intro_model.linear_mapping.weight[1] * xx + intro_model.linear_mapping.bias[1]

yy = intro_model.forward(xx)

yy == torch.cat((yy0, yy1))  # they are the same


# Now we want to change the behavior of the intro_model, so that it produces
# two times the input as the first number and the negative input + 3 as the
# second number.

yy_true = torch.tensor([2 * xx, -xx + 3])


# now we let the intro_model predict the (wrong) numbers
yy = intro_model.forward(xx)

# calculate an error
error = torch.mean((yy_true - yy) ** 2)

# backpropagete the error to get gradients on all weights and biases
error.backward()

intro_model.linear_mapping.weight.grad
intro_model.linear_mapping.bias.grad

# The gradients give us the information how we have to change the weights and
# biases to minimize the resulting error. We will change the weights and biases
# only a little bit. This is called the learning rate.

learning_rate = 0.01

new_weights = intro_model.linear_mapping.weight - learning_rate * intro_model.linear_mapping.weight.grad
new_biases = intro_model.linear_mapping.bias - learning_rate * intro_model.linear_mapping.bias.grad

intro_model.linear_mapping.weight = torch.nn.Parameter(new_weights)
intro_model.linear_mapping.bias = torch.nn.Parameter(new_biases)


# Now lets do this 10 times in a loop and see if the error gets smaller:

for epoch in range(30):
    yy = intro_model.forward(xx)
    error = torch.mean((yy_true - yy) ** 2)
    error.backward()
    new_weights = intro_model.linear_mapping.weight - learning_rate * intro_model.linear_mapping.weight.grad
    new_biases = intro_model.linear_mapping.bias - learning_rate * intro_model.linear_mapping.bias.grad

    intro_model.linear_mapping.weight = torch.nn.Parameter(new_weights)
    intro_model.linear_mapping.bias = torch.nn.Parameter(new_biases)

    print(f"Error in epoch {epoch} is: {float(error)}")


# The model can now create the desired output for this single input xx =
# 3.1415, but what happens if we give it a new input?

xx = torch.tensor([1.5])
yy_true = torch.tensor([2 * xx, -xx + 3])
yy = intro_model.forward(xx)
error = torch.mean((yy_true - yy) ** 2)

print(f"Error: {float(error)}")

# For this new number we have a huge error :-(

# But can we minimize the error for this values as well?

for epoch in range(30):
    yy = intro_model.forward(xx)
    error = torch.mean((yy_true - yy) ** 2)
    error.backward()
    new_weights = intro_model.linear_mapping.weight - learning_rate * intro_model.linear_mapping.weight.grad
    new_biases = intro_model.linear_mapping.bias - learning_rate * intro_model.linear_mapping.bias.grad

    intro_model.linear_mapping.weight = torch.nn.Parameter(new_weights)
    intro_model.linear_mapping.bias = torch.nn.Parameter(new_biases)

    print(f"Error in epoch {epoch} is: {float(error)}")


# The problem is that this might increase the error for the first number again.
# Solution: Let us do it for many numbers over and over again.

import numpy as np
xxs = np.array(np.random.normal(size=100), dtype=np.float32)

for epoch in range(30):
    np.random.shuffle(xxs)  # we don't want to have the same order in each epoch
    errors = list()
    for xx in xxs:
        xx = torch.tensor([xx])
        yy_true = torch.tensor([2 * xx, -xx + 3])
        yy = intro_model.forward(xx)
        error = torch.mean((yy_true - yy) ** 2)
        error.backward()
        new_weights = intro_model.linear_mapping.weight - learning_rate * intro_model.linear_mapping.weight.grad
        new_biases = intro_model.linear_mapping.bias - learning_rate * intro_model.linear_mapping.bias.grad

        intro_model.linear_mapping.weight = torch.nn.Parameter(new_weights)
        intro_model.linear_mapping.bias = torch.nn.Parameter(new_biases)
        errors.append(float(error))
    print(f"Average Error in epoch {epoch} is: {np.mean(errors)}")



print(intro_model.linear_mapping.weight)
print(intro_model.linear_mapping.bias)


# Can you find a -1, 0, 2, and 3 in the weights and biases? Where are these four numbers present as well?

# TODO


# Instantiate a sencond intro_model that takes a vector with two numbers as
# input and outputs a single number. Train the second intro model to produce
# the sum of the two numbers.

xx = torch.tensor([1.0, 2.0])

yy_true = torch.tensor([3.0])

# TODO
intro_model2 = IntroModel(????)

yy = intro_model2(xx)

error = torch.mean((yy_true - yy) ** 2)

print(f"Initial error: {float(error)}")

xxs = np.array(np.random.normal(size=(100, 100)), dtype=np.float32)
    
# TODO


# Bonus Questions: Can we model the square of the input or the logarithm of the input with our IntroModel?

# TODO

