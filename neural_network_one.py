# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Last Update: 12/22/25
# File Name: neural_network_one.py
#
# Description:
#   This project implements a very simple neural network from scratch,
#   following the fundamental training loop shared by all modern models.
#   The network is intentionally limited to a single linear layer, as this
#   is my first full from-scratch implementation.
#
#   Future iterations will extend this into a multi-layer network with
#   activation functions. While this implementation is trivial in scope,
#   each line was written deliberately to build a concrete understanding
#   of forward passes, loss computation, backpropagation, and optimization.


# =========================
# Inference (Forward Pass)
# =========================
# 1. Data (batch)
# 2. Linear Layer (Wx + b)
# 3. Logits *Skipped here only 1 output*

# =========================
# Training Loop
# =========================
# 4. Loss Computation (from logits) 
# 5. Backpropagation (gradients)
# 6. Optimization step
# 7. Repeat for all batches → epochs


import numpy as np
import random 

# ======================================================
# Step 1 — Define the Problem (Before Writing Any Code)
# ======================================================

# ---- Batch Definition ----
# Q: What is a "batch" mathematically?
# - Is a batch a vector or a matrix?
# - If it is a matrix:
#     • What does EACH ROW represent?
#     • What does EACH COLUMN represent?
#
# Write your answer here in words:
# A: I want my batch to be a list of 100 3D vectors using xyz or ijk.
# So each row is a new vector with column 1 being x or i,
# column 2 being y or j, column 3 being z or k.
#    
#    

# ---- Input Dimensionality ----
# Q: What is the dimensionality of ONE input example?
# - Pick a concrete number (e.g. 2, 3, 10)
# - This represents how many features each sample has
#
# Chosen input_dim = 3
#
# Why did you choose this number?
# Reason: Im comfortable with 3D vectors and am familiar with their operations.

# ---- Output Dimensionality ----
# Q: How many outputs does the model produce?
# - One value → regression
# - Multiple values → classification (logits)
#
# Chosen output_dim = 1 output
#
# What does EACH output value represent?
# Meaning: sum of all components in the vector

# ---- Shape Expectations (Must Be True Later) ----
# These are NOT code — they are invariants.
#
# X shape      = (batch_size, input_dim)
# W shape      = (input_dim, output_dim)
# b shape      = (output_dim,)
# logits shape = (batch_size, output_dim)
#
# If any of these do not hold, the model is wrong.

batch_size = 100
input_dim = 3
output_dim = 1
epsilon = 1e-3           # how much we nudge a weight to test direction
learning_rate = 0.1      # how much we actually move it
epochs = 3

# Want to fit each internal part of the shape as random
random_num = 0.0  

# This is our shape: data(100, 3) with 3 adjustable weights and 1 output/bias
# BIAS IS INDEPENDANT OF WEIGHTS -> 
# Weights = how loud each input speaks
# Bias = where the volume knob starts before anyone speaks
data = np.zeros((batch_size,input_dim), dtype=np.float32)

# This fills the shape with randomized data 1-10
for i in range(batch_size):
    for j in range(input_dim):
        random_num = random.uniform(1, 10)
        data[i][j] = random_num

# Time to set our parameters:
#   - Need 3 for the weights (x, y, z) within the shape
#   - Need 1 for the bias which produces 1 output
# Shapes for the params:
#   - Weights (3, 1) 
#   - Bias (1,) -> **NOT (1,0) because No value exists then **
# It is NOT (100, 3) or (100, 1) because then we are implying a parameter
# is attached to each individual vector in our 2d matricies.


# Matrix Multiplication:
#   Essential for the next steps because we want an input shape of (100, 3)
#   multiplied by weight shape of (3, 1) to get an output of (100, 1)

# EXAMPLES:
#   *Ex1:
#            [1,3]           [1,3]
#       Data [2,3,4] Weights [1,1,1] Bias = 0
#       Math: 2*1 + 3*1 + 4*1 = 9
#
#   *Ex2:
#            [3,3]           [3,1]
#       Data:        Weights:        Bias = 5
#       [1,2,3]            [2]
#       [4,5,6]            [0]
#       [7,8,9]            [1]
#
#       Math: 
#           Row1: 1*2 + 2*0 + 3*1 + 5 = 10
#           Row2: 4·2 + 5·0 + 6·1 + 5 = 19
#           Row3: 7·2 + 8·0 + 9·1 + 5 = 28
#           Output: [10]
#                   [19]
#                   [28]

# Make weights shape (3,1)
weights = np.zeros((input_dim, output_dim), dtype=np.float32)
weights[0][0] = 0.9         # These were manually set for reference
weights[1][0] = 1.1         
weights[2][0] = 1.3
# Make bias shape (1,)
bias = np.zeros((output_dim,), dtype=np.float32)

# Forward pass
# Output = data @ weights + bias
# f = (Wx + b) For each row: (x * w1) + (y * w2) + (z * w3) + b
output = data @ weights + bias

# We have to set our target values. The Target is the sum of all of the 
# internal components of the row. So we preset all of these target values.
# We are training a small NN to do this thats the whole point.
targets = np.zeros((batch_size, output_dim), dtype=np.float32)

for i in range(batch_size):
    targets[i][0] = data[i][0] + data[i][1] + data[i][2]


# Loss Computation via MSE (Mean-squared-error)
# Steps:
#   1. Find error per row:
#       error = output - targets
#   2. Square it:
#       error²
#   3. Average across all rows:
#       mean(error²)

# Wrapped as a helper function:
def compute_loss(data, targets, weights, bias):
    output = data @ weights + bias
    error = output - targets
    return np.mean(error ** 2)

# weights[0][0]  x
# weights[1][0]  y
# weights[2][0]  z

for epoch in range(epochs):

    current_loss = compute_loss(data, targets, weights, bias)

    # Reweight the gradients
    for i in range(input_dim):
        original = weights[i, 0]

        # probe up
        weights[i, 0] = original + epsilon
        loss_up = compute_loss(data, targets, weights, bias)

        # probe down
        weights[i, 0] = original - epsilon
        loss_down = compute_loss(data, targets, weights, bias)

        # decide direction
        if loss_up < current_loss:
            weights[i, 0] = original + learning_rate
        elif loss_down < current_loss:
            weights[i, 0] = original - learning_rate
        else:
            weights[i, 0] = original

        current_loss = compute_loss(data, targets, weights, bias)

    print(f"Epoch {epoch} | Loss: {current_loss}")
    print(weights.T)









