import math

# Explicitly adjusted hyperparameters
conv_out_channels = 8
conv_kernel_size = 5
conv_stride = 1

pool_kernel_size = 3
pool_stride = 3

lin_out_features = 32

embedding_length = 4

# Implicitly adjusted hyperparameters
lin_in_features = int(math.pow(((26 - conv_kernel_size - pool_kernel_size) \
    /3 + 1), 2)) * conv_out_channels

head_in_features = lin_out_features + embedding_length
