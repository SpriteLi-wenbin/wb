import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

tfb = tfp.bijectors
tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels


@tf.function
def optimize(mdl, input, output, loss_func, optimizer):
    with tf.GradientTape() as tape:
        loss = -loss_func(mdl(input), output)
    grads = tape.gradient(loss, mdl.trainable_variables)
    optimizer.apply_gradients(zip(grads, mdl.trainable_variables))
    return loss


kernel = psd_kernels
tf_gpr = tfd.GaussianProcessRegressionModel(kernel=kernel)



