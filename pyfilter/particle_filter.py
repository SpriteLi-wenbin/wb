import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as optimize


class ParticleFilter:
    """
    In the case that the post-pdf is unknown, update the particle weights based on least-square algorithms on the
    condition that summary of weights stays at one
    Apply softmax function to the output of scipy.optimize.least_square() to ensure the summary equals one.
    """

    # initialize the particle group as uniform distribution by default
    def __init__(self, size_particle, max_val=1, min_val=0):
        self.size_particle = size_particle
        self.particles = np.array([np.random.uniform(low=min_val, high=max_val, size=size_particle),
                                   np.ones(size_particle) / size_particle])
        return

    # least square error based weights updating and normalizing based on softmax function
    def weight_update(self, loss_fun, x_data, y_data):
        temp = optimize.least_squares(fun=loss_fun, x0=self.particles[1], args=(x_data, y_data))
        # apply softmax function before output
        return self.softmax(temp)

    def state_update(self, op, **kwargs):
        ans = np.zeros(self.particles[0].shape)
        for idx in range(self.size_particle):
            ans[idx] = op(self.particles[0][idx], kwargs)
        return ans

    @staticmethod
    def softmax(weights):
        ans = np.exp(weights) / np.exp(weights).sum()
        return ans

    def post_predict(self):
        ans = np.sum(self.weights * self.particles)
        return ans

    def estimate(self, input, observ, update_func, observ_func, **kwargs):
        def loss_func(weights, x, y):
            temp = self.softmax(weights)
            return (y - observ_func(np.sum(temp * x)), kwargs) ** 2

        length = np.min(len(input), len(observ))
        step = 5
        batch = np.arange(length + 1, step=step)
        if batch[-1] < length:
            batch = np.append(batch, length)
        length = len(batch)
        ans = []

        for index in range(length - 1):
            idx_slice = range(batch[index], batch[index + 1])
            temp_state = []
            for idx in idx_slice:
                temp_state.append(self.state_update(op=update_func, input=input[idx]))  # updating particle status
            self.particles[0] = temp_state[-1]
            # batch size data for least square regression
            self.particles[1] = self.weight_update(loss_fun=loss_func, x_data=temp_state, y_data=observ[idx_slice])
            ans.append(np.sum(self.particles[0] * self.particles[1]))
            self.particles = self.resampling(method='stratified')

        return np.array(ans)

    def resampling(self, method, **kwargs):
        func = {'stratified': self.stratified_resampling}
        return func[method](kwargs)

    # create targeted N-dimension weights within equal N-intervals in cdf format, N is the size of particles;
    # accept the first particles where the accumulated weight >= targeted cdf;
    def stratified_resampling(self):
        ascend_idx_weights = np.argsort(self.particles[1])  # index mapping from sorted weights to original weights
        ans = np.zeros(self.particles.shape)
        accu_weights = self.particles[1][ascend_idx_weights].cumsum()
        # resampling weight points in equal interval with random offset
        pro_weights = (np.arange(self.size_particle) + np.random.random()) / self.size_particle
        idx_pro, idx_accu = 0, 0
        while idx_pro < self.size_particle:
            idx = ascend_idx_weights[idx_accu]
            if pro_weights[idx_pro] < accu_weights[idx_accu]:
                ans[1][idx_pro] = self.particles[1][idx]
                ans[0][idx_pro] = self.particles[0][idx]
                idx_pro += 1
            else:
                idx_accu += 1

        return ans
