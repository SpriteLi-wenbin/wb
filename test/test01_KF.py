import numpy as np
import pytest_cov
import pfilter.KalmanFilter as KalmanFilter


# create test object
def step_1():
    return KalmanFilter.KF()


class TestClass:

    def test_init_01(self):
        num_state = 3
        num_input = 1
        num_output = 1

        test_object = KalmanFilter.KF(num_state, num_input, num_output)

        assert np.all(test_object.xhat_prior == np.zeros((num_state, 1)))
        assert np.all(test_object.xhat_post == np.zeros((num_state, 1)))

        assert np.all(test_object.shape_transit_matrix == (num_state, num_state))
        assert np.all(test_object.shape_input_matrix == (num_state, num_input))
        assert np.all(test_object.shape_obsv_matrix == (num_output, num_state))
        assert np.all(test_object.shape_cov == (num_state, num_state))
        assert np.all(test_object.shape_process_noise == (num_state, num_state))
        assert np.all(test_object.shape_measure_noise == (num_output, num_output))
        assert np.all(test_object.shape_kalman_gain == (num_state, num_output))

        assert np.all(test_object.prior_cov == 0)
        assert np.all(test_object.post_cov == 0)
        assert np.all(test_object.kalman_gain == 0)

        return

    def test_init_02(self):
        num_state = 3
        num_input = 1
        num_output = 1
        init_state = [1, 2, 3]

        test_object = KalmanFilter.KF(num_state, num_input, num_output, init_state)

        assert np.all(test_object.xhat_prior == np.reshape(np.array(init_state), (np.array(init_state).size, 1)))
        assert np.all(test_object.xhat_post == np.reshape(np.array(init_state), (np.array(init_state).size, 1)))

        return

    def test_symmetrize(self):
        arg = np.array([1, 2, 3])
        assert KalmanFilter.KF.symmetrize(arg) is None
        arg = np.array([[1, 2, 3], [4, 5, 6]])
        assert KalmanFilter.KF.symmetrize(arg) is None
        arg = np.array([[1, 2], [3, 4]])
        arg = KalmanFilter.KF.symmetrize(arg)
        assert np.all(arg == np.transpose(arg))

        return

    def test_prior_estimate(self):
        num_state = 3
        num_input = 1
        num_output = 1
        transmit_matrix = np.eye(3)
        input_matrix = np.array([[1], [2], [3]])
        Q = np.zeros((3, 3))
        state_label = np.ones((num_state, 1))
        state_pred = np.zeros((num_state, 1))
        cov = np.dot((state_pred - state_label), np.transpose((state_pred - state_label)))

        test_object = KalmanFilter.KF(num_state, num_input, num_output)
        test_object.post_cov = cov
        inputs = 1
        test_object.prior_estimate(inputs, transmit_matrix, input_matrix, Q)
        state_label = np.linalg.multi_dot([transmit_matrix, state_label]) + np.linalg.multi_dot([input_matrix, inputs])
        state_pred = np.linalg.multi_dot([transmit_matrix, state_pred]) + np.linalg.multi_dot([input_matrix, inputs])
        cov = np.dot((state_pred - state_label), np.transpose((state_pred - state_label)))

        assert np.all(test_object.xhat_prior == state_pred)
        assert np.all(test_object.prior_cov == cov)

        return

    def test_kalman_predict(self):
        num_state = 3
        num_input = 1
        num_output = 1
        transmit_matrix = np.eye(3)
        input_matrix = np.array([[1], [2], [3]])
        obsv_matrix = np.array([1, 2, 4]).reshape((num_output, num_state))
        Q = np.zeros((num_state, num_state)) + 10
        R = np.zeros((num_output, num_output)) + 0.001
        state_init = np.ones((num_state, 1))
        inputs = 0

        test_object = KalmanFilter.KF(num_state, num_input, num_output, np.array([1.2, 0.9, 1.2]).reshape((num_state, 1)))
        ans = [test_object.xhat_post]
        len_iterate = 50
        err = [test_object.xhat_post - state_init]

        for idx in range(len_iterate):
            state_init = np.dot(transmit_matrix, state_init) + np.dot(input_matrix, inputs)
            y = np.linalg.multi_dot([obsv_matrix, state_init]) + np.random.normal(0, 0.001)
            ans.append(test_object.kalman_predict(inputs, y, transmit_matrix, input_matrix, obsv_matrix, Q, R))
            err.append(test_object.xhat_post - state_init)

        return

    def test_FLS(self):
        num_state = 3
        num_input = 1
        num_output = 1
        transmit_matrix = np.eye(3)
        input_matrix = np.array([[1], [2], [3]])
        obsv_matrix = np.array([1, 2, 4]).reshape((num_output, num_state))
        Q = np.zeros((num_state, num_state)) + 10
        R = np.zeros((num_output, num_output)) + 0.001

        len_data = 100
        inputs = np.random.uniform(size=len_data).reshape((len_data, num_input, 1))
        state = [np.ones(num_state).reshape((num_state, 1))]
        y = [np.dot(obsv_matrix, np.dot(transmit_matrix, state[0])) + np.random.normal(0, R[0], (num_output, 1))]
        for idx in range(1, len_data):
            state.append(np.dot(transmit_matrix, state[-1]).reshape((num_state, 1)) +
                         np.dot(input_matrix, inputs[idx]).reshape((num_state, 1)))
            y.append(np.dot(obsv_matrix, state[-1]).reshape((num_output, 1)) +
                     np.random.normal(0, R[0], (num_output, 1)))

        y = np.array(y)
        lag = 10

        test_object = KalmanFilter.KF(num_state, num_input, num_output, np.array([1.2, 0.9, 1.2]).reshape((num_state, 1)))

        ans = test_object.FLS_Kalman(inputs, y, lag, transmit_matrix, input_matrix, obsv_matrix, Q, R)

        return

    def test_RTS(self):
        num_state = 3
        num_input = 1
        num_output = 1
        transmit_matrix = np.eye(3)
        input_matrix = np.array([[1], [2], [3]])
        obsv_matrix = np.array([1, 2, 4]).reshape((num_output, num_state))
        Q = np.zeros((num_state, num_state)) + 10
        R = np.zeros((num_output, num_output)) + 0.001

        len_data = 100
        inputs = np.random.uniform(size=len_data).reshape((len_data, num_input, 1))
        state = [np.ones(num_state).reshape((num_state, 1))]
        y = [np.dot(obsv_matrix, np.dot(transmit_matrix, state[0])) + np.random.normal(0, R[0], (num_output, 1))]

        return