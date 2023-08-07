import numpy as np
import scipy.linalg as linalg


class KF:
    '''
    x{k+1}=F_k*x{k}+G_k*u{k}+w{k}, w~N(0,Q)  <- transition function
    y{k+1} = H_k*x{k+1}+v{k}, v~N(0,R) <- observation function
    Step:
    1. prior estimate: x_hat{k|k-1} = F_{k}*x_hat{k-1|k-1}+G_{k}*u{k}
    2. prior covariance update: prior_cov_{k} = F_{k-1}*post_cov_P_{k-1}*transpose(F_{k-1})+Q_{k-1}
    3. Kalman Gain update: gain_{k} = prior_cov_{k}*H_{k}*inv(H_{k}*prior_cov_{k}*transpose(H_{k})+R_{k})
    4. post estimate: x_hat{k|k} = x_hat{k|k-1} + gain{k} * (y_{k} - H_{k} * x_hat{k|k-1})
    5. post covariance update: post_cov_P_{k} = prior_cov_{k} - gain_{k} * H_{k} * prior_cov_{k}
    '''

    def __init__(self, size_state, size_input, size_output, x_0=None, p_0=None):

        # state initialize
        if x_0 is None:
            self.xhat_prior = np.zeros((size_state, 1))
            self.xhat_post = np.zeros((size_state, 1))
        else:
            init_state = np.array(x_0)
            size_state = init_state.size
            self.xhat_prior = np.reshape(init_state, (init_state.size, 1))
            self.xhat_post = np.reshape(init_state, (init_state.size, 1))

        # matrix initialize
        self.shape_transit_matrix = (size_state, size_state)
        self.shape_input_matrix = (size_state, size_input)
        self.shape_obsv_matrix = (size_output, size_state)
        self.shape_cov = (size_state, size_state)
        self.shape_process_noise = (size_state, size_state)
        self.shape_measure_noise = (size_output, size_output)

        # covariance & kalman gain initialize
        if p_0 is None:
            self.prior_cov = 10**6 * np.eye(self.shape_cov[0])
            self.post_cov = 10**6 * np.eye(self.shape_cov[0])
        else:
            self.prior_cov = np.array(p_0).reshape(self.shape_cov)
            self.post_cov = np.array(p_0).reshape(self.shape_cov)

        return

    @staticmethod
    # make covariance symmetric
    def symmetrize(arg):
        arg = np.array(arg)
        if arg.ndim != 2:
            return None
        if arg.shape[0] != arg.shape[1]:
            return None

        return (arg + np.transpose(arg)) / 2

    '''
    perform prior estimation
    x_hat = F * x_hat + G * u
    cov_prior = F * cov_post * transpose(F) + R
    '''

    def prior_estimate(self, u, transmit_matrix, input_matrix, Q):
        transmit_matrix = np.array(transmit_matrix)
        input_matrix = np.array(input_matrix)
        Q = np.array(Q)

        # prior estimate
        self.xhat_prior = np.dot(transmit_matrix, self.xhat_post) + np.dot(input_matrix, u)
        self.prior_cov = np.linalg.multi_dot([transmit_matrix, self.post_cov, np.transpose(transmit_matrix)]) + Q
        self.prior_cov = self.symmetrize(self.prior_cov)

        return self.xhat_prior, self.prior_cov

    '''
    perform posterior estimation
    y_hat = H * x_hat
    x_hat = x_hat + K_gain(y - y_hat)
    cov_post = cov_prior - K_gain * H * cov_prior
    '''

    def post_estimate(self, output_error, obsv_matrix, R):

        s = np.linalg.multi_dot([obsv_matrix, self.prior_cov, np.transpose(obsv_matrix)]) + R  # H*Cov_prior*H_T+R

        # zero denominator validation
        if np.linalg.det(s) == 0:
            s = 100 * np.eye(R.shape)
        else:
            s = np.linalg.inv(s)
        kalman_gain = np.linalg.multi_dot([self.prior_cov, np.transpose(obsv_matrix), s])  # Cov_prior*H_T*(...)^-1
        self.xhat_post = self.xhat_prior + np.dot(kalman_gain, output_error)
        self.post_cov = self.prior_cov - np.linalg.multi_dot([kalman_gain, obsv_matrix, self.prior_cov])
        self.post_cov = self.symmetrize(self.post_cov)

        return self.xhat_post, self.post_cov

    @staticmethod
    def shape_validate(matrix, shape):
        matrix = np.array(matrix)
        if matrix.shape != shape:
            return False
        else:
            return True

    @staticmethod
    def matrix_validate(a, b):
        if (np.ndim(a) != np.ndim(b)) or (np.ndim(a) != 2):
            return False

        if np.shape(a)[1] == np.shape(b)[0]:
            return True
        else:
            return False

    def kalman_predict(self, u, y, transmit_matrix, input_matrix, obsv_matrix, output_matrix, Q, R):
        u = np.array(u)
        u = np.reshape(u, (u.size, 1))
        y = np.array(y)
        y = np.reshape(y, (y.size, 1))
        if not self.shape_validate(u, (self.shape_input_matrix[1], 1)):
            return None
        if not self.shape_validate(y, (self.shape_obsv_matrix[0], 1)):
            return None

        transmit_matrix = np.array(transmit_matrix)
        if not self.matrix_validate(transmit_matrix, self.xhat_post):
            return None
        input_matrix = np.array(input_matrix)
        if not self.matrix_validate(input_matrix, u):
            return None
        obsv_matrix = np.array(obsv_matrix)
        if not self.matrix_validate(obsv_matrix, self.xhat_prior):
            return None
        Q = np.array(Q)
        if not self.shape_validate(Q, self.shape_process_noise):
            return None
        R = np.array(R)
        if not self.shape_validate(R, self.shape_measure_noise):
            return None

        temp = self.prior_estimate(u, transmit_matrix, input_matrix, Q)
        '''
        #  debug
        temp = self.xhat_prior
        temp[1] = min(0.02, temp[1])
        temp[2] = min(0.02, temp[2])
        # debug end
        '''
        temp = y - (obsv_matrix.dot(self.xhat_prior) + output_matrix.dot(u))
        temp = self.post_estimate(temp, obsv_matrix, R)

        return self.xhat_post

    '''
    perform kalman fixed lag smoothing to estimate xhat[k - M|y(1), y(2), ..., y(k)]
    for M lag smoother:
    xhat_new[k] = transpose([xhat[k], xhat[k|1], xhat[k|2], ... ,xhat[k|M]])
    F_new[k] = [[F[k], 0, 0, ..., 0], [I, 0, ...,0], [0, I,...,0], [0,..., I, 0]]
    G_new[k] = transpose([G[k], 0, ..., 0])
    H_new[k] = [H[k], 0, ..., 0]
    w_new[k] = transpose([I, 0, ..., 0]) * w[k]
    v_new[k] = v[k]
    x_new[k] = F_new[k-1] * x[k-1] + G[k-1] * u[k-1] + w_new[k-1]
    y[k] = H_new[k] * x[k] + v_new[k]
    '''

    def FLS_Kalman(self, u, y, lag, transmit_matrix, input_matrix, obsv_matrix, output_matrix, Q, R,
                   x_0=None, p_0=None):
        # not enough measurement for fixed lag smoothing
        if min(len(u), len(y)) <= lag:
            return None
        num_state = transmit_matrix.shape[0]

        # reform matrix and noise term
        transmit_matrix = np.block([transmit_matrix, np.zeros((self.shape_transit_matrix[0],
                                                               lag * self.shape_transit_matrix[1]))])
        transmit_matrix = np.block([[transmit_matrix],
                                    [np.eye(lag * self.shape_transit_matrix[0],
                                            (lag + 1) * self.shape_transit_matrix[1])]])
        input_matrix = np.block([[input_matrix], [np.zeros((lag * (self.shape_input_matrix[0]),
                                                            self.shape_input_matrix[1]))]])
        obsv_matrix = np.block([obsv_matrix, np.zeros((self.shape_obsv_matrix[0],
                                                       lag * self.shape_obsv_matrix[1]))])
        temp = np.block([[np.eye(self.shape_transit_matrix[0])],
                         [np.zeros((lag * self.shape_transit_matrix[0], self.shape_transit_matrix[1]))]])
        Q = np.linalg.multi_dot([temp, Q, np.transpose(temp)])
        temp = np.zeros(transmit_matrix.shape[0])
        if not x_0 is None:
            temp[:num_state] = x_0
        x_0 = np.reshape(temp, (temp.size, 1))
        temp = 10**6 * np.eye(transmit_matrix.shape[0])
        if not p_0 is None:
            temp[:num_state, :num_state] = p_0
            p_0 = temp
        # perform kalman predict with new matrix
        temp = KF(transmit_matrix.shape[0], input_matrix.shape[1], obsv_matrix.shape[0], x_0=x_0, p_0=p_0)
        ans = []
        for i in range(min(len(u), len(y))):
            temp.kalman_predict(u[i], y[i], transmit_matrix, input_matrix, obsv_matrix, output_matrix, Q, R)
            ans.append(temp.xhat_post)

        return [np.array(item[-self.xhat_post.shape[0]:]).reshape(self.xhat_post.shape) for item in ans[lag:]]

    '''
    perform RTS fixed interval smoothing
    '''

    def RTS_Kalman(self, u, y, transmit_matrix, input_matrix, obsv_matrix, Q, R):
        length = min(len(u), len(y))
        if length < 2:
            return None
        u = u[0:length]
        y = y[0:length]

        transmit_matrix = np.array(transmit_matrix)
        input_matrix = np.array(input_matrix)
        obsv_matrix = np.array(obsv_matrix)
        Q = np.array(Q)
        R = np.array(R)

        if min(np.ndim(transmit_matrix), np.ndim(input_matrix), np.ndim(obsv_matrix),
               np.ndim(Q), np.ndim(R)) < 2:
            return None

        flag_constant_matrix = False
        # constant matrix
        if min(np.ndim(transmit_matrix), np.ndim(input_matrix), np.ndim(obsv_matrix),
               np.ndim(Q), np.ndim(R)) == 2:
            flag_constant_matrix = True
            transmit_matrix = np.reshape(transmit_matrix,
                                         (transmit_matrix.shape[0], transmit_matrix.shape[1], 1))
            input_matrix = np.reshape(input_matrix,
                                      (input_matrix.shape[0], input_matrix.shape[1], 1))
            obsv_matrix = np.reshape(obsv_matrix,
                                     (obsv_matrix.shape[0], obsv_matrix.shape[1], 1))

        # varying matrix
        if ((min(len(transmit_matrix), len(input_matrix), len(obsv_matrix), len(Q), len(R)) < length) and
                (not flag_constant_matrix)):
            return None

        # forward smoothing
        temp = KF(transmit_matrix[0].shape[1], input_matrix[0].shape[1], obsv_matrix[0].shape[0])
        val_est_fwd = []
        prior_cov = []
        post_cov = []
        for idx in range(length):
            idx_matrix = 0 if flag_constant_matrix else idx
            if not ((self.shape_validate(transmit_matrix[idx_matrix], self.shape_transit_matrix)) and
                    (self.shape_validate(input_matrix[idx_matrix], self.shape_input_matrix)) and
                    (self.shape_validate(obsv_matrix[idx_matrix], self.shape_obsv_matrix)) and
                    (self.shape_validate(Q[idx_matrix], self.shape_process_noise)) and
                    (self.shape_validate(R[idx_matrix], self.shape_measure_noise))):
                return None
            temp.kalman_predict(u[idx], y[idx], transmit_matrix[idx_matrix], input_matrix[idx_matrix],
                                obsv_matrix[idx_matrix], Q[idx_matrix], R[idx_matrix])
            val_est_fwd.append(temp.xhat_post)
            prior_cov.append(temp.prior_cov)
            post_cov.append(temp.post_cov)

        # backward smoothing
        val_est = np.zeros((self.shape_transit_matrix[0], length))
        val_est[-1] = val_est_fwd[-1]
        for idx in range(length - 2, -1, -1):
            val_est[idx] = val_est_fwd[idx] + np.linalg.multi_dot([post_cov[idx], np.transpose(transmit_matrix[idx]),
                                                                   np.linalg.inv(post_cov[idx + 1]),
                                                                   (val_est[idx + 1] - val_est_fwd[idx + 1])])

        return val_est
