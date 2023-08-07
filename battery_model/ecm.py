import numpy as np
import scipy.signal as signal
import scipy

'''
State space model based on Scipy
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.StateSpace.html#scipy.signal.StateSpace
'''


class StateSpace:
    stateMatrix = None
    ctrlMatrix = None
    observMatrix = None
    transferMatrix = None
    lenInput = 0
    lenOutput = 0
    lenState = 0
    '''
    discrete state-space function
    x(k + 1) = G*x(k) + H*u(k)
    y(k) = C*x(k) + D*u(k)
    G: state matrix
    H: control matrix
    C: observation matrix
    D: transfer matrix
    '''

    def __init__(self, stateMatrix, ctrlMatrix, observMatrix, transferMatrix):
        self.stateMatrix = np.array(stateMatrix)
        self.ctrlMatrix = np.array(ctrlMatrix)
        self.observMatrix = np.array(observMatrix)
        self.transferMatrix = np.array(transferMatrix)
        self.lenState = self.stateMatrix.shape[-1]
        self.lenInput = self.ctrlMatrix.shape[-1]
        self.lenOutput = self.transferMatrix.shape[-1]

        return

    # calculate y(k)
    def get_output(self, input_state, input_ctrl):
        try:
            ans = np.dot(self.observMatrix, input_state) + np.dot(self.transferMatrix, input_ctrl)
        except ValueError as error:
            print(repr(error))
            return None

        return ans

    # calculate x(k+1)
    def update_state(self, input_state, input_ctrl):
        try:
            ans = np.dot(self.stateMatrix, input_state) + np.dot(self.ctrlMatrix, input_ctrl)
        except ValueError as error:
            print(repr(error))
            return None

        return ans


class ECM:
    '''
    Equivalent circuit model based on transfer equation
    customized state stands for SOC, OCV, SOH,...
    U_component stands for terminal voltage of ohmic resistance, RC parallel circuit,...
    state vector: [customized_1, customized_2,..., U_component_1, U_component_2,..., U_component_n,]
    control vector: [i_load]
    state space function for each component:
        U(s)/I(s) =(a_0 + a_1*s + ... + a_n*s^n)/(b_0 + b_1*s + ... + b_n*s^n)
        introduce bi-linear transformation:
        z = exp(-s*T_s)-> s ≈ (2/T_s)*((z-1)/(z+1))

    parameters:
    impedance: ndarray list
        impedance of each subcomponent in series circuit in complexity format
        for example:
        [array([[R_e], [1]]), array([[R_d], [1,C_d * R_d])]) stands for a Thevinin Model
    '''

    t_sample = None
    capacity = None

    num_state = 0
    num_input = 0
    num_output = 0
    state_space_mdl = None

    '''
    OCV state-space function:
    OCV(n+1) = OCV(n) + diff(OCV, SOC) * T_s/(Capacity) * I_L(n)
    U_t(n) = OCV(n) 
    define charging_current > 0, discharging_current < 0
    '''

    def __init__(self, ts, capacity, diff_OCV=0):
        self.t_sample = ts
        self.capacity = capacity
        A = 1.0
        B = diff_OCV * ts / (3600 * capacity)
        C = 1.0
        D = 0.0
        self.state_space_mdl = signal.dlti(A, B, C, D, dt=ts)
        self.num_state = 1
        self.num_input = 1
        self.num_output = 1
        return

    #  return Jordan discrete state space function using bilinear transformation
    def cal_discrete_sys(self, stransf):
        s_sys = signal.tf2ss(stransf.num, stransf.den)
        # transfer Continuous system into discrete system
        z_sys = signal.cont2discrete(s_sys, method='bilinear', dt=self.t_sample)
        return z_sys

    #  return transfer func in z function format
    def cal_discrete_zfunc(self, stransf):
        zFunc = signal.cont2discrete((stransf.num, stransf.den), method='bilinear', dt=self.t_sample)
        return zFunc

    '''
    U_d(k)= b11*U_d(k-1) + b12*U_d(k-2) + ... + b1n*U_d(k-n) +
            b21*I_L(k) + b22*I_L(k-1) + ... + b2h*I_L(k-h+1)
          =[b_n, c_h] * [U_d, I_L]
    U_d(s)/I_L(s) = r/(rc*s + 1)
    '''

    def cal_RC_zTransFunc(self, r, c):
        sFunc = signal.lti([r], [r * c, 1])
        return self.cal_discrete_zfunc(sFunc)

    #  add ohmic resistance to D in observation function
    def add_ohmicLoss(self, r):
        A, B, C, D = (self.state_space_mdl.A, self.state_space_mdl.B, self.state_space_mdl.C, self.state_space_mdl.D)
        D[0][0] += r
        self.state_space_mdl = signal.dlti(A, B, C, D, dt=self.t_sample)
        return

    '''
    U_e(k) = 1 * U_e(k - 1) + 0 * I_L(k)
    y(k) = 0 * U_e(k) + R_e * I_L(k)
    U_e(s)/I_L(s) = R_e
    '''

    def cal_R_zTransFunc(self, r):
        sFunc = signal.lti(np.array([r]), np.array([1]))
        return self.cal_discrete_zfunc(sFunc)

    '''
    zFunc = (a_n*z^0 + a_(n-1)*z^(1) +... a_0 * z^n)/(b_n*z^0 + b_(n-1)*z^(1) +... b_0 * z^n)
    state_k = [state[k], state[k-1], ..., state[k-ndim+1]]
    state_k= sigma(a_i * state_(k-i)) + sigma(a_i * input_(k-i))
    output = state[k]
    '''
    def zTrans2zsys(self, zFunc):
        numerator = zFunc[0]
        denominator = zFunc[1]
        if len(denominator) > 1:
            A = np.array(-1 * denominator[1:] / denominator[0])
            A = np.reshape(A, (A.size, 1))
            # sts[k] = f(sts[k-1], sts[k-2],...), sts[k-1] = sts[k-1]
            if A.shape[0] > 1:
                A = np.block([[A], [np.block([np.eye(A.shape[0] - 1), np.zeros((1, A.shape[0] - 1))])]])
            C = np.eye(np.shape(A)[0])
        else:
            A = np.zeros((1, 1))
            C = np.eye((1, 1))

        # part of sts[k] = f(input[k], input[k-2],...), part of sts[k-1] = 0 * (input[k], input[k-2],...)
        B = np.array(numerator / denominator[0])
        B = np.reshape(B, (1, B.size))
        if A.shape[0] > 1:
            B = np.block([[B], [np.zeros((A.shape[0] - 1, B.shape[1]))]])

        D = np.zeros((C.shape[0], B.shape[1]))

        return signal.dlti(A, B, C, D, dt=self.t_sample)

    '''
    merge the state space of each component circuit : OCV, RC circuit, resistance,...
    X_d(n+1) = A_d*X_d(n) + B_d*U(n)
    Y(n) = C_d*X_d(n) + D_d*U(n)
    =>
    (X_1, X_2, ...)(n+1) = ((A_1,0,0,...), (0, A_2, -,...),...,(0,...A_n))*(X_1, X_2, ...)(n) + (B_1,...B_n)*I(n)
    U_t(n) = (C_1, C_2, ...C_n)*(X_1, X_2, ...)(n) + sum(-1*D)*I(n)
    '''

    def add_sub_circuit(self, z_sys):

        # update A
        self.num_state += z_sys[0].shape[1]
        shape_LU = self.state_space_mdl.A.shape
        shape_RL = z_sys[0].shape
        A = np.block([
            [self.state_space_mdl.A, np.zeros(shape=(shape_LU[0], shape_RL[1]))],
            [np.zeros(shape=(shape_RL[0], shape_LU[1])), z_sys[0]]
        ])
        # update B
        new_shape = (self.num_state, max(self.num_input, z_sys[1].shape[1]))
        self.num_input = max(self.num_input, z_sys[1].shape[1])
        B = np.zeros(shape=new_shape)
        B[:self.state_space_mdl.B.shape[0], :self.state_space_mdl.B.shape[1]] = self.state_space_mdl.B
        B[self.state_space_mdl.B.shape[0]:, :z_sys[1].shape[1]] = z_sys[1]
        # update C
        C = np.concatenate([self.state_space_mdl.C, z_sys[2]], axis=1)
        C = C.reshape((1, C.size))
        # update D
        new_shape = (self.num_output, max(self.num_input, z_sys[3].shape[1]))
        D = np.zeros(shape=new_shape)
        D[:self.state_space_mdl.D.shape[0], :self.state_space_mdl.D.shape[1]] += self.state_space_mdl.D
        D[:z_sys[3].shape[0], :z_sys[3].shape[1]] += z_sys[3]


        self.state_space_mdl = signal.dlti(A, B, C, D, dt=self.t_sample)

        return

    def sim_mdl(self, input, t=None, X_0=None):
        input = np.array(input)
        if self.num_input != input.size / len(input):
            return
        if X_0 is None:
            X_0 = np.zeros(self.num_state)
        if t is None:
            t = np.arange(len(input))
        ans = signal.dlsim(self.state_space_mdl, input, t, x0=X_0)
        return ans
