from utils import *
from optimization import optimize, exact_GLR_k2
from lowerbound import L_bound, L_naive, optimal_w as lowerbound_optimal_w
from algorithms.TS import TS

# general Separator Track and Stop (GTS-Unknown)
# glr="lb":    GTS-Unknown (ours), stops on min_i L_i(t) > beta(t, delta) and tracks the maximizer of the lower bound
#              (convex). For k = 2 the GLR has a closed form, which is used for both instead of L.
# glr="naive": GTS-Unknown (Naive), stops on min_i L_naive_i(t) > naive_beta(t, delta) and tracks the maximizer of
#              L_naive, for every k
class STS(TS):
    def __init__(self, n, k, confidence, mode = {'fast': True}, glr="lb"):
        super().__init__(n, k, confidence, mode)

        self.N_A = np.zeros(n)
        self.N_Z = np.zeros(k)
        self.cnt_post_actions = np.zeros((n, k))
        self.sum_of_rewards = np.zeros(k)

        self.glr = glr
        self.closed_form = glr == "naive" or k > 2  # stops and samples with a closed-form bound

    def get_mu_hat(self):
        return self.sum_of_rewards / self.N_Z

    def get_A_hat(self):
        return self.cnt_post_actions / self.N_A.reshape((self.n, 1))

    def get_means_hat(self):
        return self.get_A_hat() @ self.get_mu_hat()

    def best_empirical_arm(self):
        mu_hat = self.get_mu_hat()
        A_hat = self.get_A_hat()

        means_hat = np.dot(A_hat, mu_hat)
        delta_hat = np.max(means_hat) - means_hat

        best_arm = np.argmax(means_hat)
        return best_arm, means_hat, delta_hat

    def lambda_hat(self):
        mu_hat = self.get_mu_hat()
        A_hat = self.get_A_hat()
        if self.glr == "naive":
            return np.min(L_naive(mu_hat, A_hat, self.N_A, self.N_Z))
        if self.closed_form:
            return np.min(L_bound(mu_hat, A_hat, self.N_A, self.N_Z))
        return exact_GLR_k2(mu_hat, A_hat, self.N_A, self.N_Z)[0]

    def beta_t_mu(self, delta):
        return (
            self.k * C_exp(np.log(1/delta) / self.k) +
            3 * sum([np.log(1 + np.log(self.N_Z[j])) for j in range(self.k)])
        )

    def beta_t_A(self, delta):
        return (
            np.log(1/delta) +
            (self.k-1) * sum([np.log(np.e * (1 + self.N_A[i] / (self.k-1))) for i in range(self.n)])
        )

    def beta_t(self, delta):
        # This threshold is derived from the combined martingale of categorical and exponential
        return (
            self.k * C_exp(np.log(1/delta) / self.k) +
            3 * sum([np.log(1 + np.log(self.N_Z[j])) for j in range(self.k)]) +
            (self.k-1) * sum([np.log(np.e * (1 + self.N_A[i] / (self.k-1))) for i in range(self.n)])
        )

    def naive_beta_t(self, delta):
        # union bound of the categorical and the exponential family thresholds
        return self.beta_t_A(delta / 2) + self.beta_t_mu(delta / 2)

    def stopping_rule(self):
        # returns True if need to stop and are confident enough
        beta_t = self.naive_beta_t(self.confidence) if self.glr == "naive" else self.beta_t(self.confidence)
        lambda_hat_t = self.lambda_hat()
        return lambda_hat_t > beta_t, lambda_hat_t, beta_t

    def optimal_w(self):
        if self.skip_optimization():
            return self.last_w

        mu_hat = self.get_mu_hat()
        A_hat = self.get_A_hat()
        self.n_optimizations += 1

        if self.closed_form:
            try:
                _, w_star = lowerbound_optimal_w(mu_hat, A_hat, bound="L" if self.glr == "lb" else "naive")
            except Exception as e:
                print("Optimization failed:", e)
                self.optimization_failed_number_of_rounds += 1
                w_star = np.ones(self.n) / self.n if self.last_w is None else self.last_w
        else:
            # warm start, mixed with the uniform allocation: in the softmax parametrization a weight close to 0 has
            # a vanishing gradient, so starting from a corner of the simplex would keep the solver stuck there
            w0 = np.ones(self.n) / self.n if self.last_w is None else (self.last_w + 1 / self.n) / 2
            _, w_star = optimize(mu_hat, A_hat, w0=w0, inner_alg="exact_k2")
        self.last_w = w_star

        return w_star

    def get_action(self):
        # Initialization phase
        if np.any(self.cnt_post_actions == 0):  # explore all of A
            unexplored_a = np.where(self.cnt_post_actions == 0)
            return unexplored_a[0][0], True

        return self.G_tracking()

    def state(self):
        # sufficient statistics, used by glr_trace.py to log a run
        return {
            'T': self.T,
            'cnt_post_actions': self.cnt_post_actions.tolist(),
            'sum_of_rewards': self.sum_of_rewards.tolist(),
        }

    def load_state(self, state):
        self.T = state['T']
        self.cnt_post_actions = np.array(state['cnt_post_actions'])
        self.sum_of_rewards = np.array(state['sum_of_rewards'])
        self.N_A = self.cnt_post_actions.sum(axis=1)
        self.N_Z = self.cnt_post_actions.sum(axis=0)

    def update(self, main_action, post_action, reward):
        self.T += 1
        self.cnt_post_actions[main_action][post_action] += 1
        self.N_A[main_action] += 1
        self.sum_of_rewards[post_action] += reward
        self.N_Z[post_action] += 1

        self.optimization_failed_flag = False
