from utils import *
from abc import ABC, abstractmethod

# Base Track and Stop
class TS(ABC):
    def __init__(self, n, k, confidence, mode = {'fast': True}):
        self.n = n
        self.k = k
        self.confidence = confidence

        self.T = 0

        self.mode = mode

        self.last_w = None  # last solution of the sampling rule optimization
        self.n_optimizations = 0

        self.optimization_failed_flag = False
        self.optimization_failed_number_of_rounds = 0

    def skip_optimization(self):
        # fast mode: once T has d digits, the optimal w is only recomputed every 10^(d-2) rounds
        if not self.mode.get('fast', False) or self.last_w is None:
            return False
        period = 10 ** max(len(str(self.T)) - 2, 0)
        return self.T % period != 0

    @abstractmethod
    def stopping_rule(self):
        pass

    @abstractmethod
    def optimal_w(self):
        pass

    def G_projection(self, w, v):
        if np.array_equal(w, v):  # already too close to w
            return w
        dir = w - v
        t = np.divide(dir, w, out=np.zeros_like(w), where=dir!=0)
        if np.min(t) == 0:
            return w
        proj = w - dir / np.min(t)
        return proj

    def G_tracking(self):
        # Forced exploration
        if np.sqrt(self.T) % 1 == 0:
            # Selects an arm at random
            arm = np.random.randint(0, self.n)
            return arm, False

        # Direct tracking
        w = self.optimal_w()
        w_projected = self.G_projection(w, self.N_A / sum(self.N_A))
        target = w_projected * np.sum(self.N_A)

        result = target - self.N_A
        return np.argmax(result), False

    @abstractmethod
    def get_action(self):
        pass

    @abstractmethod
    def update(self, main_action, post_action, reward):
        pass
