import functools
import math
import time
from typing import Optional

import numpy as np

EPSILON_WEIGHT = 0.02

EPSILON = EPSILON_WEIGHT * 10 ** (18 / 3)

# 논문에서 제시한 값들 (최적화 검토 필요)
# 10 ** (15 / 3)
# 10 ** (16 / 3)
# 10 ** (17 / 3)
# 10 ** (18 / 3)
# 10 ** (19 / 3)
# 10 ** (20 / 3)
# 10 ** (21 / 3)


class Arm:

    def __init__(self, created_at: int, decay_dimension: int = 9):

        self.created_at = created_at
        self.decay_dimension = decay_dimension
        self.A = np.array([[0] * self.decay_dimension] * self.decay_dimension)
        self.B = np.array([[0]] * self.decay_dimension)

    @property
    def theta(self):
        A_inv = np.linalg.pinv(self.A)
        theta = np.dot(A_inv, self.B)
        return theta

    def confidence(self, current_timestamp: int) -> float:
        """Confidence Upper Bound (UCB)"""

        def vector_norm(timestamp: int) -> float:
            x = self.time_decay_vector(timestamp)
            A_inv = np.linalg.pinv(self.A)
            norm = np.sqrt(x.T @ A_inv @ x)
            norm = norm[0][0]
            return norm

        def a(timestamp: int) -> float:
            return math.sqrt(math.log(timestamp, math.e))

        result = a(current_timestamp) * vector_norm(current_timestamp)
        return result

    def pull(self, current_timestamp: Optional[int] = None, confidence: bool = False) -> float:
        """Predict an expected reward."""
        current_timestamp = current_timestamp or int(time.time())
        x = self.time_decay_vector(current_timestamp)
        prediction = np.dot(x.T, self.theta)[0][0]
        if confidence:
            prediction += self.confidence(current_timestamp)
        return prediction

    def update(self, reward: bool, current_timestamp: Optional[int] = None):
        current_timestamp = current_timestamp or int(time.time())
        decay_vector = self.time_decay_vector(current_timestamp)
        self.A = self.A + np.outer(decay_vector, decay_vector)
        self.B = self.B + (int(reward) * decay_vector)
        print("")

    def time_decay_vector(self, current_timestamp: float) -> np.array:
        """시간 감쇄 벡터를 반환 한다."""
        def _decay(index: int, timestamp: float) -> float:
            timestamp_delta = timestamp - self.created_at
            weight = 2 ** index
            value = weight * timestamp_delta / EPSILON
            result = 1 / math.sqrt(value + 1)
            return result

        decay = functools.partial(_decay, timestamp=current_timestamp)
        decay_vector = [[decay(index)] for index in range(self.decay_dimension)]
        decay_vector = np.array(decay_vector)
        return decay_vector
