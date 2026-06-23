from __future__ import annotations

from collections import deque

import numpy as np


class StateHistory:
    def __init__(self, sequence_length: int, state_dim: int):
        self.sequence_length = sequence_length
        self.state_dim = state_dim
        self._items: deque[np.ndarray] = deque(maxlen=sequence_length)
        self.reset()

    def reset(self) -> np.ndarray:
        self._items.clear()
        zero = np.zeros(self.state_dim, dtype=np.float32)
        for _ in range(self.sequence_length):
            self._items.append(zero.copy())
        return self.as_array()

    def append(self, state: np.ndarray) -> np.ndarray:
        arr = np.asarray(state, dtype=np.float32)
        if arr.shape != (self.state_dim,):
            raise ValueError(f"expected state shape {(self.state_dim,)}, got {arr.shape}")
        self._items.append(arr)
        return self.as_array()

    def as_array(self) -> np.ndarray:
        return np.stack(list(self._items), axis=0)
