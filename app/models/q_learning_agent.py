import numpy as np
from typing import Optional, Tuple

class QLearningAgent:
    def __init__(
        self,
        n_bins: int = 31,
        total_steps: int = 10_000,
        lr: float = 0.3,
        gamma: float = 0.99,
        eps: float = 1.0,
    ) -> None:
        self.n_bins = n_bins
        self.q_table = np.zeros((n_bins, n_bins, n_bins), dtype=np.float32)
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.global_step = 0
        self.total_steps = total_steps

        self.episode_return = 0.0
        self.returns: list[dict] = []
        self.last_total: Optional[float] = None
        self.last_avg: Optional[float] = None
        self.last_steps: Optional[int] = None

    @staticmethod
    def _bin(value: float, v_min: float, v_max: float, n: int) -> int:
        return int(np.clip((value - v_min) / (v_max - v_min) * n, 0, n - 1))

    def _state(self, room_t: float, out_t: float) -> Tuple[int, int]:
        return (
            self._bin(room_t, 10.0, 30.0, self.n_bins),
            self._bin(out_t, -10.0, 30.0, self.n_bins),
        )

    def act(self, room_t: float, out_t: float) -> Tuple[float, Tuple[int, int], int]:
        room_bin, out_bin = self._state(room_t, out_t)
        if np.random.rand() < self.eps:
            action_bin = np.random.randint(self.n_bins)
        else:
            action_bin = int(np.argmax(self.q_table[room_bin, out_bin]))
        pct = action_bin / (self.n_bins - 1)
        return pct, (room_bin, out_bin), action_bin

    def learn(
        self,
        state: Tuple[int, int],
        action_bin: int,
        reward: float,
        next_state: Tuple[int, int],
    ) -> bool:
        s_r, s_o = state
        ns_r, ns_o = next_state
        best_next = float(np.max(self.q_table[ns_r, ns_o]))
        td_error = reward + self.gamma * best_next - self.q_table[s_r, s_o, action_bin]
        self.q_table[s_r, s_o, action_bin] += self.lr * td_error
        self.global_step += 1
        self.eps = max(0.01, self.eps * 0.995)
        self.episode_return += reward
        done = self.global_step >= self.total_steps - 1
        if done:
            avg = self.episode_return / self.global_step
            self.returns.append(
                {"total": self.episode_return, "avg": avg, "steps": self.global_step}
            )
            self.last_total = self.episode_return
            self.last_avg = avg
            self.last_steps = self.global_step
        return done
