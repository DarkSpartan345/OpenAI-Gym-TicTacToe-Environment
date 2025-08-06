import gymnasium as gym
import numpy as np
import math
from tabulate import tabulate
from typing import Tuple, List
from gymnasium.utils import seeding


class TicTacToeEnv(gym.Env):
    """
    Implementation of a TicTacToe Environment based on Gymnasium standards
    """
    metadata = {
            "render_modes": ["human"],
            "render_fps": 4
        }


    def __init__(self, small: int = -1, large: int = 10,render_mode=None) -> None:
        
        self.render_mode = render_mode
        self.small = small
        self.large = large
        self.fields_per_side = 3

        n_actions = 9
        self.current_player = 1  # X empieza
        self.action_space = gym.spaces.Discrete(n_actions)
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(n_actions,), dtype=int)

        self.colors = [1, 2]
        self.reset(seed=42)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_player = 1  # Reiniciar al jugador 1
        self.state = np.zeros((self.fields_per_side, self.fields_per_side), dtype=int)
        self.info = {"players": {1: {"actions": []}, 2: {"actions": []}}}
        return self.state.flatten(), self.info
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, user_action:int) -> Tuple[np.ndarray, float, bool, bool, dict]:

        """
        Args:
            user_action: Tuple(action:int, color:int)

        Returns:
            obs, reward, terminated, truncated, info
        """
        color = self.current_player
        action= user_action

        if not self.action_space.contains(action):
            raise ValueError(f"action '{action}' is not in action_space")

        if color not in self.colors:
            raise ValueError(f"color '{color}' is not an allowed color")

        reward = self.small
        row, col = self.decode_action(action)

        if self.state[row, col] != 0:
            raise ValueError(f"action '{action}' has already been played!")

        self.state[row, col] = color
        terminated = self._is_winner(color)
        truncated = False  # not using time limits

        if terminated:
            reward += self.large
        self.current_player = 2 if self.current_player == 1 else 1
        self.info["players"][color]["actions"].append(action)
        return self.state.flatten(), reward, terminated, truncated, self.info

    def _is_winner(self, color: int) -> bool:
        s = self.state
        b = s == color
        return any([
            np.all(b[i, :]) for i in range(3)   # rows
        ]) or any([
            np.all(b[:, j]) for j in range(3)   # cols
        ]) or np.all([b[0, 0], b[1, 1], b[2, 2]]) or np.all([b[0, 2], b[1, 1], b[2, 0]])

    def decode_action(self, action: int) -> List[int]:
        col = action % 3
        row = action // 3
        assert 0 <= col < 3
        return [row, col]

    def render(self) -> None:
        board = np.full((3, 3), "-", dtype=str)
        board[self.state == 1] = "X"
        board[self.state == 2] = "O"
        print(tabulate(board, tablefmt="fancy_grid"))


# OPTIONAL: Register with gymnasium
from gymnasium.envs.registration import register

register(
    id="TicTacToeCustom-v0",
    entry_point=TicTacToeEnv,
)

# Usage example
if __name__ == "__main__":
    env = gym.make("TicTacToeCustom-v0", small=-1, large=10)
    obs, info = env.reset()
    env.render()
