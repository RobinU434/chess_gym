from typing import Any, Dict
from gymnasium import Env
import chess
import numpy as np

from chess_gym.envs.spaces.action_space import ACTION_ENCODING, ChessAction
from chess_gym.envs.spaces.observation_space import BoardEncoding, ChessObservation

class ChessEnv(Env):
    def __init__(
        self,
        action_encoding: ACTION_ENCODING = ACTION_ENCODING.action_wise,
        board_encoding: BoardEncoding = BoardEncoding.PIECE_MAP,
        max_steps: int = 200,
        win_reward: float = 1,
        draw_reward: float = 0,
        lose_reward: float = -1,
        chess_960: bool = False
    ):
        super().__init__()

        self._max_steps = max_steps
        self._win_reward = win_reward
        self._draw_reward = draw_reward
        self._lose_reward = lose_reward
        self._chess_960 = chess_960

        self.action_space: ChessAction = ChessAction(action_encoding)
        self.observation_space: ChessObservation = ChessObservation(action_space=self.action_space, board_encoding=board_encoding)
        self.board = chess.Board(chess960=self._chess_960)

        self._step_counter = 0

    def reset(self, *, seed=None, options=None):
        self.board.reset()
        self._step_counter = 0

    def step(self, action: np.ndarray) -> tuple[Dict[str, str | np.ndarray], float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action)
        uci_action = self.action_space.action_to_uci(action)
        move = chess.Move.from_uci(uci_action)

        self._step_counter += 1
        if self.board.is_legal(move):        
            self.board.push(move)
        
        terminated = self.board.is_checkmate()
        observation = self._observe()
        truncated = self._is_truncated()
        reward = self._get_reward()

        return observation, reward, terminated, truncated, {}

    def render(self):
        return super().render()

    def close(self):
        return super().close()

    def _observe(self) -> Dict[str, str | np.ndarray]:
        actions = np.empty((self.board.legal_moves.count(), *self.action_space.shape))
        for idx, move in enumerate(board_encoding.legal_moves):
            actions[idx] = self._action_space.uci_to_action(move.uci())
        board_encoding = self.observation_space.encode(self.board)
        return {"board": board_encoding, "actions": actions}
    
    def _is_truncated(self) -> bool:
        return self._step_counter >= self._max_steps
    
    def _get_reward(self):
        if self.board.is_checkmate():
            return self._win_reward
        elif self.board.is_variant_draw():
            return self._draw_reward
        else:
            return 0
