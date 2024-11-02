from io import BytesIO
from typing import Any, Dict, Tuple

import cairosvg
import chess
import numpy as np
from gymnasium import Env
from PIL import Image

from chess_gym.spaces.action_space import ACTION_ENCODING, ChessAction
from chess_gym.spaces.observation_space import BoardEncoding, ChessObservation

from gymnasium.envs.classic_control


class ChessEnv(Env):
    def __init__(
        self,
        action_encoding: ACTION_ENCODING = ACTION_ENCODING.action_wise,
        board_encoding: BoardEncoding = BoardEncoding.PIECE_MAP,
        max_steps: int = 200,
        win_reward: float = 1,
        draw_reward: float = 0,
        lose_reward: float = -1,
        chess_960: bool = False,
        render_size: Tuple[int, int] = (400, 400),
    ):
        # only supports rgb_array as a render mode. If you would like to have human please have a look at: https://gymnasium.farama.org/main/_modules/gymnasium/wrappers/human_rendering/
        self.metadata["render_mode"].append("rgb_array")
        super().__init__()

        self._max_steps = max_steps
        self._win_reward = win_reward
        self._draw_reward = draw_reward
        self._lose_reward = lose_reward
        self._chess_960 = chess_960
        self._render_size = render_size

        self.action_space: ChessAction = ChessAction(action_encoding)
        self.observation_space: ChessObservation = ChessObservation(
            action_space=self.action_space, board_encoding=board_encoding
        )
        self.board = chess.Board(chess960=self._chess_960)

        self._step_counter = 0

    def reset(self, *, seed=None, options=None):
        self.board.reset()
        self._step_counter = 0

    def step(
        self, action: np.ndarray
    ) -> tuple[Dict[str, str | np.ndarray], float, bool, bool, dict[str, Any]]:
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
        img = self._get_image()
        return img
        
    def close(self):
        return super().close()

    def _observe(self) -> Dict[str, str | np.ndarray]:
        actions = np.empty((self.board.legal_moves.count(), *self.action_space.shape))
        for idx, move in enumerate(self.board.legal_moves):
            actions[idx] = self.action_space.uci_to_action(move.uci())
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
        
    def _get_image(self) -> np.ndarray:
        out = BytesIO()
        bytestring = chess.svg.board(self.board, size = self.render_size).encode('utf-8')
        cairosvg.svg2png(bytestring = bytestring, write_to = out)
        image = Image.open(out)
        return np.asarray(image)
