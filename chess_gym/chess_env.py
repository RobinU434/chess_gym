from io import BytesIO
from typing import Any, Dict, Optional, Tuple

import cairosvg
import chess
import chess.pgn
import chess.svg
import numpy as np
from chess import Board
from gymnasium import Env
from gymnasium.error import DependencyNotInstalled
from gymnasium.spaces import MultiBinary
from PIL import Image

from chess_gym.spaces import ChessAction
from chess_gym.utils import fen2one_hot
from chess_gym.agent import _Agent, RandomAgent


class ChessEnv(Env):
    metadata = {"render_modes": ["rbg_array", "human"], "render_fps": 1}

    def __init__(
        self,
        max_steps: int = 200,
        win_reward: float = 1,
        draw_reward: float = 0,
        lose_reward: float = -1,
        chess_960: bool = False,
        render_size: int = 400,
        render_mode: Optional[str] = None,
        seed: int = None,
    ):
        super().__init__()
        # only supports rgb_array as a render mode. If you would like to have human please have a look at: https://gymnasium.farama.org/main/_modules/gymnasium/wrappers/human_rendering/
        self._max_steps = max_steps
        self._win_reward = win_reward
        self._draw_reward = draw_reward
        self._lose_reward = lose_reward
        self._chess_960 = chess_960
        self._render_size = render_size
        self.render_mode = render_mode

        self.action_space: ChessAction = ChessAction(seed=seed)
        self.observation_space = MultiBinary(
            (8, 8, 12)
        )  # BoardSpace(BoardEncoding.PIECE_MAP) #
        self.board = chess.Board(chess960=self._chess_960)

        self._step_counter = 0

        # render with pygame
        self.screen = None
        self.clock = None

    def reset(self, *, seed=None, options=None):
        self.board.reset()
        self._step_counter = 0
        board_state, info = self._observe()
        info = {**self._get_info(), **info}
        return board_state, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action)
        uci_action = self.action_space.to_uci(action)
        move = chess.Move.from_uci(uci_action)

        self._step_counter += 1
        if self.board.is_legal(move):
            self.board.push(move)

        terminated = self.board.is_checkmate()
        board_state, info = self._observe()
        info = {**self._get_info(), **info}
        truncated = self._is_truncated()
        reward = self._get_reward()

        return board_state, reward, terminated, truncated, info

    def uci_step(self, action: str):
        action = self.action_space.from_uci(action)
        return self.step(action)

    def render(self):
        if self.render_mode is None:
            return
        elif self.render_mode == "rgb_array":
            return np.asarray(self._get_image())
        elif self.render_mode == "human":
            if self.screen is None:
                try:
                    import pygame
                    from pygame import gfxdraw  # noqa: F401
                except ImportError:
                    raise DependencyNotInstalled(
                        "pygame is not installed, run `pip install gym[box2d]`"
                    )
                pygame.init()
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self._render_size, self._render_size)
                )
            if self.clock is None:
                self.clock = pygame.time.Clock()
            rgb_array = np.asarray(self._get_image()).astype(float)
            rgb_array = np.swapaxes(rgb_array, axis1=0, axis2=1)
            surface = pygame.surfarray.make_surface(rgb_array[..., :-1])
            self.screen.blit(surface, (0, 0))
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:
            raise NotImplementedError(f"no implementation for {self.render_mode=}")

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

        return super().close()

    def _get_uci_actions(self, board: Board) -> np.ndarray:
        actions = []
        for move in self.board.legal_moves:
            actions.append(move.uci())
        actions = np.array(actions)
        return actions

    def _observe(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        board_encoding = fen2one_hot(self.board.fen().split(" ")[0])
        available_uci_actions = self._get_uci_actions(self.board)
        available_actions = self.action_space.from_uci(available_uci_actions)
        info = {
            "available_actions": available_actions,
            "available_uci_actions": available_uci_actions,
        }
        return board_encoding, info

    def _is_truncated(self) -> bool:
        return self._step_counter >= self._max_steps

    def _get_reward(self):
        if self.board.is_checkmate():
            return self._win_reward
        elif self.board.is_variant_draw():
            return self._draw_reward
        else:
            return 0

    def _get_info(self):
        info = {
            "next_turn": self.board.turn,
            "castling_rights": self.board.castling_rights,
            "fullmove_number": self.board.fullmove_number,
            "halfmove_clock": self.board.halfmove_clock,
            "promoted": self.board.promoted,
            "chess960": self.board.chess960,
            "ep_square": self.board.ep_square,
        }
        return info

    def _get_image(self) -> np.ndarray:
        bytes = self._get_image_bytes()
        image = Image.open(bytes)
        return image

    def _get_image_bytes(self) -> BytesIO:
        out = BytesIO()
        bytestring = chess.svg.board(self.board, size=self._render_size).encode("utf-8")
        cairosvg.svg2png(bytestring=bytestring, write_to=out)
        return out


class SinglePlayerChess(ChessEnv):
    def __init__(
        self,
        opponent: _Agent = None,
        max_steps=200,
        win_reward=1,
        draw_reward=0,
        lose_reward=-1,
        chess_960=False,
        render_size=400,
        render_mode=None,
        seed=None,
        player_starts: bool = True,
    ):
        super().__init__(
            max_steps,
            win_reward,
            draw_reward,
            lose_reward,
            chess_960,
            render_size,
            render_mode,
            seed,
        )

        self.opponent = opponent
        if self.opponent is None:
            self.opponent = RandomAgent()
        self.player_starts = player_starts

    def reset(self, *, seed=None, options=None, player_starts: bool = True):
        self.player_starts = player_starts
        obs, info = super().reset()
        if player_starts:
            return obs, info
        
        obs = self._observe_opponent()
        obs, _, _, _, info = self._opponent_step(obs, info)
        self._step_counter = 0
        return obs, info
        
    def _reorder_board_state(self, obs: np.ndarray) -> np.ndarray:
        shuffle_idx = np.array([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5])
        obs = obs[shuffle_idx]
        return obs

    def _observe_opponent(self):
        obs, info = self._observe()
        if self.player_starts:
            obs = self._reorder_board_state(obs)
        return obs, info

    def _observe(self):
        obs, info = super()._observe()
        if not self.player_starts:
            obs = self._reorder_board_state(obs)
        return obs, info
    
    def _opponent_step(self, obs, info):
        opponent_action = self.opponent.act(obs, info)
        obs, reward, done, truncated, info = super().step(opponent_action)
        self._step_counter -= 1
        return obs, reward, done, truncated, info

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        if done or truncated:
            return obs, reward, done, truncated, info
        
        obs = self._observe_opponent()
        obs, reward, done, truncated, info = self._opponent_step(obs, info)
        return obs, reward, done, truncated, info