# Chess `Gym`

[![PyPI download month](https://img.shields.io/pypi/dm/chess-gym.svg)](https://pypi.python.org/pypi/chess-gym/)
[![PyPI - Status](https://img.shields.io/pypi/status/chess-gym)](https://pypi.python.org/pypi/chess-gym/)
[![PyPI](https://img.shields.io/pypi/v/chess-gym)](https://pypi.python.org/pypi/chess-gym/)
![GitHub](https://img.shields.io/github/license/Ryan-Rudes/chess-gym)


Gym Chess is an environment for reinforcement learning with the OpenAI gym module.

<a href="https://imgbb.com/"><img src="https://i.ibb.co/Fw4fhzK/Screen-Shot-2020-10-27-at-2-30-21-PM.png" alt="Screen-Shot-2020-10-27-at-2-30-21-PM" border="0"></a>

## Installation

1. Install [Farama Gymnasium](https://gymnasium.farama.org/) ([GitHub](https://github.com/Farama-Foundation/Gymnasium)) and its dependencies with poetry. \

```bash
pip install gymnasium
```

2. Download and install `chess_gym`: \

```bash
git clone https://github.com/RobinU434/chess_gym
cd chess_gym 
poetry install  
```

## Environments
<a href="https://ibb.co/dgLW9rH"><img src="images/envs.png" border="0"></a>

## Example
You can use the standard `Chess-v0` environment as so:
```python
from chess_gym import ChessEnv

env = ChessEnv()
env.reset()

terminal = False

while not (terminal or truncated):
    action = env.action_space.sample()
    observation, reward, terminal, truncated, info = env.step(action)
    env.render()
env.close()
```

There is also an environment for the Chess960 variant. You can add it modified the existing class as `ChessEnv(chess960=True)`

Please note that the environment only supports `rgb_array` rendering. For `human` rendering please turn to the [HumanRenderingWrapper](https://gymnasium.farama.org/main/_modules/gymnasium/wrappers/human_rendering/). 

## Further Info
This environment will return 0 reward until the game has reached a terminal state. In the case of a draw, it will still return 0 reward. Otherwise, the reward will be either 1 or -1, depending upon the winning player.
```python
observation, reward, terminal, truncated, info = env.step(action)
```
Here, `info` will be a dictionary containing the following information pertaining to the board configuration and game state:
* [`next_turn`](https://python-chess.readthedocs.io/en/latest/core.html#chess.Board.turn): The side to move (`chess.WHITE` or `chess.BLACK`).
* [`castling_rights`](https://python-chess.readthedocs.io/en/latest/core.html#chess.Board.castling_rights): Bitmask of the rooks with castling rights.
* [`fullmove_number`](https://python-chess.readthedocs.io/en/latest/core.html#chess.Board.fullmove_number): Counts move pairs. Starts at 1 and is incremented after every move of the black side.
* [`halfmove_clock`](https://python-chess.readthedocs.io/en/latest/core.html#chess.Board.halfmove_clock): The number of half-moves since the last capture or pawn move.
* [`promoted`](https://python-chess.readthedocs.io/en/latest/core.html#chess.Board.promoted): A bitmask of pieces that have been promoted.
* [`ep_square`](https://python-chess.readthedocs.io/en/latest/core.html#chess.Board.ep_square): The potential en passant square on the third or sixth rank or `None`.
