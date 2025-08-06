from gymnasium.envs.registration import register

register(
    id="TicTacToe-v0",
    entry_point="gym_TicTacToe.envs.tictactoe_env:TicTacToeEnv",  # Usa el nombre real del archivo sin `.py`
    max_episode_steps=9,
)