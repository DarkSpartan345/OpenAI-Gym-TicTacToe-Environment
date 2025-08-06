import gymnasium as gym
import gym_TicTacToe  # Asegúrate que esté registrado

env = gym.make("TicTacToe-v0", render_mode="human", small=-1, large=10)
obs, _ = env.reset()
env.render()
done = False
truncated = False
action = 0
while not (done or truncated):
    obs, reward, done, truncated, info = env.step(action)
    action +=1
    env.render()
