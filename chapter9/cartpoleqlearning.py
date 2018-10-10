import gym
import numpy as np
import math

env = gym.make('CartPole-v0')

buckets = (1, 1, 6, 12,)
n_episodes = 100
goal_duration = 195
alpha = 0.1
min_epsilon = 0.1
gamma = 1.0
ada_divisor = 25
Q = np.zeros(buckets + (env.action_space.n,))


def discretize(obs):
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
    ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
    return tuple(new_obs)


def choose_action(state, epsilon):
    return env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(Q[state])


def update_q(state_old, action, reward, state_new, alpha):
    Q[state_old][action] += alpha * (reward + gamma * np.max(Q[state_new]) - Q[state_old][action])


def get_epsilon(t):
    return max(min_epsilon, min(1, 1.0 - math.log10((t + 1) / ada_divisor)))


def run_episode():
    observation = env.reset()
    current_state = discretize(observation)
    epsilon = get_epsilon(episode)

    done = False

    while not done:
        env.render()
        action = choose_action(current_state, epsilon)
        obs, reward, done, _ = env.step(action)
        new_state = discretize(obs)
        update_q(current_state, action, reward, new_state, alpha)
        current_state = new_state


if __name__ == '__main__':
    for episode in range(n_episodes):
        run_episode()
