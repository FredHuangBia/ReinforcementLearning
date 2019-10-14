import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
import gym
from keras.optimizers import Adam

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr):
        self.model = model
        #       your variables, or alternately compile your model here.
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr))

    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using REINFORCE.
        #       method generate_episode() to generate training data.
        states, actions, rewards, action_array = self.generate_episode(env)
        episode_length = states.shape[0]
        action_size = actions.shape[1]
        
        discount = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, episode_length)):
            running_add = running_add * gamma + rewards[t]
            discount[t] = running_add
        discount -= np.mean(discount)
        discount /= np.std(discount)
        
        advantages = np.zeros((episode_length, action_size))
        for i in range(episode_length):
            advantages[i][action_array[i]] = discount[i]

        self.model.fit(states, advantages, epochs=1, verbose=0)

    def action_to_one_hot(self, env, action):
        action_vec = np.zeros(env.action_space.n)
        action_vec[action] = 1
        return action_vec

    def stochastic_policy(self, q_values):
        actions = [i for i in range(4)]
        action = np.random.choice(actions, 1, p=q_values)
        return action[0]

    def test(self, env, episodes=100, render=False):
        sum_rewards = []
        for i in range(episodes):
            sum_reward = 0
            state = env.reset()
            done = False
            while not done:
                if render and i % 20 == 0:
                    env.render()
                state = state.reshape(1, state.shape[0])
                q_values = self.model.predict(state)[0]
                action = self.stochastic_policy(q_values)
                state, reward, done, _ = env.step(action)
                sum_reward += reward
            sum_rewards.append(sum_reward)
        avg_reward = np.mean(sum_rewards)
        std_reward = np.std(sum_rewards)
        return avg_reward, std_reward

    def generate_episode(self, env, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        states = []
        actions = []
        rewards = []
        action_array = []
        done = False

        state = env.reset()
        while not done:
            state = state.reshape(1, state.shape[0])
            states.extend(state)
            q_values = self.model.predict(state)[0]
            action = self.stochastic_policy(q_values)
            action_array.append(action)
            state, reward, done, _ = env.step(action)
            action = self.action_to_one_hot(env, action)
            actions.append(action)
            rewards.append(reward)  
        return np.array(states), np.array(actions), np.array(rewards), action_array


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=20000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    num_episodes = args.num_episodes
    lr = args.lr
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
    state_size = env.observation_space.shape[0]

    # Create the model.
    init = keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None)
    model = Sequential()
    model.add(Dense(64, input_shape=(state_size, ), bias_initializer='zeros', activation='relu', kernel_initializer=init))
    model.add(Dense(64, input_shape=(64, ), bias_initializer='zeros', activation='relu', kernel_initializer=init))
    model.add(Dense(4, input_shape=(64, ), bias_initializer='zeros', activation='softmax', kernel_initializer=init))

    # Train the model using REINFORCE and plot the learning curve.
    rein = Reinforce(model, lr)
    trained_epi = []
    avg_rewards = []
    std_rewards = []
    for i in range(num_episodes):
        rein.train(env, gamma=0.99)
        if i % 100 == 0:
            avg_reward, std_reward = rein.test(env, render=False)
            trained_epi.append(i)
            avg_rewards.append(avg_reward)
            std_rewards.append(std_reward)
            print("Average reward and std after %d episodes: %f  %f" % (i, avg_reward, std_reward))

    f1 = plt.figure(1)
    plt.errorbar(trained_epi, avg_rewards, std_rewards)
    plt.title("Avg Test Reward - Training Episodes")
    f1.savefig("reward_REINFORCE.png")

if __name__ == '__main__':
    main(sys.argv)
