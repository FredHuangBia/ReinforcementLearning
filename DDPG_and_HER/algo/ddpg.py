import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from .ReplayBuffer import ReplayBuffer
from .ActorNetwork import ActorNetwork
from .CriticNetwork import CriticNetwork

BUFFER_SIZE = 1000000
BATCH_SIZE = 1024
GAMMA = 0.98                    # Discount for rewards.
TAU = 0.05                      # Target network update rate.
LEARNING_RATE_ACTOR = 0.0001
LEARNING_RATE_CRITIC = 0.001
EPSILON = 0.01
SIGMA = 0.05

class EpsilonNormalActionNoise(object):
	"""A class for adding noise to the actions for exploration."""

	def __init__(self, mu, sigma, epsilon):
		"""Initialize the class.
		Args:
			mu: (float) mean of the noise (probably 0).
			sigma: (float) std dev of the noise.
			epsilon: (float) probability in range [0, 1] with
			which to add noise.
		"""
		self.mu = mu
		self.sigma = sigma
		self.epsilon = epsilon

	def __call__(self, action):
		"""With probability epsilon, adds random noise to the action.
		Args:
			action: a batched tensor storing the action.
		Returns:
			noisy_action: a batched tensor storing the action.
		"""
		if np.random.uniform() > self.epsilon:
			return action + np.random.normal(self.mu, self.sigma)
		else:
			return np.random.uniform(-1.0, 1.0, size=action.shape)


class DDPG(object):
	"""A class for running the DDPG algorithm."""

	def __init__(self, env, outfile_name):
		"""Initialize the DDPG object.
		Args:
			env: an instance of gym.Env on which we aim to learn a policy.
			outfile_name: (str) name of the output filename.
		"""
		action_dim = len(env.action_space.low)
		state_dim = len(env.observation_space.low)
		np.random.seed(1337)
		self.env = env

		self.sess = tf.Session()
		tf.keras.backend.set_session(self.sess)
		self.actor = ActorNetwork(self.sess, state_dim, action_dim, BATCH_SIZE, TAU, LEARNING_RATE_ACTOR)
		self.critic = CriticNetwork(self.sess, state_dim, action_dim, BATCH_SIZE, TAU, LEARNING_RATE_CRITIC)
		self.noise = EpsilonNormalActionNoise(0, SIGMA, EPSILON)
		self.outfile = outfile_name
		self.replayBuffer=ReplayBuffer(BUFFER_SIZE)

	def evaluate(self, num_episodes):
		"""Evaluate the policy. Noise is not added during evaluation.
		Args:
			num_episodes: (int) number of evaluation episodes.
		Returns:
			success_rate: (float) fraction of episodes that were successful.
			average_return: (float) Average cumulative return.
		"""
		test_rewards = []
		success_vec = []
		plt.figure(figsize=(12, 12))
		for i in range(num_episodes):
			s_vec = []
			state = self.env.reset()
			s_t = np.array(state)
			total_reward = 0.0
			done = False
			step = 0
			success = False
			while not done:
				s_vec.append(s_t)
				a_t = self.actor.model.predict(s_t[None])[0]
				new_s, r_t, done, info = self.env.step(a_t)
				if done and "goal" in info["done"]:
					success = True
				new_s = np.array(new_s)
				total_reward += r_t
				s_t = new_s
				step += 1
			success_vec.append(success)
			test_rewards.append(total_reward)
			# if i < 9:
			#     plt.subplot(3, 3, i+1)
			#     s_vec = np.array(s_vec)
			#     pusher_vec = s_vec[:, :2]
			#     puck_vec = s_vec[:, 2:4]
			#     goal_vec = s_vec[:, 4:]
			#     plt.plot(pusher_vec[:, 0], pusher_vec[:, 1], '-o', label='pusher')
			#     plt.plot(puck_vec[:, 0], puck_vec[:, 1], '-o', label='puck')
			#     plt.plot(goal_vec[:, 0], goal_vec[:, 1], '*', label='goal', markersize=10)
			#     plt.plot([0, 5, 5, 0, 0], [0, 0, 5, 5, 0], 'k-', linewidth=3)
			#     plt.fill_between([-1, 6], [-1, -1], [6, 6], alpha=0.1,
			#                      color='g' if success else 'r')
			#     plt.xlim([-1, 6])
			#     plt.ylim([-1, 6])
			#     if i == 0:
			#         plt.legend(loc='lower left', fontsize=28, ncol=3, bbox_to_anchor=(0.1, 1.0))

		return np.mean(success_vec), np.mean(test_rewards)

	def train(self, num_episodes, hindsight=False):
		"""Runs the DDPG algorithm.
		Args:
			num_episodes: (int) Number of training episodes.
			hindsight: (bool) Whether to use HER.
		"""
		self.actor.target_model.set_weights(self.actor.model.get_weights())
		self.critic.target_model.set_weights(self.critic.model.get_weights())

		for i in range(num_episodes):
			state = self.env.reset()
			s_t = np.array(state)
			total_reward = 0.0
			done = False
			step = 0
			loss = 0
			store_states = []
			store_actions = []
			while not done:
				# Collect one episode of experience, saving the states and actions
				# to store_states and store_actions, respectively.
				store_states.append(s_t.copy())
				a_t = self.actor.model.predict(s_t[None])[0]
				a_t = self.noise(a_t)
				store_actions.append(a_t.copy())
				new_s, r_t, done, info = self.env.step(a_t)

				new_s = np.array(new_s)
				self.replayBuffer.add(s_t, a_t, r_t, new_s, done)

				# buff sample
				samples = self.replayBuffer.get_batch(BATCH_SIZE)
				states=np.array([sample[0] for sample in samples])
				actions=np.array([sample[1] for sample in samples])
				rewards=np.array([sample[2] for sample in samples])
				new_states=np.array([sample[3] for sample in samples])
				dones=np.array([sample[4] for sample in samples])
				# update critic and actor
				target_value = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)])
				target_value=np.reshape(target_value,target_value.shape[0])
				y=rewards.astype('float64')
				y+=GAMMA*target_value*(1-dones.astype(int))

				loss+= self.critic.model.train_on_batch([states,actions],y)
				critic_grads=self.critic.gradients(states, self.actor.model.predict(states))
				self.actor.train(states,critic_grads)

				# update target
				self.critic.update_target()
				self.actor.update_target()

				# update others
				total_reward += r_t
				s_t = new_s
				step += 1

			if hindsight:
				# For HER, we also want to save the final next_state.
				store_states.append(new_s)
				self.add_hindsight_replay_experience(store_states,
													 store_actions)
				print(len(store_states))
				print(len(store_actions))
			del store_states, store_actions
			store_states, store_actions = [], []

			# Logging
			# print("Episode %d: Total reward = %d" % (i, total_reward))
			# print("\tTD loss = %.2f" % (loss / step,))
			# print("\tSteps = %d; Info = %s" % (step, info['done']))
			if i % 100 == 0:
				successes, mean_rewards = self.evaluate(100)
				print('Evaluation %d: success = %.2f; return = %.2f' % (i, successes, mean_rewards))
				with open(self.outfile, "a") as f:
					f.write("%.2f, %.2f,\n" % (successes, mean_rewards))


	def add_hindsight_replay_experience(self, states, actions):
		"""Relabels a trajectory using HER.
		Args:
			states: a list of states.
			actions: a list of states.
		"""
		new_goal_states, new_goal_rewards= self.env.apply_hindsight(states)
		for i in range(len(actions)):
			self.replayBuffer.add(new_goal_states[i], actions[i], new_goal_rewards[i], new_goal_states[i+1], False)