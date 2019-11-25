import os
import tensorflow as tf
import numpy as np
import gym
import copy


class MPC:
	def __init__(self, env, plan_horizon, model, popsize, num_elites, max_iters,
				 num_particles=6,
				 use_gt_dynamics=True,
				 use_mpc=True,
				 use_random_optimizer=False):
		"""

		:param env:
		:param plan_horizon:
		:param model: The learned dynamics model to use, which can be None if use_gt_dynamics is True
		:param popsize: Population size
		:param num_elites: CEM parameter
		:param max_iters: CEM parameter
		:param num_particles: Number of trajectories for TS1
		:param use_gt_dynamics: Whether to use the ground truth dynamics from the environment
		:param use_mpc: Whether to use only the first action of a planned trajectory
		:param use_random_optimizer: Whether to use CEM or take random actions
		"""
		self.env = env
		self.use_gt_dynamics, self.use_mpc, self.use_random_optimizer = use_gt_dynamics, use_mpc, use_random_optimizer
		self.num_particles = num_particles
		self.plan_horizon = plan_horizon
		self.num_nets = None if model is None else model.num_nets

		self.state_dim, self.action_dim = 8, env.action_space.shape[0]
		self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low

		# Set up optimizer
		self.model = model

		if use_gt_dynamics:
			self.predict_next_state = self.predict_next_state_gt
			assert num_particles == 1
		else:
			self.predict_next_state = self.predict_next_state_model

		# Initialize your planner with the relevant arguments.
		# Write different optimizers for cem and random actions respectively
		self.M = popsize
		self.e = num_elites
		self.I = max_iters
		self.state_acts = []
		self.new_states = []
		self.reset()

	def obs_cost_fn(self, state):
		""" Cost function of the current state """
		# Weights for different terms
		W_PUSHER = 1
		W_GOAL = 2
		W_DIFF = 5

		pusher_x, pusher_y = state[0], state[1]
		box_x, box_y = state[2], state[3]
		goal_x, goal_y = self.goal[0], self.goal[1]

		pusher_box = np.array([box_x - pusher_x, box_y - pusher_y])
		box_goal = np.array([goal_x - box_x, goal_y - box_y])
		d_box = np.sqrt(np.dot(pusher_box, pusher_box))
		d_goal = np.sqrt(np.dot(box_goal, box_goal))
		diff_coord = np.abs(box_x / box_y - goal_x / goal_y)
		# the -0.4 is to adjust for the radius of the box and pusher
		return W_PUSHER * np.max(d_box - 0.4, 0) + W_GOAL * d_goal + W_DIFF * diff_coord

	def predict_next_state_model(self, states, actions):
		""" Given a list of state action pairs, use the learned model to predict the next state"""
		states = states[:, 0:8]
		state_act = np.concatenate((states, actions), axis=1)
		mean, log_var = self.model.predict(state_act)
		sampled_states = np.random.normal(mean, np.sqrt(np.exp(log_var)))
		return sampled_states

	def predict_next_state_gt(self, states, actions):
		""" Given a list of state action pairs, use the ground truth dynamics to predict the next state"""
		new_states = []
		for i in range(len(states)):
			new_states.append(self.env.get_nxt_state(states[i], actions[i]))
		return new_states

	def train(self, obs_trajs, acs_trajs, rews_trajs, epochs=5):
		"""
		Take the input obs, acs, rews and append to existing transitions the train model.
		Arguments:
		  obs_trajs: states
		  acs_trajs: actions
		  rews_trajs: rewards (NOTE: this may not be used)
		  epochs: number of epochs to train for
		"""
		num_traj = len(obs_trajs)
		for i in range(num_traj):
			obs = obs_trajs[i]
			acs = acs_trajs[i]
			for j in range(len(acs)):
				state = obs[j][0:8]
				action = acs[j]
				new_state = obs[j+1][0:8]
				state_act = np.concatenate((state, action), axis=0)
				self.state_acts.append(state_act)
				self.new_states.append(new_state)
		self.model.train(self.state_acts, self.new_states, epochs=epochs)

	def reset(self):
		self.mu = np.zeros([self.plan_horizon, self.action_dim], dtype=np.float32)
		self.actions = None

	def act(self, state, t):
		"""
		Use model predictive control to find the action give current state.

		Arguments:
		  state: current state
		  t: current timestep
		"""
		self.goal = state[8:10]
		if self.use_mpc:
			sigma = 0.5 * np.ones([self.plan_horizon, self.action_dim], dtype=np.float32)
			mu = self.CEM(state, self.mu, sigma)
			action = mu[0, :]
			self.mu = np.concatenate((self.mu[1:self.plan_horizon], np.zeros([1, self.action_dim])), axis=0)
		else:
			if self.actions is None:
				sigma = 0.5 * np.ones([self.plan_horizon, self.action_dim], dtype=np.float32)
				mu = np.zeros([self.plan_horizon, self.action_dim], dtype=np.float32)
				self.actions = self.CEM(state, mu, sigma)
				action = self.actions[t]
			elif len(self.actions) > t:
				action = self.actions[t]
			else:
				sigma = 0.5 * np.ones([self.plan_horizon, self.action_dim], dtype=np.float32)
				mu = np.zeros([self.plan_horizon, self.action_dim], dtype=np.float32)
				new_actions = self.CEM(state, mu, sigma)
				self.actions = np.concatenate((self.actions, new_actions), axis=0)
				action = self.actions[t]
		return action

	def TS1(self, init_states, action_seqs):
		cost_size = len(action_seqs)
		costs = np.zeros(cost_size)
		for p in range(self.num_particles):
			states = init_states.copy()
			for t in range(self.plan_horizon):
				actions = action_seqs[:, t]
				new_states = self.predict_next_state(states, actions)
				for m in range(cost_size):
					costs[m] += self.obs_cost_fn(new_states[m]) / self.num_particles
				states = new_states
		return costs

	def CEM(self, init_state, init_mu, init_sigma):
		if self.use_random_optimizer:
			mu = np.zeros([self.plan_horizon, self.action_dim])
			sigma = 0.5 * np.ones([self.plan_horizon, self.action_dim], dtype=np.float32)
			action_seqs = np.zeros([self.M*self.I, self.plan_horizon, self.action_dim])
			for i in range(self.I*self.M):
				action_seq = np.random.normal(mu, sigma)
				action_seqs[i] = action_seq
			states = np.asarray([init_state.copy()] * self.I * self.M)
			costs = self.TS1(states, action_seqs)
			order_idx = np.argsort(costs)
			best_idx = order_idx[0]
			best = action_seqs[best_idx]
			mu = best
		else:
			mu = init_mu.copy()
			sigma = init_sigma.copy()
			for i in range(self.I):
				action_seqs = np.zeros([self.M, self.plan_horizon, self.action_dim])
				for m in range(self.M):
					action_seq = np.random.normal(mu, sigma)
					action_seqs[m] = action_seq
				states = np.asarray([init_state.copy()] * self.M)
				costs = self.TS1(states, action_seqs)
				order_idx = np.argsort(costs)
				elites_idx = order_idx[:self.e]
				elites = action_seqs[elites_idx]
				mu = np.mean(elites, axis=0)
				sigma = np.std(elites, axis=0)

		return mu

