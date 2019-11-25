"""LQR, iLQR and MPC."""

import scipy
from scipy import linalg
import gym
import numpy as np
from deeprl_hw6.arm_env import *
from time import sleep
import matplotlib.pyplot as plt

def simulate_dynamics(env, x, u, dt=1e-5):
    """Step simulator to see how state changes.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    xdot: np.array
      This is the **CHANGE** in x. i.e. (x[1] - x[0]) / dt
      If you return x you will need to solve a different equation in
      your LQR controller.
    """
    env.state = x.copy()
    x_new, _, _, _ = env.step(u, dt)
    return (x_new - x) / dt


def approximate_A(env, x, u, delta=1e-5, dt=1e-5):
    """Approximate A matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. You will need to perturb this.
    u: np.array
      The command to test.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    A: np.array
      The A matrix for the dynamics at state x and command u.
    """
    A = np.zeros((x.shape[0], x.shape[0]))
    up = x.copy()
    down = x.copy()
    for r in range(x.shape[0]):
        up[r] += delta
        down[r] -= delta
        diff = simulate_dynamics(env, up, u, dt) - simulate_dynamics(env, down, u, dt)
        up[r] -= delta
        down[r] += delta
        A[:, r] = diff / (2*delta)
    return A


def approximate_B(env, x, u, delta=1e-5, dt=1e-5):
    """Approximate B matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test.
    u: np.array
      The command to test. You will ned to perturb this.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    B: np.array
      The B matrix for the dynamics at state x and command u.
    """
    B = np.zeros((x.shape[0], u.shape[0]))
    up = u.copy()
    down = u.copy()
    for r in range(u.shape[0]):
        up[r] += delta
        down[r] -= delta
        v1 = simulate_dynamics(env, x, up, dt)
        v2 = simulate_dynamics(env, x, down, dt)
        diff = v1 - v2
        up[r] -= delta
        down[r] += delta
        B[:, r] = diff / (2*delta)
    return B


def calc_lqr_input(env, sim_env):
    """Calculate the optimal control input for the given state.

    If you are following the API and simulate dynamics is returning
    xdot, then you should use the scipy.linalg.solve_continuous_are
    function to solve the Ricatti equations.

    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.

    Returns
    -------
    u: np.array
      The command to execute at this point.
    """
    x = env.state
    Q = env.Q
    R = env.R
    A = approximate_A(sim_env, x, np.zeros(2,))
    B = approximate_B(sim_env, x,  np.zeros(2,))
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    px = np.matmul(P, x-env.goal)
    btpx = np.matmul(B.T, px)
    u = -np.matmul(scipy.linalg.inv(R), btpx)
    return u


def run():
    env = gym.make("TwoLinkArm-v0")
    sim_env = gym.make("TwoLinkArm-v0")
    max_step = 200
    total_reward = 0
    finish = False
    u1s = []
    u2s = []
    q1s = [env.state[0]]
    q2s = [env.state[1]]
    q1_dots = [env.state[2]]
    q2_dots = [env.state[3]]

    for i in range(max_step):
        sim_env.state = env.state.copy()
        u = calc_lqr_input(env, sim_env)
        new_state, reward, done, _ = env.step(u)
        total_reward += reward
        if done and not finish:
            print("Finish after %d steps with reward %f"%(i+1, total_reward))
            finish = True
        env.render()
        u1s.append(u[0])
        u2s.append(u[1])
        q1s.append(new_state[0])
        q2s.append(new_state[1])
        q1_dots.append(new_state[2])
        q2_dots.append(new_state[3])

    plt.figure()
    u1_plt, = plt.plot(u1s, label='u_1')
    u2_plt, = plt.plot(u2s, label='u_2')
    plt.title("u - step")
    plt.legend([u1_plt, u2_plt], ["u_1", "u_2"])
    plt.savefig('us.png')

    plt.figure()
    q1_plt, = plt.plot(q1s, label='q_1')
    q2_plt, = plt.plot(q2s, label='q_2')
    plt.title("q - step")
    plt.legend([q1_plt, q2_plt], ["q_1", "q_2"])
    plt.savefig('qs.png')

    plt.figure()
    q_dot1_plt, = plt.plot(q1_dots, label='q_dot_1')
    q_dot2_plt, = plt.plot(q2_dots, label='q_dot_2')
    plt.title("q_dot - step")
    plt.legend([q_dot1_plt, q_dot2_plt], ["q_dot_1", "q_dot_2"])
    plt.savefig('q_dots.png')

if __name__ == "__main__":
    run()