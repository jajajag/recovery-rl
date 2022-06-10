import gym
import mujoco_py
import numpy as np
from gym.envs.registration import register

ENV_ID = {
    'ant_no_bonus': 'AntNoBonus-v0',
    'cheetah_no_flip': 'CheetahNoFlipEnv-v0',
    'hopper_no_bonus': 'HopperNoBonusEnv-v0',
    'humanoid_no_bonus': 'HumanoidNoBonusEnv-v0',
}

ENV_CLASS = {
    'ant_no_bonus': 'AntNoBonusEnv',
    'cheetah_no_flip': 'CheetahNoFlipEnv',
    'hopper_no_bonus': 'HopperNoBonusEnv',
    'humanoid_no_bonus': 'HumanoidNoBonusEnv',
}


def register_env(env_name):
    env_id = ENV_ID[env_name]
    env_class = ENV_CLASS[env_name]
    register(id=env_id, entry_point='env.' + env_name + ":" + env_class)


def make_env(env_name):
    env_id = ENV_ID[env_name]
    return gym.make(env_id)

def get_offline_data(env, num_transitions, task_demos=False,
        save_rollouts=False):
    #env = AntNoBonusEnv()
    transitions = []
    rollouts = []
    done = False
    for i in range(num_transitions // env._max_episode_steps):
        rollouts.append([])
        state = env.reset()
        for j in range(env._max_episode_steps):
            action = np.clip(np.random.randn(env.action_space.shape[0]), -1, 1)
            next_state, reward, done, info = env.step(action)
            constraint = info['violation']
            transitions.append(
                (state, action, constraint, next_state, not constraint))
            rollouts[-1].append(
                (state, action, constraint, next_state, not constraint))
            state = next_state
            if constraint:
                break

    if save_rollouts:
        return rollouts
    else:
        return transitions
