import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
from env.make_utils import get_offline_data

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class HumanoidNoBonusEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
        COM inertia (cinert), COM velocity (cvel), actuator forces (qfrc_actuator),
        and external forces (cfrc_ext) are removed from the observation, and alive_bonus = 0.
        Otherwise identical to Humanoid-v2 from
        https://github.com/openai/gym/blob/master/gym/envs/mujoco/humanoid.py
    """
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'humanoid.xml', 5)
        utils.EzPickle.__init__(self)
        self._max_episode_steps = 1000
        self.transition_function = get_offline_data

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               # data.cinert.flat,
                               # data.cvel.flat,
                               # data.qfrc_actuator.flat,
                               # data.cfrc_ext.flat
                               ])

    def step(self, a):
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        alive_bonus = 0.0 #5.0
        data = self.sim.data
        lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))

        done2 = self.check_done(np.array([self._get_obs()]))[0]
        if done != done2:
            print(done, done2)
            import pdb; pdb.set_trace()

        info = dict(
            reward_linvel=lin_vel_cost,
            reward_quadctrl=-quad_ctrl_cost,
            reward_alive=alive_bonus,
            reward_impact=-quad_impact_cost,
            violation=done
        )
        return self._get_obs(), reward, done, info

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20

    def check_done(self, states):
        return self.check_violation(states)

    def check_violation(self, states):
        heights = states[:,0]
        return (heights < 1.0) | (heights > 2.0)
