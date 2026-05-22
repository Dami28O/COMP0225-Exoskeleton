""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import logging

import mujoco
import numpy as np

from myosuite.envs import env_base
from myosuite.envs.myo.fatigue import CumulativeFatigue

from myosuite.envs import env_base
from myosuite.envs.myo.fatigue import CumulativeFatigue
from myosuite.utils import gym
import collections
class BaseTSAV0(env_base.MujocoEnv):

    MYO_CREDIT = """
    MyoSuite: A contact-rich simulation suite for musculoskeletal motor control
        Vittorio Caggiano, Huawei Wang, Guillaume Durandau, Massimo Sartori, Vikash Kumar
        L4DC-2019 | https://sites.google.com/view/myosuite
    """
    
    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)

        self._setup(**kwargs)

    def _setup(
        self,
        obs_keys: list = ['qpos', 'qvel', 'tip_pos', 'reach_err'],
        weighted_reward_keys: dict = {
        "reach": 1.0,
        "bonus": 4.0,
        "penalty": 50,
        "act_reg": 1
    },
        sites: list = None,
        frame_skip=10,
        muscle_condition="",
        fatigue_reset_vec=None,
        fatigue_reset_random=False,
        **kwargs,
    ):
        self.far_th = .35
        if self.sim.model.na > 0 and "act" not in obs_keys:
            obs_keys = (
                obs_keys.copy()
            )  # copy before editing incase other envs are using the defaults
            obs_keys.append("act")
        # ids
        self.tip_sids = []
        self.target_sids = []
        if sites:
            for site in sites:
                self.tip_sids.append(self.sim.model.site_name2id(site))
                self.target_sids.append(self.sim.model.site_name2id(site + "_target"))

        self.muscle_condition = muscle_condition
        self.fatigue_reset_vec = fatigue_reset_vec
        self.fatigue_reset_random = fatigue_reset_random
        self.frame_skip = frame_skip
        self.initializeConditions()
        super()._setup(
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            frame_skip=frame_skip,
            **kwargs,
        )
        self.viewer_setup(azimuth=90, distance=1.5, render_actuator=True)
        self.init_qpos[:] = self.sim.model.key_qpos[0]
        self.init_qvel[:] = self.sim.model.key_qvel[0]
        # find geometries with ID == 1 which indicates the skins
        geom_1_indices = np.where(self.sim.model.geom_group == 1)
        # Change the alpha value to make it transparent
        self.sim.model.geom_rgba[geom_1_indices, 3] = 0


        
        

    def initializeConditions(self):
        # for muscle weakness we assume that a weaker muscle has a
        # reduced maximum force
        if self.muscle_condition == "sarcopenia":
            for mus_idx in range(self.sim.model.actuator_gainprm.shape[0]):
                self.sim.model.actuator_gainprm[mus_idx, 2] = (
                    0.5 * self.sim.model.actuator_gainprm[mus_idx, 2].copy()
                )

        # for muscle fatigue we used the 3CC-r model
        elif self.muscle_condition == "fatigue":
            self.muscle_fatigue = CumulativeFatigue(
                self.sim.model, self.frame_skip, seed=self.get_input_seed()
            )

        # Tendon transfer to redirect EIP --> EPL
        # https://www.assh.org/handcare/condition/tendon-transfer-surgery
        elif self.muscle_condition == "reafferentation":
            self.EPLpos = self.sim.model.actuator_name2id("EPL")
            self.EIPpos = self.sim.model.actuator_name2id("EIP")

    # step the simulation forward
    def step(self, a, **kwargs):
        muscle_a = a.copy()
        muscle_act_ind = self.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        # Explicitely project normalized space (-1,1) to actuator space (0,1) if muscles
        if self.sim.model.na and self.normalize_act:
            # find muscle actuators
            muscle_a[muscle_act_ind] = 1.0 / (
                1.0 + np.exp(-5.0 * (muscle_a[muscle_act_ind] - 0.5))
            )
            # TODO: actuator space may not always be (0,1) for muscle or (-1, 1) for others
            isNormalized = (
                False  # refuse internal reprojection as we explicitly did it here
            )
        else:
            isNormalized = self.normalize_act  # accept requested reprojection

        # implement abnormalities
        if self.muscle_condition == "fatigue":
            # import ipdb; ipdb.set_trace()
            muscle_a[muscle_act_ind], _, _ = self.muscle_fatigue.compute_act(
                muscle_a[muscle_act_ind]
            )
        elif self.muscle_condition == "reafferentation":
            # redirect EIP --> EPL
            muscle_a[self.EPLpos] = muscle_a[self.EIPpos].copy()
            # Set EIP to 0
            muscle_a[self.EIPpos] = 0
        # step forward
        self.last_ctrl = self.robot.step(
            ctrl_desired=muscle_a,
            ctrl_normalized=isNormalized,
            step_duration=self.dt,
            realTimeSim=self.mujoco_render_frames,
            render_cbk=self.mj_render if self.mujoco_render_frames else None,
        )

        return self.forward(**kwargs)

    def reset(self, fatigue_reset=True, *args, **kwargs):
        if fatigue_reset:
            if self.muscle_condition == "fatigue":
                self.muscle_fatigue.reset(
                    fatigue_reset_vec=self.fatigue_reset_vec,
                    fatigue_reset_random=self.fatigue_reset_random,
                )
            else:
                pass

        return super().reset(*args, **kwargs)

    def set_fatigue_reset_random(self, fatigue_reset_random):  #
        if self.muscle_condition != "fatigue":
            logging.warning("This has no effect, as no fatigue model is provided.")
        self.fatigue_reset_random = fatigue_reset_random
    
    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['qpos'] = sim.data.qpos[:].copy()
        obs_dict['qvel'] = sim.data.qvel[:].copy()*self.dt
        if sim.model.na>0:
            
            obs_dict['act'] = sim.data.act[:].copy()

        # reach error
        obs_dict['tip_pos'] = np.array([])
        obs_dict['target_pos'] = np.array([])
        for isite in range(len(self.tip_sids)):
            obs_dict['tip_pos'] = np.append(obs_dict['tip_pos'], sim.data.site_xpos[self.tip_sids[isite]].copy())
            obs_dict['target_pos'] = np.append(obs_dict['target_pos'], sim.data.site_xpos[self.target_sids[isite]].copy())
        obs_dict['reach_err'] = np.array(obs_dict['target_pos'])-np.array(obs_dict['tip_pos'])
        return obs_dict


    def get_reward_dict(self, obs_dict):
        reach_dist = np.linalg.norm(obs_dict['reach_err'], axis=-1)[0][0]
        vel_dist = np.linalg.norm(obs_dict['qvel'], axis=-1)[0][0]
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        far_th = self.far_th*len(self.tip_sids) if np.squeeze(obs_dict['time'])>2*self.dt else np.inf
        
        # near_th = len(self.tip_sids)*.0125
        near_th = len(self.tip_sids)*.050
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('reach',   10.-1.*reach_dist -10.*vel_dist),
            ('bonus',   1.*(reach_dist<2*near_th) + 1.*(reach_dist<near_th)),
            ('act_reg', -100.*act_mag),
            ('penalty', -1.*(reach_dist>far_th)),
            # Must keys
            ('sparse',  -1.*reach_dist),
            ('solved',  reach_dist<near_th),
            ('done',    reach_dist > far_th),
        ))
        
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict
