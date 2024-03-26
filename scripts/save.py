import isaacgym

assert isaacgym
import torch
import numpy as np

import glob
import pickle as pkl

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm

import random

torch.cuda.set_per_process_memory_fraction(fraction=0.4, device=0)

def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_env(label, headless=False):
    dirs = glob.glob(f"../runs/{label}/*")
    logdir = sorted(dirs)[0]

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 5000
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

    policy = load_policy(logdir)

    return env, policy


def save(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    label = "gait-conditioned-agility/pretrain-v0/train"

    env, policy = load_env(label, headless=headless)

    num_envs = env.num_envs
    num_eval_steps = 250
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 1.5, 0.0
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["pacing"]) # change gait here
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    dataset = {}
    state_ = []
    action_ = []
    reward_ = []
    timeout_ = None
    episodes = 10

    for episode in range(episodes):
        x_vel_cmd = torch.tensor([random.uniform(-1.5, 1.5) for _ in range(num_envs)])
        y_vel_cmd = torch.tensor([random.uniform(-1.5, 1.5) for _ in range(num_envs)])
        yaw_vel_cmd = torch.tensor([random.uniform(-0.5, 0.5) for _ in range(num_envs)])

        gait_index = [random.choice(range(len(gaits))) for _ in range(num_envs)]
        random_gaits = [list(gaits.values())[idx] for idx in gait_index]
        gait = torch.tensor(random_gaits)

        recorded_obs = []
        recorded_acts = []
        recorded_rewards = []

        done_envs = []

        obs = env.reset()
        for i in tqdm(range(num_eval_steps)):
            with torch.no_grad():
                actions = policy(obs)

                env.commands[:, 0] = x_vel_cmd
                env.commands[:, 1] = y_vel_cmd
                env.commands[:, 2] = yaw_vel_cmd
                env.commands[:, 3] = body_height_cmd
                env.commands[:, 4] = step_frequency_cmd
                env.commands[:, 5:8] = gait
                env.commands[:, 8] = 0.5
                env.commands[:, 9] = footswing_height_cmd
                env.commands[:, 10] = pitch_cmd
                env.commands[:, 11] = roll_cmd
                env.commands[:, 12] = stance_width_cmd

                obs, rew, done, info = env.step(actions)

            # if True in done: print("here")

            done_indices = [index for index, value in enumerate(done) if value]
            for ind in done_indices:
                if ind not in done_envs:
                    done_envs.append(ind)

            this_obs = torch.cat([ env.root_states[:,2:3].detach().cpu(), env.root_states[:,3:7].detach().cpu(),
                                        env.root_states[:,7:10].detach().cpu(), env.base_ang_vel[:,:].detach().cpu(),
                                        env.projected_gravity.detach().cpu(), env.clock_inputs.detach().cpu(),
                                        env.dof_pos[:,:12].detach().cpu(), env.dof_vel[:,:12].detach().cpu()], dim=-1) # (500,42)
            this_reward = torch.stack([torch.tensor(gait_index), x_vel_cmd, y_vel_cmd, yaw_vel_cmd], dim=-1) # (500,4) size this_reward

            recorded_obs.append(this_obs)
            recorded_acts.append(actions)
            recorded_rewards.append(this_reward)

            del this_obs, this_reward

        recorded_obs = torch.stack(recorded_obs, dim=1) # 250*(500,42) -> (500,250,42)
        recorded_acts = torch.stack(recorded_acts, dim=1)
        recorded_rewards = torch.stack(recorded_rewards, dim=1)

        sliced_obs = []
        sliced_acts = []
        sliced_rewards = []
        if len(done_envs) != 0:
            done_envs.sort()
            done_envs = [0] + done_envs + [env.num_envs]
            slice_indices = list(zip(done_envs[:-1], done_envs[1:]))
            for i, (a,b) in enumerate(slice_indices):
                if i != 0:
                    a += 1
                    slice_indices[i] = (a,b)
            for a,b in slice_indices:
                sliced_obs.append(recorded_obs[a:b,:,:])
                sliced_acts.append(recorded_acts[a:b,:,:])
                sliced_rewards.append(recorded_rewards[a:b,:,:])

        if len(sliced_obs) != 0:
            recorded_obs = torch.cat(sliced_obs, axis=0)
            recorded_acts = torch.cat(sliced_acts, axis=0)
            recorded_rewards = torch.cat(sliced_rewards, axis=0)

        del sliced_obs, sliced_acts, sliced_rewards

        recorded_obs = recorded_obs.view(-1, 42)
        recorded_acts = recorded_acts.view(-1, 12)
        recorded_rewards = recorded_rewards.view(-1, 4)

        state_.append(recorded_obs)
        action_.append(recorded_acts)
        reward_.append(recorded_rewards)

        del recorded_obs, recorded_acts, recorded_rewards

    state_ = torch.cat(state_, dim=0)
    action_ = torch.cat(action_, dim=0)
    reward_ = torch.cat(reward_, dim=0)

    state_ = state_.view(-1, 42)
    action_ = action_.view(-1, 12)
    reward_ = reward_.view(-1, 4)

    state_ = state_.detach().cpu().numpy()
    action_ = action_.detach().cpu().numpy()
    reward_ = reward_.detach().cpu().numpy()

    dataset['actions'] = action_
    dataset['observations'] = state_
    # dataset['rewards'] = np.array([0.9 for i in range(state_.shape[0])])
    # dataset['rewards'] = np.array([(gait_emb + dir_emb) for i in range(state_.shape[0])])
    dataset['rewards'] = reward_
    timeouts = [False for i in range(249)] + [True]
    dataset['terminals'] = np.array([False for i in range(state_.shape[0])])
    true_eps = int(state_.shape[0] / 250)
    dataset['timeouts'] = np.array(timeouts * true_eps)

    del state_, action_, reward_

    file_path = os.path.expanduser('~/Desktop/data.pkl')
    with open(file_path, 'wb') as f:
        pkl.dump(dataset, f)

    for key in dataset:
        print(f"Key: {key}, Shape: {dataset[key].shape}")



if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    save(headless=False)
