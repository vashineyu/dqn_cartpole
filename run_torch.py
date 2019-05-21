# run_torch.py

import argparse
import gym
import os
import numpy as np
from pyvirtualdisplay import Display
from collections import deque
import torch

from cartpole.config import get_cfg_defaults
from cartpole.utils import screen_to_state
from cartpole.agent_torch import DqnAgent

parser = argparse.ArgumentParser()
parser.add_argument('--message', type=str, help="Must add message for recording this experiment info")
parser.add_argument(
    "--config-file",
    default=None,
    metavar="FILE",
    help="path to config file",
    type=str,
    )
parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
args = parser.parse_args()

cfg = get_cfg_defaults()
if args.config_file is not None:
    cfg.merge_from_file(args.config_file)
if args.opts is not None:
    cfg.merge_from_list(args.opts)
cfg.freeze()
print(cfg)

devices = ",".join(str(i) for i in cfg.SYSTEM.DEVICES)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = devices

torch_devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
If you want to access the behind-the.scenes dynamics of a specific environment, 
then you use the unwrapped property.
"""
display = Display(visible=0, size=cfg.SYSTEM.VIRTUAL_SCREEN)
display.start()
env = gym.make("CartPole-v0").unwrapped

agent = DqnAgent(state_shape=cfg.MODEL.INPUT_SIZE,
                 action_space=env.action_space.n, 
                 device=torch_devices,
                 gamma=cfg.AGENT.GAMMA, 
                 batch_size=cfg.AGENT.BATCH_SIZE, 
                 memory_size=int(cfg.AGENT.NUM_MEMORY_CAPACITY)
                 )

scores = []
scores_window = deque(maxlen=100)
eps = cfg.AGENT.EPS_START
for i_episode in range(cfg.AGENT.NUM_EPISODE):
    env.reset()
    last_screen = screen_to_state(env, target_size=cfg.MODEL.INPUT_SIZE[:2])
    current_screen = screen_to_state(env, target_size=cfg.MODEL.INPUT_SIZE[:2])
    state = current_screen - last_screen
    total_rewards = 0
    for t_counter in range(cfg.AGENT.MAX_T):
        action = agent.act(state, eps)
        
        vector_state, reward, is_done, _ = env.step(action)
        total_rewards += reward
        
        last_screen = current_screen
        current_screen = screen_to_state(env, target_size=cfg.MODEL.INPUT_SIZE[:2])
        if not is_done:
            next_state = current_screen - last_screen
        else:
            next_state = current_screen - current_screen
        state = next_state
        
        # Train the model
        agent.step(state, action, next_state, reward, is_done*1)
        #print("\rEpisode {}, Accumulated Reward: {:.3f}, remain time: {}".format(i_episode, total_rewards, t_counter), end='')
        
        if is_done:
            break
    scores.append(total_rewards)
    scores_window.append(total_rewards)
    print("\rEpisode {}, Accumulated Reward: {:.1f}, passed n mean reward: {:.1f}".format(i_episode, 
                                                                                          total_rewards, 
                                                                                          np.mean(scores_window)
                                                                                         ))
    eps = max(cfg.AGENT.EPS_END, eps*cfg.AGENT.EPS_DECAY)