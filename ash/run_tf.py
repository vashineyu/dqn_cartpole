"""run.py
Run the training script
"""
import argparse
import gym
import os
import numpy as np
from pyvirtualdisplay import Display
from collections import deque

from cartpole.config import get_cfg_defaults
from cartpole.utils import screen_to_state, process_env_state_image
from cartpole.agent_tf import DqnAgent

parser = argparse.ArgumentParser()
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

display = Display(visible=0, size=cfg.SYSTEM.VIRTUAL_SCREEN)
display.start()
env = gym.make("CartPole-v0").unwrapped


agent = DqnAgent(input_shape=cfg.MODEL.INPUT_SIZE,
                 action_space=env.action_space.n,
                 gamma=cfg.AGENT.GAMMA,
                 memory_capacity=cfg.AGENT.NUM_MEMORY_CAPACITY,
                 batch_size=cfg.AGENT.BATCH_SIZE,
                 conv_mode=False
                 )
scores = []
scores_window = deque(maxlen=100)
eps = cfg.AGENT.EPS_START
for i_episode in range(cfg.AGENT.NUM_EPISODE):
    if cfg.AGENT.CONV_MODE:
        # use screen as state
        env.reset()
        state = process_env_state_image(env.render(mode="rgb_array"),
                                        cfg.MODEL.INPUT_SIZE)
    else:
        state = env.reset()

    total_rewards = 0
    for t_counter in range(cfg.AGENT.MAX_T):
        action = agent.act(state, eps)
        
        vector_state, reward, is_done, _ = env.step(action)
        total_rewards += reward
        
        if cfg.AGENT.CONV_MODE:
            # use screen as state
            if not is_done:
                next_state = process_env_state_image(env.render(mode="rgb_array"),
                                                     cfg.MODEL.INPUT_SIZE)
            else:
                next_state = np.zeros(cfg.MODEL.INPUT_SIZE, dtype=np.float32)
        else:
            # if use vector mode (non-conv mode)
            next_state = vector_state

        agent.step(state, action, next_state, reward, is_done * 1)
        state = next_state
        
        if is_done:
            break
    print("")
    scores.append(total_rewards)
    scores_window.append(total_rewards)
    print("\rEpisode {}, Accumulated Reward: {:.1f}, passed n mean reward: {:.1f}".format(i_episode,
                                                                                          total_rewards,
                                                                                          np.mean(scores_window)
                                                                                          ))
    eps = max(cfg.AGENT.EPS_END, eps*cfg.AGENT.EPS_DECAY)
