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
from cartpole.utils import ReplayMemory, screen_to_state
from cartpole.agent_tf import DqnAgent

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

display = Display(visible=0, size=cfg.SYSTEM.VIRTUAL_SCREEN)
display.start()
env = gym.make("CartPole-v0").unwrapped

memory = ReplayMemory(capacity=int(cfg.AGENT.NUM_MEMORY_CAPACITY))
agent = DqnAgent(action_space=env.action_space.n, gamma=cfg.AGENT.GAMMA, memory=memory,
                 batch_size=cfg.AGENT.BATCH_SIZE, 
                 input_shape=cfg.MODEL.INPUT_SIZE,
                 fully_random_mode=False,
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
            
        memory.push(state, action, next_state, reward, is_done*1)
        state = next_state
        
        # Train the model
        agent.learn()
        
        print("\rEpisode {}, Accumulated Reward: {:.3f}, elapse time: {}".format(i_episode, total_rewards, t_counter+1), end='')
        
        if is_done:
            break
    print("")
    scores.append(total_rewards)
    scores_window.append(total_rewards)
    eps = max(cfg.AGENT.EPS_END, eps*cfg.AGENT.EPS_DECAY)
