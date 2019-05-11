"""run.py
Run the training script
"""

import gym
import os
from cartpole.config import get_cfg_defaults

cfg = get_cfg_defaults()
devices = ",".join(str(i) for i in cfg.SYSTEM.DEVICES)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = devices
