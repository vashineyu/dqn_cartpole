from yacs.config import CfgNode as CN

_C = CN()
_C.SYSTEM = CN()
_C.SYSTEM.DEVICES = []
_C.SYSTEM.VIRTUAL_SCREEN = (400, 600)

_C.AGENT = CN()
_C.AGENT.NUM_EPISODE = 50
_C.AGENT.NUM_MEMORY_CAPACITY = 1e+6
_C.AGENT.BATCH_SIZE = 128
_C.AGENT.GAMMA = 0.999
_C.AGENT.EPS_START = 0.9
_C.AGENT.EPS_END = 0.05
_C.AGENT.EPS_DECAY = 200
_C.AGENT.TARGET_UPDATE = 10

_C.MODEL = CN()
_C.MODEL.OPTIMIZER = "sgd"


def get_cfg_defaults():
    return _C.clone()