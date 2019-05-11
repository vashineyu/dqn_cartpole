from yacs.config import CfgNode as CN

_C = CN()
_C.SYSTEM = CN()
_C.SYSTEM.DEVICES = []

def get_cfg_defaults():
    return _C.clone()