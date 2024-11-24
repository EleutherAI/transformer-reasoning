from importlib.resources import files
from pathlib import Path

def get_project_root():
    return Path('/mnt/ssd-1/david/transformer-reasoning')
    # return files("transformer_reasoning")._paths[0].parent.parent

def get_src_root():
    return Path('/mnt/ssd-1/david/transformer-reasoning/src')
    # return files("transformer_reasoning")._paths[0]