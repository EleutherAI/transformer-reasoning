from importlib.resources import files

def get_project_root():
    return files("transformer_reasoning")._paths[0].parent.parent

def get_src_root():
    return files("transformer_reasoning")._paths[0]