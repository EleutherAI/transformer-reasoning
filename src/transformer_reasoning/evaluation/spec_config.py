from typing import List, Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, field_validator
from typing_extensions import Annotated

class CommandConfig(BaseModel):
    skip_mode: bool = False
    subjectwise: bool = False
    no_mup: bool = False
    base_path: str = "."
    compute_histograms: bool = False
    latest_only: bool = False
    relations: Optional[List[Optional[int]]] = None
    output_dir: str = "."
    hops: int = 2
    scheme: str = "2-hop-double"
    selection_scheme: str = "enumerate"

    @field_validator('relations')
    def validate_relations(cls, v):
        if v is None:
            return None
        return [None if x is None or x == "None" else int(x) for x in v]

class ModelSpec(BaseModel):
    commit_hashes: List[str]
    n_params: Optional[List[int]] = None
    N_profiles: Optional[List[int]] = None
    layers: Optional[List[int]] = None
    min_train_hops: Optional[List[int]] = None
    max_train_hops: Optional[List[int]] = None
    weight_decay: Optional[List[float]] = None
    
    # Analysis steps to run
    run_steps: List[Literal["loss_over_time", "measure_capacity"]]
    
    # Command line options
    cmd_options: CommandConfig = CommandConfig()

    def matches_config(self, config: Dict[str, Any]) -> bool:
        """Check if a model config matches this specification."""
        for param, values in {
            'n_params': self.n_params,
            'N_profiles': self.N_profiles,
            'layers': self.layers,
            'min_train_hops': self.min_train_hops,
            'max_train_hops': self.max_train_hops,
            'weight_decay': self.weight_decay,
        }.items():
            if values is not None and config[param] not in values:
                return False
        return True 