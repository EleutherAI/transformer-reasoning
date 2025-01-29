import argparse
from pathlib import Path
import yaml
import subprocess
from transformer_reasoning.evaluation.spec_config import ModelSpec

def load_spec(spec_path: str) -> ModelSpec:
    """Load specification from YAML file."""
    with open(spec_path) as f:
        spec_dict = yaml.safe_load(f)
    return ModelSpec(**spec_dict)

def run_loss_over_time(spec: ModelSpec):
    """Run loss over time analysis."""
    cmd = ["python", "-m", "transformer_reasoning.evaluation.loss_over_time"]
    
    # Add command line options
    if spec.cmd_options.skip_mode:
        cmd.append("--skip_mode")
    if spec.cmd_options.no_mup:
        cmd.append("--no_mup")
    if spec.cmd_options.compute_histograms:
        cmd.append("--compute_histograms")
    if spec.cmd_options.latest_only:
        cmd.append("--latest_only")
    if spec.cmd_options.base_path != ".":
        cmd.extend(["--base_path", spec.cmd_options.base_path])
    if spec.cmd_options.output_dir != ".":
        cmd.extend(["--output_dir", spec.cmd_options.output_dir])
    
    # Add commit hashes
    cmd.extend(["--commit_hashes"] + spec.commit_hashes)
    
    # Add relations if specified
    if spec.cmd_options.relations:
        # Convert None to "None" in the command line
        relation_args = [str(r) for r in spec.cmd_options.relations if r is not None ]
        cmd.extend(["--relations"] + relation_args)
    
    # Run the command
    subprocess.run(cmd, check=True)

def run_measure_capacity(spec: ModelSpec):
    """Generate capacity plots."""
    cmd = ["python", "-m", "transformer_reasoning.evaluation.measure_capacity"]
    
    # Add command line options
    if spec.cmd_options.scheme:
        cmd.extend(["--scheme", spec.cmd_options.scheme])
    if spec.cmd_options.selection_scheme:
        cmd.extend(["--selection_scheme", spec.cmd_options.selection_scheme])
    if spec.cmd_options.skip_mode:
        cmd.append("--skip_mode")
    if spec.cmd_options.subjectwise:
        cmd.append("--subjectwise")
    if spec.cmd_options.no_mup:
        cmd.append("--no_mup")
    if spec.cmd_options.compute_histograms:
        cmd.append("--compute_histograms")
    if spec.cmd_options.base_path != ".":
        cmd.extend(["--base_path", spec.cmd_options.base_path])
    if spec.cmd_options.output_dir != ".":
        cmd.extend(["--output_dir", spec.cmd_options.output_dir])
    if spec.cmd_options.hops:
        cmd.extend(["--hops", str(spec.cmd_options.hops)])
    
    # Add filtering parameters
    if spec.n_params:
        cmd.extend(["--n_params"] + [str(x) for x in spec.n_params])
    if spec.N_profiles:
        cmd.extend(["--N_profiles"] + [str(x) for x in spec.N_profiles])
    if spec.layers:
        cmd.extend(["--layers"] + [str(x) for x in spec.layers])
    if spec.min_train_hops:
        cmd.extend(["--min_train_hops"] + [str(x) for x in spec.min_train_hops])
    if spec.max_train_hops:
        cmd.extend(["--max_train_hops"] + [str(x) for x in spec.max_train_hops])
    if spec.weight_decay:
        cmd.extend(["--weight_decay"] + [str(x) for x in spec.weight_decay])
    
    # Add commit hashes
    cmd.extend(["--commit_hashes"] + spec.commit_hashes)
    
    # Add relations if specified
    if spec.cmd_options.relations:
        relation_args = [str(r) for r in spec.cmd_options.relations if r is not None]
        if relation_args:
            cmd.extend(["--relations"] + relation_args)
    
    # Run the command
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec_file", type=str, help="Path to YAML specification file")
    args = parser.parse_args()
    
    spec = load_spec(args.spec_file)
    
    # Create output directory
    Path(spec.cmd_options.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(spec.cmd_options.output_dir) / 'results').mkdir(parents=True, exist_ok=True)
    
    # Run specified analysis steps in order
    for step in spec.run_steps:
        if step == "loss_over_time":
            print("Running loss over time analysis...")
            run_loss_over_time(spec)
        elif step == "measure_capacity":
            print("Generating capacity plots...")
            run_measure_capacity(spec)

if __name__ == "__main__":
    main() 