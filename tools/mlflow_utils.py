# tools/mlflow_utils.py
from __future__ import annotations
import os, json, subprocess
from pathlib import Path
from typing import Dict, Any
import mlflow

def set_tracking(experiment_name: str, tracking_uri: str | None = None):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    else:
        mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)

def current_git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return None

def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_dict(v, key + "."))
        else:
            out[key] = v
    return out

def log_yaml_as_params_and_artifact(cfg_path: str | Path, artifact_subdir="configs"):
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        return
    mlflow.log_artifact(str(cfg_path), artifact_path=artifact_subdir)
    try:
        import yaml
        y = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        flat = flatten_dict(y)
        # Limita cantidad para UI
        for k, v in list(flat.items())[:200]:
            mlflow.log_param(k, v)
    except Exception:
        pass

def start_run(run_name: str, tags: Dict[str, str] | None = None):
    r = mlflow.start_run(run_name=run_name)
    if tags:
        for k, v in tags.items():
            if v is not None:
                mlflow.set_tag(k, v)
    commit = current_git_commit()
    if commit:
        mlflow.set_tag("git_commit", commit)
    return r

def log_epoch_metrics(epoch: int, metrics: Dict[str, float], prefix: str = "train_"):
    for k, v in metrics.items():
        mlflow.log_metric(f"{prefix}{k}", float(v), step=epoch)

def log_artifacts_safely(path: str | Path, artifact_path: str | None = None):
    p = Path(path)
    if p.exists():
        mlflow.log_artifact(str(p), artifact_path=artifact_path)

def log_model_file(model_path: str | Path, artifact_subdir="models"):
    p = Path(model_path)
    if p.exists():
        mlflow.log_artifact(str(p), artifact_path=artifact_subdir)
