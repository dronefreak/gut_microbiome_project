# Hydra Integration Guide

This guide explains how to integrate Hydra properly into the gut microbiome project, replacing the current plain YAML loading with dictionary access.

## Current State vs. Desired State

### Current (What We Have)
```python
# utils/data_utils.py
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# main.py
config = load_config()
dataset_path = config['data']['dataset_path']  # Dictionary access - ugly!
checkpoint = config['data']['mirobiome_transformer_checkpoint']
batch_size = config['data']['batch_size_embedding']
```

**Problems:**
- Dictionary access: `config['data']['dataset_path']` is verbose and error-prone
- No type safety: Typos only caught at runtime
- No IDE autocomplete
- No validation of required fields
- Can't override from command line easily
- No structured defaults

### Desired (With Hydra)
```python
# config.py
from dataclasses import dataclass
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

@dataclass
class DataConfig:
    dataset_path: str = MISSING
    srs_to_otu_parquet: str = "data_preprocessing/mapref_data/samples-otus-97.parquet"
    device: str = "cpu"
    # ... other fields

@dataclass
class Config:
    data: DataConfig
    # ... other sections

# main.py
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    dataset_path = cfg.data.dataset_path  # Clean dot notation!
    checkpoint = cfg.data.mirobiome_transformer_checkpoint
    batch_size = cfg.data.batch_size_embedding

    # Easy CLI override: python main.py data.device=cuda

if __name__ == "__main__":
    main()
```

**Benefits:**
- Clean dot notation: `cfg.data.dataset_path`
- Type safety with dataclasses
- IDE autocomplete
- Validation of required fields (MISSING)
- Easy command-line overrides
- Structured configuration with defaults

## Integration Plan

### Phase 1: Create Structured Config (30 minutes)

**Step 1.1: Create `config.py` in project root**

```python
# config.py
from dataclasses import dataclass, field
from typing import List, Dict, Any
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

@dataclass
class DataConfig:
    """Data paths and settings."""
    # Required fields (must be set)
    dataset_path: str = MISSING

    # Preprocessing paths
    srs_to_otu_parquet: str = "data_preprocessing/mapref_data/samples-otus-97.parquet"
    otu_to_dna_parquet: str = "data_preprocessing/mapref_data/otus_97_to_dna.parquet"

    # Embeddings paths
    dna_csv_dir: str = "data_preprocessing/dna_sequences"
    dna_embeddings_dir: str = "data_preprocessing/dna_embeddings"
    microbiome_embeddings_dir: str = "data_preprocessing/microbiome_embeddings"

    # Model paths
    mirobiome_transformer_checkpoint: str = "data/checkpoint_epoch_0_final_epoch3_conf00.pt"
    embedding_model: str = "neuralbioinfo/prokbert-mini-long"

    # Processing settings
    batch_size_embedding: int = 6
    device: str = "cpu"  # Options: cpu, cuda, mps


@dataclass
class ParamGrids:
    """Parameter grids for hyperparameter tuning."""
    logreg: Dict[str, Any] = field(default_factory=lambda: {
        "C": [0.001, 0.01, 0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "class_weight": ["balanced"],
        "solver": ["saga"],
        "max_iter": [1000, 2000]
    })

    svm: Dict[str, Any] = field(default_factory=lambda: {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
        "class_weight": ["balanced"]
    })

    rf: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "class_weight": ["balanced"]
    })

    mlp: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_layer_sizes": [[64], [128], [64, 32]],
        "activation": ["relu", "tanh"],
        "solver": ["adam"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate": ["constant", "adaptive"],
        "max_iter": [500, 1000]
    })


@dataclass
class ModelConfig:
    """Model configuration."""
    classifier: str = "logreg"  # Options: logreg, svm, rf, mlp
    use_scaler: bool = True
    param_grids: ParamGrids = field(default_factory=ParamGrids)


@dataclass
class EvaluationConfig:
    """Evaluation settings."""
    results_output_dir: str = "eval_results"
    cv_folds: int = 5
    grid_search_cv_folds: int = 5
    grid_search_scoring: str = "roc_auc"
    grid_search_random_state: int = 42
    final_eval_random_state: int = 123
    save_normalized_cm: bool = True


@dataclass
class Config:
    """Main configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def register_configs():
    """Register configs with Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)


# Register on import
register_configs()
```

**Step 1.2: Keep `config.yaml` (Hydra will still read it)**

Your existing `config.yaml` stays the same! Hydra will merge it with the structured config.

### Phase 2: Update Entry Points (30 minutes)

**Step 2.1: Update `main.py`**

```python
# main.py
import hydra
from omegaconf import DictConfig
from pathlib import Path
from typing import List, Optional, Dict, Any

from data_loading import load_dataset_df
from modules.classifier import SKClassifier
from utils.evaluation_utils import ResultsManager, EvaluationResult
from utils.data_utils import prepare_data
import config as config_module  # Import to register configs


def run_evaluation(cfg: DictConfig, classifiers: list = None):
    """
    Run evaluation pipeline for specified classifiers.

    Args:
        cfg: Hydra configuration (DictConfig)
        classifiers: List of classifier types to evaluate
    """
    # Load and prepare data - note the clean dot notation!
    print("Loading dataset...")
    dataset_df = load_dataset_df(cfg)
    X, y = prepare_data(dataset_df)
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    unique_labels = sorted(set(y))
    class_names = [str(label) for label in unique_labels]

    # Clean dot notation access
    results_manager = ResultsManager(
        results_output_dir=cfg.evaluation.results_output_dir,
        dataset_path=cfg.data.dataset_path,
        class_names=class_names
    )

    # Determine which classifiers to run
    if classifiers is None:
        classifiers = [cfg.model.classifier]

    # Run evaluation for each classifier
    for clf_type in classifiers:
        print(f"\n{'='*60}")
        print(f"Evaluating: {clf_type}")
        print(f"{'='*60}")

        classifier = SKClassifier(clf_type, cfg)
        metrics = classifier.evaluate_model(X, y, cv=cfg.evaluation.cv_folds)

        eval_result = EvaluationResult(
            classifier_name=metrics.classifier_name,
            y_true=metrics.y_true,
            y_pred=metrics.y_pred,
            y_prob=metrics.y_prob,
            cv_folds=metrics.cv_folds
        )

        results_manager.add_result(eval_result)
        results_manager.save_all_results(eval_result)

    if len(classifiers) > 1:
        results_manager.save_combined_report()
        results_manager.save_comparison_roc_curves()

    print(f"\n{'='*60}")
    print(f"All results saved to: {results_manager.output_dir}")
    print(f"{'='*60}")

    return results_manager


def run_grid_search_experiment(
    cfg: DictConfig,
    classifiers: Optional[List[str]] = None,
    custom_param_grids: Optional[Dict[str, Dict[str, Any]]] = None
):
    """Run grid search with unbiased final evaluation."""
    print("Loading dataset...")
    dataset_df = load_dataset_df(cfg)
    X, y = prepare_data(dataset_df)
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    unique_labels = sorted(set(y))
    class_names = [str(label) for label in unique_labels]

    results_manager = ResultsManager(
        results_output_dir=cfg.evaluation.results_output_dir,
        dataset_path=cfg.data.dataset_path,
        class_names=class_names
    )

    # Access param_grids cleanly - OmegaConf converts to dict automatically
    from omegaconf import OmegaConf
    config_param_grids = OmegaConf.to_container(cfg.model.param_grids, resolve=True)

    if custom_param_grids:
        param_grids = {**config_param_grids, **custom_param_grids}
    else:
        param_grids = config_param_grids

    if classifiers is None:
        classifiers = list(param_grids.keys())

    valid_classifiers = [c for c in classifiers if c in param_grids]

    if not valid_classifiers:
        raise ValueError("No classifiers with param_grids to evaluate.")

    best_params_summary = {}

    for clf_type in valid_classifiers:
        param_grid = param_grids[clf_type]

        classifier = SKClassifier(clf_type, cfg)

        # Clean dot notation for all config access
        metrics = classifier.grid_search_with_final_eval(
            X, y,
            param_grid=param_grid,
            grid_search_cv=cfg.evaluation.grid_search_cv_folds,
            final_eval_cv=cfg.evaluation.cv_folds,
            scoring=cfg.evaluation.grid_search_scoring,
            grid_search_random_state=cfg.evaluation.grid_search_random_state,
            final_eval_random_state=cfg.evaluation.final_eval_random_state,
            verbose=True
        )

        best_params_summary[clf_type] = metrics.best_params

        eval_result = EvaluationResult(
            classifier_name=metrics.classifier_name,
            y_true=metrics.y_true,
            y_pred=metrics.y_pred,
            y_prob=metrics.y_prob,
            cv_folds=metrics.cv_folds,
            additional_metrics={'best_params': metrics.best_params}
        )

        results_manager.add_result(eval_result)
        results_manager.save_all_results(eval_result)

    if len(valid_classifiers) > 1:
        results_manager.save_combined_report()
        results_manager.save_comparison_roc_curves()

    import json
    summary_path = results_manager.output_dir / "best_params_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(best_params_summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Grid Search Experiment Complete!")
    print(f"All results saved to: {results_manager.output_dir}")
    print(f"{'='*60}")

    print("\nBest Parameters Summary:")
    for clf_type, params in best_params_summary.items():
        print(f"  {clf_type}: {params}")

    return results_manager


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra."""

    # Option 1: Simple evaluation
    # run_evaluation(cfg)

    # Option 2: Multiple classifiers
    # run_evaluation(cfg, classifiers=["logreg", "rf", "svm", "mlp"])

    # Option 3: Grid search (recommended)
    run_grid_search_experiment(cfg, classifiers=["logreg", "rf", "svm", "mlp"])


if __name__ == "__main__":
    main()
```

**Step 2.2: Update function signatures across codebase**

Change all functions that take `config: dict` to take `cfg: DictConfig`:

```python
# Before
def some_function(config: dict):
    path = config['data']['dataset_path']

# After
from omegaconf import DictConfig

def some_function(cfg: DictConfig):
    path = cfg.data.dataset_path
```

### Phase 3: Update Module Files (45 minutes)

**Step 3.1: Update `modules/classifier.py`**

```python
# modules/classifier.py
from omegaconf import DictConfig

class SKClassifier:
    def __init__(self, classifier_type: str, cfg: DictConfig):  # Changed from config: dict
        self.classifier_type = classifier_type
        self.cfg = cfg  # Store as cfg

        # Clean dot notation access
        self.use_scaler = cfg.model.use_scaler

        # ... rest of __init__
```

**Step 3.2: Update `data_loading.py`**

```python
# data_loading.py
from omegaconf import DictConfig

def load_dataset_df(cfg: DictConfig):  # Changed from config: dict
    """Load dataset with embeddings."""

    # Clean dot notation throughout
    dataset_path = Path(cfg.data.dataset_path)
    microbiome_embeddings_dir = Path(cfg.data.microbiome_embeddings_dir)

    # ... rest of function using cfg.section.key instead of config['section']['key']
```

### Phase 4: Remove Old Config Loading (5 minutes)

**Step 4.1: Update `utils/data_utils.py`**

```python
# utils/data_utils.py
# Remove or deprecate load_config() function
# def load_config(config_path: str = "config.yaml"):
#     # DEPRECATED - use Hydra instead
#     ...

# Keep only data preparation functions
def prepare_data(dataset_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array(dataset_df['embedding'].tolist())
    y = np.array(dataset_df['label'])
    return X, y
```

### Phase 5: Migration Checklist

Go through each file and replace dictionary access with dot notation:

```python
# Find and replace pattern:

# Before                                    # After
config['data']['dataset_path']         →   cfg.data.dataset_path
config['data']['device']               →   cfg.data.device
config['model']['classifier']          →   cfg.model.classifier
config['model']['use_scaler']          →   cfg.model.use_scaler
config['evaluation']['cv_folds']       →   cfg.evaluation.cv_folds
config.get('evaluation', {})           →   cfg.evaluation  # Direct access
config['model']['param_grids']         →   cfg.model.param_grids
```

**Files to update:**
1. `main.py` ✓ (already shown above)
2. `data_loading.py`
3. `modules/classifier.py`
4. `modules/model.py`
5. `generate_embeddings.py` (if it uses config)
6. Any files in `example_scripts/` that load config

### Phase 6: Advanced Features (Optional)

**Feature 1: Command-line Overrides**

```bash
# Override any config value from CLI
python main.py data.device=cuda
python main.py model.classifier=rf
python main.py data.batch_size_embedding=16 evaluation.cv_folds=10

# Multiple overrides
python main.py data.device=cuda model.classifier=svm data.dataset_path=data/other.csv
```

**Feature 2: Config Groups (for multiple datasets)**

Create `conf/dataset/` folder:

```yaml
# conf/dataset/diabimmune.yaml
dataset_path: "data_preprocessing/datasets/diabimmune/Month_2.csv"

# conf/dataset/goldberg.yaml
dataset_path: "data_preprocessing/datasets/goldberg/T1.csv"
```

Use with:
```bash
python main.py dataset=diabimmune
python main.py dataset=goldberg
```

**Feature 3: Multirun (sweep parameters)**

```bash
# Run with multiple parameter combinations
python main.py -m data.device=cpu,cuda model.classifier=logreg,rf,svm

# This runs 6 experiments: 2 devices × 3 classifiers
```

## Testing the Migration

**Step 1: Test basic run**
```bash
python main.py
```

**Step 2: Test CLI overrides**
```bash
python main.py data.device=cuda
```

**Step 3: Verify dot notation works**
```python
# In any function with cfg parameter
print(cfg.data.dataset_path)  # Should work
print(cfg.data.device)  # Should work
```

**Step 4: Check validation**
```python
# Remove dataset_path from config.yaml temporarily
# Should raise error: "Missing mandatory value: data.dataset_path"
```

## Common Patterns

### Pattern 1: Accessing Nested Config
```python
# Before
path = config['data']['dataset_path']

# After
path = cfg.data.dataset_path
```

### Pattern 2: Getting with Default
```python
# Before
folds = config.get('evaluation', {}).get('cv_folds', 5)

# After
from omegaconf import OmegaConf
folds = OmegaConf.select(cfg, "evaluation.cv_folds", default=5)
# Or just access directly (will raise error if missing)
folds = cfg.evaluation.cv_folds
```

### Pattern 3: Checking if Key Exists
```python
# Before
if 'data' in config and 'dataset_path' in config['data']:
    path = config['data']['dataset_path']

# After
from omegaconf import OmegaConf
if OmegaConf.select(cfg, "data.dataset_path") is not None:
    path = cfg.data.dataset_path
# Or use hasattr
if hasattr(cfg, 'data') and hasattr(cfg.data, 'dataset_path'):
    path = cfg.data.dataset_path
```

### Pattern 4: Converting to Dict (for compatibility)
```python
# If you need a plain dict for some reason
from omegaconf import OmegaConf

config_dict = OmegaConf.to_container(cfg, resolve=True)
# Now config_dict is a regular Python dict
```

### Pattern 5: Iterating Over Config
```python
# Before
for key in config['model']['param_grids']:
    grid = config['model']['param_grids'][key]

# After
for key in cfg.model.param_grids:
    grid = cfg.model.param_grids[key]
# Or convert to dict first
param_grids = OmegaConf.to_container(cfg.model.param_grids)
for key, grid in param_grids.items():
    ...
```

## Troubleshooting

### Error: "Missing mandatory value: data.dataset_path"
**Cause**: Required field not set in config.yaml
**Solution**: Add the value to config.yaml or override from CLI

### Error: "AttributeError: 'DictConfig' object has no attribute 'xyz'"
**Cause**: Typo in config access or field doesn't exist
**Solution**: Check spelling, verify field exists in config.yaml

### Error: "ConfigStore instance has already been created"
**Cause**: Multiple imports of config.py
**Solution**: This is normal, Hydra handles it. Ignore the warning.

### None values from OmegaConf
**Cause**: Using `null` in YAML (becomes None)
**Solution**: Use proper default values or check for None

## Benefits After Migration

1. **Type Safety**: IDE knows what fields exist
2. **Autocomplete**: IDE suggests field names
3. **Cleaner Code**: `cfg.data.device` vs `config['data']['device']`
4. **Validation**: Missing required fields caught immediately
5. **CLI Overrides**: `python main.py data.device=cuda`
6. **Better Errors**: "Missing mandatory value: data.dataset_path" vs "KeyError: 'dataset_path'"
7. **Documentation**: Dataclasses serve as config documentation

## Summary

**Before Hydra:**
```python
config = load_config()
path = config['data']['dataset_path']  # Ugly, error-prone
```

**After Hydra:**
```python
@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    path = cfg.data.dataset_path  # Clean, type-safe
```

**Migration Time Estimate:**
- Phase 1 (Create config.py): 30 minutes
- Phase 2 (Update main.py): 30 minutes
- Phase 3 (Update modules): 45 minutes
- Phase 4 (Remove old code): 5 minutes
- Phase 5 (Find/replace): 20 minutes
- Testing: 20 minutes
- **Total: ~2.5 hours**

The investment pays off immediately with cleaner code and better maintainability!
