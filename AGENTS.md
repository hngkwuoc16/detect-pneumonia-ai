# Agent Instructions for Detect-Pneumonia-AI

## Project Overview

**Detect-Pneumonia-AI** is an explainable AI research prototype for binary pneumonia classification from chest X-ray images using DenseNet121. The project demonstrates clean ML architecture with modular components: data pipeline → model training → evaluation → explainability (GradCAM visualizations).

**Status**: Skeleton project—most implementation files are module templates. Focus on building coherent, compatible components.

## Essential Commands & Workflow

### Python Environment Setup
```bash
pip install -r requirements.txt
```

### Typical Development Workflow
1. **Load config**: Use `load_yaml("configs/config.yaml")` or `load_yaml("configs/model_config.yaml")`
2. **Set seed**: Call `set_seed(42)` early in any script for reproducibility
3. **Create directories**: Use `ensure_dir()` for output paths
4. **Build in order**: Dataset/DataModule → Model (Backbone + Classifier) → Trainer → Evaluator → GradCAM

### Testing
```bash
pytest tests/
```

## Key Conventions

### Configuration
- **config.yaml**: Project paths, data directories, image size (224×224), num_channels (3 for RGB)
- **model_config.yaml**: Hyperparameters (batch_size=32, epochs=20, learning_rate=0.0001, dropout=0.2)
- Always use `load_yaml()` from `src/utils.py` and validate config keys early

### File Operations
- Always use `pathlib.Path` (not `os.path`)
- Use `ensure_dir()` from `src/utils.py` to create output directories
- Outputs go to `outputs/` (checkpoints, logs, reports)

### Data & Images
- Input size: 224×224 RGB (3 channels)
- Use albumentations for transforms
- Data flows: raw images → Dataset → DataModule → DataLoader

### Model Architecture
- **Backbone**: DenseNet121 pretrained from torchvision
- **Classifier Head**: Custom binary classification head
- **Loss**: BCEWithLogitsLoss (binary cross-entropy)
- **Optimizer**: Adam
- **Regularization**: Dropout=0.2

### Reproducibility
- Always call `set_seed(42)` in training scripts
- Seeds set: random, numpy, torch, CUDA
- Match seed value to `config.yaml` project.seed

## Directory Structure & Components

```
src/
├── data/           # Data loading pipeline
│   ├── dataset.py      # PyTorch Dataset class
│   ├── datamodule.py   # DataModule for train/val/test splits
│   └── transforms.py   # Albumentations transforms
├── models/
│   ├── backbone.py     # DenseNet121 backbone
│   └── classifier.py   # Binary classification head
├── training/
│   ├── trainer.py      # Training loop
│   ├── callbacks.py    # Training callbacks
│   └── metrics.py      # Binary classification metrics
├── evaluation/
│   └── evaluator.py    # Evaluation on test set
├── explainability/
│   └── gradcam.py      # GradCAM visualizations
└── utils.py            # set_seed, load_yaml, ensure_dir
```

## Common Pitfalls

1. **Component Compatibility**: This is a skeleton—ensure data transforms, model input size, and GradCAM conv layers align (all 224×224 RGB, GradCAM needs final classifier conv layers)

2. **Seed Management**: Don't forget `set_seed()` early in scripts; without it, runs are not reproducible

3. **GPU Memory**: Default batch_size=32 is reasonable; adjust in model_config.yaml if OOM errors occur

4. **Config Loading**: Always validate that required keys exist in loaded YAML before using them

5. **Path Handling**: Use absolute Path objects or resolve relative paths early to avoid issues with working directory changes

## Quick Start for Agents

- **Exploring code**: Check `src/utils.py` first for helper functions
- **Understanding flow**: Review `src/training/trainer.py` (once implemented) to see component integration
- **Adding features**: Follow modular pattern—isolate changes to single components (e.g., new transform only touches `src/data/transforms.py`)
- **Testing changes**: Implement unit tests in `tests/` following pytest conventions
- **Debugging**: Config-driven design means most issues are config or data mismatch—check config.yaml and data paths first

## Documentation & References

- [docs/ROADMAP.md](docs/ROADMAP.md) — Feature roadmap and next steps
- [docs/DECISIONS.md](docs/DECISIONS.md) — Architectural decisions and rationale
- [docs/REFERENCES.md](docs/REFERENCES.md) — Research papers and external resources
- [setup.py](setup.py) — Package metadata and installation options

## Notes for AI Agents

- This project is currently a **skeleton with empty module files**—treat it as defining interfaces and architecture, not as runnable code yet
- When implementing a module, follow the existing patterns (e.g., config loading in `__init__`, utility functions in utils.py)
- If unsure about a pattern, check similar projects using PyTorch + config-driven ML workflows
- Use type hints and docstrings consistently
- Run `pytest tests/` after implementing modules to catch integration issues early
