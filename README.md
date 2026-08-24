# CytoGen: Adaptive Synthetic Supervision for Microscopy Segmentation

This is the official implementation of **CytoGen: Adaptive Synthetic Supervision for Microscopy Segmentation**.

CytoGen generates paired microscopy images and instance masks by separating cellular-layout construction from microscopy-appearance rendering. It prioritizes biologically feasible cellular configurations that are underrepresented in real annotations and remain difficult for the current segmentation model.

## Environment Preparation

Please install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) first. The environment below reproduces the core software stack used in our experiments, including Python 3.11, PyTorch 2.9.1 with CUDA 12.8, and Hugging Face Diffusers 0.35.2.

```shell
git clone https://github.com/Zhiyuan-Weng/CytoGen.git
cd CytoGen
conda env create -f environment.yml
conda activate cytogen
```

Verify the installation with:

```shell
python -c "import torch, diffusers; print(torch.__version__, torch.version.cuda, diffusers.__version__)"
```

The reported experiments were run on NVIDIA H800 GPUs with 80 GB of memory. Other CUDA-capable NVIDIA GPUs can be used, although memory requirements depend on image resolution, batch size, and training stage.

## Method Overview

CytoGen consists of the following core stages:

1. **Biological constraint estimation:** summarize field-level cell count, foreground coverage, instance area, local density, contact number, and elongation from real training annotations.
2. **Failure-aware layout allocation:** estimate segmentation failure over cellular configuration space and prioritize biologically feasible configurations that remain difficult for the current model.
3. **Cellular layout generation:** construct instance masks while preserving object identity, morphology, contact, and occupancy constraints.
4. **Microscopy appearance generation:** adapt a Stable Diffusion appearance model and train an HSV mask-conditioned ControlNet.
5. **Downstream segmentation:** combine generated pairs with real data for downstream segmenter training.

## Repository Structure

```text
CytoGen/
├── environment.yml
├── README.md
├── cytogen/controller/    # Failure scoring and Controller fitting
├── cytogen/data/          # Preprocessing and HSV condition encoding
├── cytogen/downstream/    # Real/synthetic mixing and model adapters
├── cytogen/layout/        # Biological priors and layout generation
├── cytogen/predictions/   # Downstream prediction adapters
├── cytogen/rendering/     # ControlNet image rendering
├── cytogen/workflow/      # Iteration orchestration
├── scripts/               # Core command-line entry points
└── configs/               # Example manifests and workflow configuration
```

## Data Preprocessing

`scripts/prepare_data.py` converts TissueNet archives or generic paired-data manifests into normalized images, instance masks, HSV condition maps, and a unified `metadata.jsonl`. The HSV representation encodes category, compartment, and instance identity. A generic manifest example is provided in `configs/paired_manifest.example.jsonl`.

Datasets are not distributed with this repository and should be obtained from their original sources.

## Diffusion Training

`scripts/train_appearance.py` performs target-domain appearance adaptation. `scripts/train_controlnet.py` trains the HSV-conditioned ControlNet while keeping the appearance model frozen. Both scripts support synchronized spatial transforms and checkpoint resumption.

Pretrained models and experiment checkpoints are not distributed with this repository.

## Adaptive Layout Synthesis

`scripts/prepare_predictions.py` standardizes Cellpose2, Omnipose, and CelloType predictions. `scripts/fit_controller.py` estimates the failure landscape from real annotations and model predictions. `scripts/generate_layouts.py` then samples biologically constrained layouts using the learned failure scores and real-data priors.

## Rendering and Downstream Integration

`scripts/render_images.py` renders generated layouts with the adapted appearance model and ControlNet. `scripts/prepare_downstream.py` exports real-only, synthetic-only, or mixed datasets in manifest, Cellpose2, Omnipose, or CelloType format.

`scripts/run_iteration.py` connects prediction adaptation, Controller fitting, layout generation, rendering, and downstream export through `configs/iteration.example.yaml`. It records per-stage logs and supports resuming interrupted iterations.

Command-line options are documented directly in each entry script through `--help`.

## Acknowledgments

CytoGen builds on [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [Hugging Face Diffusers](https://github.com/huggingface/diffusers), [ControlNet](https://github.com/lllyasviel/ControlNet), [Cellpose](https://github.com/MouseLand/cellpose), [Omnipose](https://github.com/kevinjohncutler/omnipose), and CelloType. We thank the authors for their excellent work.

## Contact

For questions, please contact Zhiyuan Weng at `zhiyuanweng111@gmail.com`.
