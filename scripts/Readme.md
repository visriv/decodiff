# DecoDiff Execution Guide

This guide provides instructions on how to run the main training scripts for different diffusion model configurations within the `decodiff` project.

## Prerequisites

Before running the scripts, ensure you have navigated to the project directory and set up the necessary environment variables.

1.  **Change Directory:** Navigate to the root of the `decodiff` project.
    ```bash
    cd decodiff
    ```

2.  **Enable Weights & Biases (Optional):** If you are using Weights & Biases for logging, ensure it's enabled.
    ```bash
    wandb online
    ```
    *(Note: Use `wandb offline` if you prefer offline logging.)*

3.  **Set PYTHONPATH:** Add the project directory to your `PYTHONPATH` to ensure modules are correctly imported.
    ```bash
    export PYTHONPATH=/path/to/your/decodiff
    ```
    *(Replace `/path/to/your/decodiff` with the actual absolute path to the `decodiff` directory, e.g., `/home/users/nus/e1333861/decodiff`)*

## Running Diffusion Models

The following commands demonstrate how to run different types of diffusion models using the provided configuration files.

### ControlNet-based Diffusion (connecting the decoders, like in controlNet )

This uses the `main_distri.py` script with the `diffusion_control_connect.yaml` configuration. (controlNet architecture with denoising objective)

```bash
python scripts/main_distri.py --config ./configs/diffusion_control_connect.yaml
```

### ControlNet-based Diffusion (Direct Estimation of signal (aka Score Based Diffusion))
This uses the main_distri_new.py script with the diffusion_control_direct.yaml configuration. (controlNet architecture with score estimation objective)
```bash
python scripts/main_distri_new.py --config ./configs/diffusion_control_direct.yaml
```

### Vanilla Diffusion

This uses the main_distri_new.py script with the diffusion_single.yaml configuration.

```bash
python scripts/main_distri_new.py --config ./configs/diffusion_single.yaml
```

Note on GPU TrainingDistributed Training: The scripts main_distri.py and main_distri_new.py are intended for distributed training across multiple GPUs. Ensure your environment (e.g., Slurm, torch.distributed.launch) is configured correctly for multi-GPU execution when using these scripts. The specific launch command might vary depending on your setup.Single GPU Training: If you intend
