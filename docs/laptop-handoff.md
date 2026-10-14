# Continue this project from another computer

Repository: [samuel1tsao/ai-driving-assistance](https://github.com/samuel1tsao/ai-driving-assistance), private.

```powershell
git clone https://github.com/samuel1tsao/ai-driving-assistance.git
cd ai-driving-assistance
```

Sign in to GitHub as an account with repository access when Git prompts. Commit and push changes from either computer, then pull on the other before continuing work.

The Git repository contains the active notebooks, support code, scripts, tests, documentation, earlier application source, and small reports. Local datasets, downloaded model weights, virtual environments, caches, generated archives, executed validation notebooks, large generated input manifests, and downloaded third-party runtime repositories are excluded. These remain on the original computer or Google Drive; cloning Git does not copy training checkpoints. Archived notebooks retain their third-party project setup instructions.

## Run the notebook

Open `notebooks/fog_density_study.ipynb` in Colab, or use the [current fog density study notebook](https://colab.research.google.com/drive/1kyRGjbEEA6I-9Xja-e4OL3jcYybQlyLC). Edit the configuration cell. Use `TRAINING_PROFILE="setup"` for an all-model smoke run before expensive training, then choose `partial-2k`, `partial-5k`, or `full`.

To resume the existing Drive study, keep the same data, training settings, `output.location`, and `output.run_name`. The configured output directory is `MyDrive/fog-study/runs/colab-full-study`. Mount the same Google Drive account. Use only one training session per output directory.

The **Overall progress and checkpoint continuation** table lists every configured density and detector run, completed epochs, saved image steps in the current epoch, overall training percentage, and status. The percentage includes earlier sessions and measures training image steps, not elapsed time or evaluation completion. Each resumed trainer prints cumulative loss history from its checkpoint. Loss charts include previous epochs because the checkpoint retains the full history. Partial profiles use separate output directories and never warm-start from another profile.

If Colab terminates, reconnect and Run all with the same settings. The engine restores the most recent completed checkpoint; unsaved work is replayed. Do not change experiment settings while reusing a checkpoint directory. Package-version mismatches may require recreating the original environment or choosing a new run directory.

## Local environment

Install Python 3.11 and Git, create a virtual environment, then install `requirements-fog-research.txt`, `requirements-notebook.txt`, and Ruff. GPU setup must match the local CUDA/PyTorch environment. The consolidated notebook's prepared-data dependencies are defined in its package section; older scripts retain their separate requirements.

Run the inexpensive checks before training:

```powershell
python -m unittest discover -s tests -v
python scripts/check_notebook_code.py
```

The regression suite imports the notebook's actual implementation. Eleven tests cover training reference parity, cumulative progress across resumed epochs, interruption recovery, atomic checkpoint failure, concurrent writers, and prediction reuse. Both real-data subset modes previously passed; the continuation addition is also checked through a completed-run ZIP subset rerun.

See [coding standards](coding-standards.md) and [notebook configuration and validation](fog-density-consolidated-notebook.md). The existing full-study completion and forced Colab runtime termination are still unverified.
