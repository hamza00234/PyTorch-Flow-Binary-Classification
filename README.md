# PyTorch Binary Classification

This repository contains two Jupyter notebooks that demonstrate core PyTorch workflows and toy classification problems using PyTorch and scikit-learn. Both notebooks are meant for learning and experimentation.

**Notebooks Included**
- `01pytorch_workflow.ipynb`: Basic PyTorch workflow and linear regression example.
  - **Purpose:** Introduces PyTorch fundamentals: tensors, building `nn.Module` models, training loop (forward, loss, backward, optimizer), evaluation modes, saving and loading model state dictionaries.
  - **Highlights:**
    - Creates a simple linear regression model from scratch (`linear_regression` class).
    - Demonstrates training and validation loops, loss plotting, inference with `torch.inference_mode()` / `torch.no_grad()`.
    - Shows how to save and load a model using `torch.save()` and `load_state_dict()`.

- `02_pytorch_Bclassification.ipynb`: Binary and multiclass classification examples.
  - **Purpose:** Builds on fundamentals to show how to prepare data for classification, construct neural networks for binary and multiclass tasks, and visualize decision boundaries.
  - **Highlights:**
    - Generates toy datasets with `sklearn.datasets.make_circles` (binary) and `make_blobs` (multiclass).
    - Converts numpy arrays to PyTorch tensors and performs train/test splitting.
    - Implements models with `nn.Module` and `nn.Sequential` (including non-linear activations like `ReLU`).
    - Uses `BCEWithLogitsLoss` for binary classification and `CrossEntropyLoss` for multiclass.
    - Trains models, computes accuracy, and visualizes decision boundaries using a helper `plot_decision_boundary` function (downloaded via `helper_functions.py` in the notebook).

**Dependencies**
- Python 3.8+ recommended.
- Key Python packages used in the notebooks:
  - `torch` (PyTorch)
  - `scikit-learn`
  - `matplotlib`
  - `pandas` (used in `02_pytorch_Bclassification.ipynb` for DataFrame display)
  - `requests` (used to download `helper_functions.py` in notebook 2)

You can install these in a virtual environment. Example (PowerShell):

```powershell
# create venv
python -m venv .venv
# activate venv
.\.venv\Scripts\Activate.ps1
# upgrade pip
python -m pip install --upgrade pip
# install required packages
pip install torch scikit-learn matplotlib pandas requests
```

Notes about GPU usage:
- The notebooks automatically select `cuda` if available (via `device = "cuda" if torch.cuda.is_available() else "cpu"`). GPU acceleration is optional — the code runs on CPU if CUDA isn't present.

How to run the notebooks
- Open the project folder in VS Code or Jupyter Lab/Notebook.
- Ensure the virtual environment is activated and dependencies installed.
- Launch the notebook server and open either notebook:
  - `01pytorch_workflow.ipynb` — start here if you are new to PyTorch basics.
  - `02_pytorch_Bclassification.ipynb` — follow this after notebook 1 to explore classification tasks and visualizations.

Special note for `02_pytorch_Bclassification.ipynb`:
- Notebook 2 downloads `helper_functions.py` from the web at runtime (this provides `plot_predictions` and `plot_decision_boundary`). If you prefer, you can download that helper file manually into the notebook directory before running.

Suggested next steps / ideas
- Convert the experiments into script files (e.g., `train.py`) and add a `requirements.txt` or `pyproject.toml` for reproducible environments.
- Add unit tests or small dataset smoke tests to validate training/inference behavior.
- Experiment with different optimizers, learning rates, and model architectures; log results with TensorBoard or `torch.utils.tensorboard`.

License & Attribution
- These notebooks are educational examples. If you reuse or adapt code, please retain attribution and add a license of your choice.

---

If you want, I can also add a `requirements.txt`, create a small `train.py` script, or open a PR with these changes. Which would you like next?
