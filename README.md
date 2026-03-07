# ResNet PyTorch Implementation

A from-scratch PyTorch implementation of the Deep Residual Learning architecture (ResNet) for image classification. This repository includes the code to generate ResNet-50, ResNet-101, and ResNet-152 models as described in the original 2015 paper by Kaiming He et al.

## 📂 Project Structure

* `resnet.py`: Contains the core PyTorch implementation of the ResNet architecture, including the residual blocks and model definitions.
* `paper_Resnet.pdf`: A local copy of the original research paper ("Deep Residual Learning for Image Recognition") for quick reference to the architecture's mathematical and structural details.
* `pyproject.toml` & `poetry.lock`: Dependency management and packaging files.
* `.venv/`: Local virtual environment.
* `.gitignore`: Specifies intentionally untracked files to ignore for Git.

## 🛠️ Installation & Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management. To set up the environment and run the code, follow these steps:

1.  **Ensure Poetry is installed** on your system.
2.  **Install dependencies** (this will read the `pyproject.toml` and `poetry.lock` files to set up your `.venv`):
    ```bash
    poetry install
    ```
3.  **Activate the virtual environment**:
    ```bash
    poetry shell
    ```

*Note: If you are bypassing Poetry and just using standard pip, ensure you have `torch` installed in your environment.*

## 🚀 Usage

You can run the main script to execute the built-in test, which initializes a ResNet-50 model, passes dummy data through it, and outputs the resulting tensor shape:

```bash
python resnet.py