# 🧠 RL & Imitation Learning Notebooks (Game AI Experiments)

A collection of practical experiments where AI agents learn to play games using Reinforcement Learning and Imitation Learning.

Instead of focusing on theory, this repository explores real challenges:

* training instability
* reward design
* exploration vs exploitation
* learning from imperfect demonstrations

## 🎮 What you'll find

* Agents learning to play fighting games and survival scenarios
* Imitation Learning (Behavioral Cloning, DAgger-style approaches)
* Reinforcement Learning (PPO, A2C, DQN)
* Custom preprocessing pipelines (frame stacking, cropping, grayscale)
* Real training runs, including failures and unexpected behaviors

## 🚀 Why this repo exists

Most RL repositories show clean results.
This one shows the messy reality of training agents — and how to make them work anyway.

## 📂 Structure

* `re4/` → Resident Evil experiments
* `stf6/` → Fighting game agents
* more experiments coming...

## 🔥Starting

1. Clone this repo: `https://github.com/paulo101977/notebooks-rl.git`

2. Move to cloned folder: `cd notebooks-rl`

3. First create a conda environment with python 3.11: `conda create -n env311 python=3.11`

4. Activate it: `conda activate env311`

5. Install requirements: `pip install -r requirements.txt`

6. Install Jax: `pip install git+https://github.com/araffin/sbx`

7. (Optional but recommended) Install Jupyter kernel:
`python -m ipykernel install --user --name env311 --display-name "Python (env311)"`

8. Run Jupyter-lab: `jupyter-lab.exe`

## ⚡ PyTorch (GPU Support)

If you have an NVIDIA GPU with CUDA support, install PyTorch separately:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

> ⚠️ Make sure to match the CUDA version (`cu118`, `cu121`, etc.) with your system.
> You can check compatibility here: <https://pytorch.org/get-started/locally/>

If you don't have a GPU, install the CPU version:
`pip install torch torchvision`

## ❤️ Support

If you find this useful, consider sponsoring:
<https://github.com/sponsors/paulo101977>
