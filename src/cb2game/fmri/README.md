# EXPT_CerealBar2

This experiment integrates CerealBar2 (or CB2), found at [`cb2.ai`](http://cb2.ai) with an fMRI experiment 
about the language and working memory/multiple-demand systems. In this directory we house code specifically 
pertaining to running the experiment in an fMRI scanner. The parent directory is a fork of the 
[cb2 repository](https://github.com/lil-lab/cb2), also found at https://github.com/EvLab-MIT/cb2.


# Usage

There are two components to running this experiment.

## 1. CB2 game server

Ensure you are in the conda environment you used to clone cb2 when doing this step. If you have not set up a conda environment, refer to the main README for cb2, or do so with the following command:
```bash
conda create -n cb2env python=3.11
conda activate cb2env
```

In one terminal, parallelly run an instance of the
`cb2` server that we can interact with and use to serve scenarios (refer to [parent README](../README.md) for more info). Use the cb2fmri.yaml found in the cb2 directory: 
```bash
python -m cb2game.server.main --config_filepath=cb2fmri.yaml
```

You can now access the game instance at `http://localhost:8080/`

If the server does not run, try running the following command first to set up the Unity client:
```bash
python -m cb2game.server.fetch_client
```

## 2. CB2 experiment driver

The experiment driver program is run as a module from the parent directory. Open a new terminal in the same conda environment. Move to the src directory, which should contain the cb2game package. Start the experiment with the following command:
```bash
python -m cb2game.fmri.main <ARGS>
```

<ARGS> should contain parameters for subject_id, run_number, run_set, task_difficulty, and linguistic_complexity.

The experiment driver uses pygame 2.1.2, which can cause problems on ARM Macs. The game will still run properly with pygame 2.1.3. If running the experiment causes an error, try installing the pynput package independently with the following command:
```bash
pip install pynput
```
