# EXPT_CerealBar2

This experiment integrates CerealBar2 (or CB2), found at [`cb2.ai`](http://cb2.ai) with an fMRI experiment 
about the language and working memory/multiple-demand systems. In this directory we house code specifically 
pertaining to running the experiment in an fMRI scanner. The parent directory is a fork of the 
[cb2 repository](https://github.com/lil-lab/cb2), also found at https://github.com/EvLab-MIT/cb2.


# Usage

There are two components to running this experiment.

## 1. CB2 game server

Parallelly (refer to [parent README](../README.md) for more info), also run an instance of the
`cb2` server that we can interact with and use to serve scenarios: 
```bash
python3 -m server.main --config_filepath="server/config/local-config.yaml"
```

## 2. CB2 experiment driver

The experiment driver program is run as a module from the parent directory like so:
```bash
python -m fmri.main
```
