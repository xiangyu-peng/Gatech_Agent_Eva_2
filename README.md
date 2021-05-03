# GATech SAIL-ON Agent implementation

Currently (for month 18 evaluation) the code being used to run our agent is, 
for for the most part, all is the `Evaluation_2/monopoly_simulator_2` subdirectory.

What follows are the setup instructions required to make it work on any machine.

## Python Environment

This implementation depends on a number of packages and is intended to be used with
an Anaconda or Miniconda environment. After cloning this repository, to create this environment, using the 
`environment.yml` file in the top directory, run:

```
conda env  create -f environment.yml
```

Once the environment has been successfully created, activate it by running:

```
conda activate monopoly-gatech
```

There are two other libraries that need to be installed before we can run the code:
the GNOME-P3 monopoly simulator itself, and our OpenAI Gym wrapper.

### Monopoly simulator

First, clone the repo found at [https://github.com/balloch/GNOME-p3](https://github.com/balloch/GNOME-p3). 
Then, at the top level of that repo, run:
```
pip install -e .
```

This will add the Monopoly simulator modules to your PYTHONPATH. Make sure you are doing this after
you have activated the `conda` environment so that you are installing in the right environment.

### OpenAI Gym installation

We need to manually add the custom Gym environment to your environment. To locate your instance
of Gym, run:

```
pip show gym
```

Then add the following code block to the end of the file `.../gym/envs/__init__.py`

        import sys
        sys.path.append('your_path/env/simulator_env')
        register(
            id='monopoly_simple-v1',
            entry_point='gym_simulator_env.envs:Sim_Monopoly'
        )

Install gym by going to `Gatech_Agent_Eva_2/env/simulator_env/gym_simulator_env` then run

        pip install .

## Absolute Paths
Regretably, this library requires some absolute paths
* modify absolute path in the following files. Search `becky` to find the path.
    * `/env/monopoly_world.py` and `/envs/simple_monopoly.py`
    * `/Evaluation_2/monopoly_simulator_2/A2C_agent_2` folder: `RL_agent_v1.py` (1) and `novelty_detection.py` (3)
    * `/KG_rule/openie_triple.py`
    * `/monopoly_simulator_background/interface.py`