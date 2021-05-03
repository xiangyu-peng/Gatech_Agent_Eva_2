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
Regrettably, this library requires some absolute paths. To run, in the following files 
search for the keyword 'becky' to find the absolute paths that you need to modify:
  * `Gatech_Agent_Eva2/env/monopoly_world.py`
  * `Gatech_Agent_Eva2/envs/simple_monopoly.py`
  * `Gatech_Agent_Eva2/Evaluation_2/monopoly_simulator_2/A2C_agent_2/RL_agent_v1.py`
  * `Gatech_Agent_Eva2/Evaluation_2/monopoly_simulator_2/A2C_agent_2/novelty_detection.py`
  * `Gatech_Agent_Eva2/KG_rule/openie_triple.py`
  * `Gatech_Agent_Eva2/monopoly_simulator_background/interface.py`