04-05-20 Update for knowledge graph

Class KG_OPENIE is in GNOME-p3/KG-rule/openie_triple.py, which is able to turn logging info to game rule.

Feature:

1. logging info from env is extracted as a triple (sub, rel, obj), then put in a dict() to record it.

2. KG_OPENIE can quickly detect if the rule exists or not, or changed

03-04-20 Update how to register the env

`cd /GNOME-p3/env/simulator_env` 

`python3 -m pip install . --user`

`pip show gym`

`cd gym/envs`

`nano __init__.py`

Then import sys and add /media/becky/GNOME-p3/env/simulator_env (plz change to ur path to env)

Then also add 

`register(
    id='monopoly_simple-v1',
    entry_point='gym_simulator_env.envs:Sim_Monopoly'
)`

\----------------------------------------------------\

02-27-20 becky

create feature branch

02-21-20 Becky

1.This is a simpler version of game, only consider mortgage, free-mortgage, buy and improve the property
The entry point for the simulator is gameplay_simple_becky_v1.py and can be run on the command line:

$ python3 gameplay_simple_becky_v1.py > log.txt

2.The knowledge graph part is in /KG-rule.
The entry point for the simulator is kg-build.py and can be run on the command line:

$ python3 kg-build.py



## Game Environment
### Environment Development
* The environment is developed as the form of [OpenIE gym library](http://gym.openai.com/docs/), which can find in **~/env**.
* **_Registeration guide_**
    1. Find the folder you install gym env package, then run `pip show gym`
    2. In the **~/envs/__init__.py** add the registeration line:
`register(
    id='monopoly_simple-v1',
    entry_point='gym_simulator_env.envs:Sim_Monopoly'
)`

    3. Setup the env in **~/env/simulator_env/setup.py**:
    `python -m pip install. --user`

* Feature of Env:
    * Simulates the game step, resets the game, sets the game seed for recovering the game
    * Takes in action and outputs state, rewards and indicator of win or lose
    * Runs hypothetical game without affecting gameboard
    * TODO: Add feature of experience replay

### [Monopoly Simulator](https://github.com/mayankkejriwal/GNOME-p3)

```
In order to make the debug process easier, we simplify the game.
1. The game never considers any trade between players
2. The game never forces the player to buy
3. The game never allows any actions other than post-roll
```
