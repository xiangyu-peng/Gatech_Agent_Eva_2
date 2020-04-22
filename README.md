## Game Environment
### Environment Development
* The environment is developed as the form of [OpenIE gym library](http://gym.openai.com/docs/), which can find in **~/env**.
* **_Registeration guide_**
    1. Find the folder you install gym env package, then run `pip show gym`
    2. In the **~/envs/__init__.py** add the registeration line: 
```
register(
    id='monopoly_simple-v1',
    entry_point='gym_simulator_env.envs:Sim_Monopoly'
)
```

    3.Setup the env in **~/env/simulator_env/setup.py**:
    `python -m pip install. --user`

* Feature of Env:
    * Simulates the game step, resets the game, sets the game seed for recovering the game
    * Takes in action and outputs state, rewards and indicator of win or lose
    * Runs hypothetical game without affecting gameboard
    * Generate logging info for each game, including everything player can know from each step. *For example, number from dice, property price and etc.*
    * TODO: Add feature of experience replay

### [Monopoly Simulator](https://github.com/mayankkejriwal/GNOME-p3)

```
In order to make the debug process easier, we simplify the game.
1. The game never considers any trade between players
2. The game never forces the player to buy
3. The game never allows any actions other than post-roll
```

## Rule Learning and Detection
### [OpenIE Stanford NLP server](https://nlp.stanford.edu/software/openie.html)
* Logging info from environment will be extracted into **relation tuples**. *For example, Baltic-Avenue is colored as Brown => {'subject': 'Baltic-Avenue', 'relationship': 'is colored as', 'object': 'Brown'}*


### Knowledge Graph
* Knowledge graph is generated from the above tuples.
* There are two types of knowledge graph:
    * 'rel': Relationship is used as the key of knowledge graph.
    * 'sub': Space name is used as the key of knowledge graph.

### Rule Detection
* **History Record**: 

    Novelty like dice state or type change will be detected by recording the history of game simulations. *For example, we roll dice many times, and record these history to check the type of dice. If the type of dice changed, we can detect it by checking the new batch of generated history.*
* **Knowledge Graph Development**:

    Novelty like changing the rent of property can be easily detected by checking the developed knowledge graph.

## A2C Agent Model
### Vanilla A2C model
The latest model is located at here. Train the agent by `python vanilla_A2C_main_v4.py`. Weights will be saved during training and hyper-parameters are set here.

To use the generated model, run `python evaluate_A2C.py`, if you want to use your own evaluated method, you can edit the code here.

    
