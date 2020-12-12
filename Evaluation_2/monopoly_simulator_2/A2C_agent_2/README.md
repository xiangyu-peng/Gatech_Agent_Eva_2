#### WHAT NEED TO BE MODIFIED BEFORE USING
##### GYM installation
    
* find gym path
        
        pip show gym

* add the following codes to `.../gym/envs/__init__.py`

        import sys
        sys.path.append('your_path/env/simulator_env')
        register(
            id='monopoly_simple-v1',
            entry_point='gym_simulator_env.envs:Sim_Monopoly'
        )

* install gym by going to `your_path/env/simulator_env/gym_simulator_env` then run

        pip install .

##### PATH
* modify absolute path in the following files. Search `becky` to find the path.
    * `/env/monopoly_world.py` and `/envs/simple_monopoly.py`
    * `/Evaluation_2/monopoly_simulator_2/A2C_agent_2` folder: `RL_agent_v1.py` (1) and `novelty_detection.py` (3)
    * `/KG_rule/openie_triple.py`
    * `/monopoly_simulator_background/interface.py`