* All the related files are located in `~/A2C_agent/` folder. To run the test harness with our agent, please add this folder to your simulator folder.
* Requirements are in `~/A2C_agent/` folder. `pip install -r requirements.txt` to install all the requirements.
* To make the path correct, please add the following code in your gameplay.py file to call our agent:`import os, sys
curr_path = os.getcwd()
curr_path = curr_path.replace("/monopoly_simulator", "")
sys.path.append(curr_path + '/A2C_agent')
import RL_agent_v1`

* Before running `test_harness.py`, please run `~/A2C_agent/run_this_first.py` first. This is a file to clear the log files. It won't make any change in ur codes.
* After running the `test_harness.py`, if you want to check the novelty the agent learnt, please run `~/A2C_agent/novelty_show.py` to print the novelty.
 
