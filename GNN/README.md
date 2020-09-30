## GUIDE OF RUNNING EXPERIMENTS
### How To Install The Prerequisites?
- `conda create --name <env_name> --file GNN/requirements.txt`
- You can find the solution of configure gym in the README of the parent directory of current directory

### How to run the experiments - offline-training
- The only you need to use. `GNOME-p3/GNN/GAT_part.py`
- How to run it?
    - run baseline: `nohup python GAT_part.py --pretrain_model /media/becky/GNOME-p3/monopoly_simulator_background/weights19_1_baseline_seed_9147000.pkl --device_id 1 --novelty_change_num 18 --novelty_change_begin 1 --novelty_introduce_begin 0 --retrain_type baseline --kg_use False --exp_name 18_1_v --seed 10`
    - run kg-a2c: `nohup python GAT_part.py --pretrain_model /media/becky/GNOME-p3/monopoly_simulator_background/weights0_0_gat_part_seed_1048000.pkl --device_id 2 --novelty_change_num 18 --novelty_change_begin 1 --novelty_introduce_begin 0 --retrain_type gat_part --exp_name 18_1_v --seed 10`
- Which hyper-parameters you need to tune?
    - seed: Make sure you have 5 different seeds for each of your experiments.
    - novelty: Change the novelty by these 3 parsers. `--novelty_change_num ... --novelty_change_begin ... --novelty_introduce_begin ...`
        - novelty_change_num: The 1st one has to be0 and the 2nd one can range from 5 to 20. i.e. 0,20 
        - novelty_change_begin: The 1st one has to be0 and the 2nd one can range from 1 to 5. i.e. 0,1
        - novelty_change_begin: Fixed. 0,30
    - Remember to change --exp_name and make it corresponding to the novelty you use
    
- The plot we want
    - Each novelty has 2 curves/lines: baseline and kg-a2c
    - Each one has 5 seeds results, so we want a stats result, like its mean and std.
    - Give me at least 10 figures (12 different novelty, each with 5 seed = 10 experiments each, totally 120 experiments.)
    - novelty I want is as follows, 5-3,5-4,5-5,10-1,10-2,10-3,15-1,15-2,15-3,20-1,20-2,20-3

### How to run the experiments - online-training
- The only you need to use. `GNOME-p3/Hypothetical_simulator/online_test.py`
- How to run it?
    `nohup python online_test.py --interval 1 --retrain_nums 100 --device_id 1 --novelty_introduce_begin 50 --num_test 10000 --novelty_change_num 5 --novelty_change_begin 3 --retrain_type gat_pre --seed 10`
- Which you need to tune?
    - interval: 1/5/10/20/50/100
    - retrain_nums: 50/100/300/500/1000
    - novelty_introduce_begin: fixed 50
    - retrain_type baseline/ gat_pre
    - seed: at least 5 seeds for one novelty.
- What we want?
    - Find the best hyper-parameters
    - Give me plots, same with offline-learning plots
        