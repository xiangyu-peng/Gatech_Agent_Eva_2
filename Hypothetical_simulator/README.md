#### Summary of KG-A2C in Monopoly
##### Offline 
- Offline means assuming baseline and KG-A2C detects novelty at the same time and begin retraining after pretraining.
- It can help converge faster, however, **not a big difference**. It provides more information, but limited.
- It is sometimes **not stable**, the winning rate may be lower than vanilla A2C after 100,000 iterations

##### Online
- Online means our agent and baseline interacts with real env and at the same time detect novelty and do retraining.
- KG-A2C converges faster. This is mostly because KG-A2C detects novelty and begins retraining much faster than baseline. 
- Baseline need to detect winning rate drop then begin retraining, so it usually take another 50 - 100 rounds of games.
- If the novelty is the one which KG-A2C cannot detect, it will go to detect winning rate drop. Hence KG-A2C will have very few advantages than baseline.

##### Game Cloing Expectation
- Baseline will changed to a vanilla A2C, which can only retrain when interacting with real env.
- Hence, the baseline performance will be much worse than the one we used now.
- But at the same time, game cloning might work worse than the copy of env we used now. 
- But with game cloning, we may retrain a long time, however baseline cannot. 
- We can expect KG-A2C with game cloning can work better than baseline.
