# skills
portable skills in Reinforcement Learning

## using the ensemble agent
```bash
python3 -m skills.ensemble.train --experiment_name debug -s skill_type [--options]  # train
python3 -m skills.ensemble.test --tag experiment_name_to_load [--options]  # test
python3 -m skills.ensemble.transfer --load experiment_name_to_load --target a_list_of_targets -s skill_type  # transfer and meta learn
python3 -m skills.ensemble.transfer --plot -l load -t a_list_of_targets -s skill_type  # plot after transfer experiment
```

## getting baseline performance (DQN)
```bash
python3 -m skills.baseline.train --experiment_name debug --agent dqn --env MontezumaRevengeNoFrameskip-v4 [--options]  # train  
```

## training a skill
```shell
python3 -m skills.train [--options]
```

## executing a skill
```shell
python3 -m skills.execute --saved_option saving_path [--options]
```

## control an agent to step through a gym env
```shell
python3 -m skills.play [--options]
```

## control an agent to generate trajectories
```shell
python3 -m skills.generate_traj [--options]
```
