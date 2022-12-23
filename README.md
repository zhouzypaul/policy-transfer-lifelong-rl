# skills
portable skills in Reinforcement Learning

## using the ensemble agent
```bash
python3 -m skills.ensemble.train -e experiment_name -s skill_type --start_state room1 --agent ensemble --steps 1000 [--options]  # train
python3 -m skills.ensemble.test -l saving_dir_to_load [--options]  # test
python3 -m skills.ensemble.transfer -l experiment_name_to_load -t a_list_of_targets -s skill_type --agent ensemble -s skill_type  # transfer and meta learn
python3 -m skills.ensemble.transfer --plot -l load -t a_list_of_targets -s skill_type --agent ensemble # plot after transfer experiment
```

## procgen experiments
```bash
python -m skills.baseline.transfer -e experiment_name --num_levels 20 --transfer_steps 500000 --env ENV --num_policies 3 --seed 0
# fix the attention masks
python -m skills.baseline.transfer -e experiment_name --num_levels 20 --transfer_steps 500000 --env ENV --num_policies 3 --seed 0 --fix_attention_masks --load ./results/saved_experiment
# remove feature learners but keep ensemble policy
python -m skills.baseline.transfer -e experiment_name --num_levels 20 --transfer_steps 500000 --env ENV --num_policies 3 --seed 0 --remove_feature_learners
# plotting
python -m skills.baseline.plot -f [-u] -l /results/path 
```

## ant environment experiments
```bash
python3 -m skills.baseline.train --agent sac --num_envs 16 --max_steps 10_000_000 --env ant_box
python3 -m skills.baseline.test -l ./results/ant_box/sac
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
