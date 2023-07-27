# Policy-transfer-lifelong-rl
This is the official repo for my undergrad honors thesis at Brown University: [Policy Transfer in Lifelong Reinforcement Learning through Learning Generalizing Features](https://cs.brown.edu/media/filer_public/c2/72/c272a1f8-1186-4a85-8f97-cfe8a1a7278a/zhouzhiyuan_honors_thesis.pdf)

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

## monte experiments
```bash
python3 -m skills.ensemble.train -e experiment_name -s skill_type --start_state room1 --agent ensemble --steps 1000 [--options]  # train
python3 -m skills.ensemble.test -l saving_dir_to_load [--options]  # test
python3 -m skills.ensemble.transfer -l experiment_name_to_load -t a_list_of_targets -s skill_type --agent ensemble -s skill_type  # transfer and meta learn
python3 -m skills.ensemble.transfer --plot -l load -t a_list_of_targets -s skill_type --agent ensemble # plot after transfer experiment
```

## ant environment experiments (not used currently)
```bash
python3 -m skills.baseline.train --agent sac --num_envs 16 --max_steps 10_000_000 --env ant_box
python3 -m skills.baseline.test -l ./results/ant_box/sac
```

## deprecated experiments 
```
bash
# getting baseline performance (DQN)
python3 -m skills.baseline.train --experiment_name debug --agent dqn --env MontezumaRevengeNoFrameskip-v4 [--options]  # train

# training a skill
python3 -m skills.train [--options]

# executing a skill
python3 -m skills.execute --saved_option saving_path [--options]

# control an agent to step through a gym env
python3 -m skills.play [--options]

# control an agent to generate trajectories
python3 -m skills.generate_traj [--options]
```
