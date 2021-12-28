## Tutorial 1: 

### Init git and DVC

1. `git init <project-folder>`
2. `git remote add <name> <remote-url>`
3. `dvc init`
4. `dvc remote add -d <name> <remote-url>`

### DVC run commands

1. `dvc run -n create-dataset -d scripts/create_dataset.py -o assets/data python scripts/create_dataset.py`
2. `dvc run -n extract-feature -d scripts/extract_features.py -d assets/data -o assets/features python scripts/extract_features.py`
3. `dvc run -n train-model -d assets/features/train_features.csv -d assets/features/train_labels.csv -o assets/models python scripts/train_model.py`
4. `dvc run -n eval-model -d assets/features/test_features.csv -d assets/features/test_labels.csv -d assets/models/model.pk -d scripts/eval_model.py  -o assets/metrics.json python scripts/eval_model.py`

### Experiment tag and push to remote git repo

1. ` git tag -a "xx-experiment" -m "Experiment with xxx"`
2. `git push origin xx-experiment`

### DVC Data and Model push to remote storage

`dvc push`

### Experiment repro

1. `git checkout xx-experiment`
2. `dvc repro`

