# gaitmaster

## Accelerometer-based gait recognition using deep neural networks

### Context

Autoencoder measurements with ZJU-GaitAcc dataset for master's thesis. Loosely based on [gaitgmm](https://github.com/nemesszili/gaitgmm).

### How it works

Obtain the dataset from [here](http://www.cs.zju.edu.cn/~gpan/database/gaitacc.html).
Move the dataset under `data/raw_data` and run [notebooks/RawToCSV.ipynb](notebooks/RawToCSV.ipynb) to obtain the raw features.

Linux:
```batch
conda env create -f environment.yml
source activate gaitmaster
```

Windows:
```batch
conda env create -f environment_win.yml
activate gaitmaster
```

Run a measurement with default settings:
```
python main.py
```

Specify measurement settings with command line arguments. 

| Settings                          | Default          | Description |
|:---------------------------------:|:----------------:|-------------|
| `--feat-ext`                      | `none`           | Type of features used for evaluation (raw/none, dense, lstm, 59) | 
| `--same-day/--cross-day`          | `--same-day`     | Testing data from session 1 **OR** session 2 |
| `--steps`                         | 5                | Number of consecutive cycles used for evaluation (range 1-10) |
| `--identification/--verification` | `--verification` | Evaluation mode |
| `--unreg/--reg`                   | `--unreg`        | Use data from unregistered users **OR** Use negative data from users that the system has already encountered |
| `--loopsteps/--regular`           | `--regular`      | Convenience option to run evaluation for all possible `--steps` options (from 1 to 10) |
| `--epochs`                        | 10               | Number of epochs used for autoencoder training. Ignored if `--feat-ext` is 'none' or '59' |

When in doubt, run:
```
python main.py --help
```

### Example

Run evaluation with the following settings:

| Settings                 | Value         | 
|:------------------------:|:-------------:|
| `--feat-ext`             | 59            |
| `--same-day/--cross-day` | `--cross-day` |
| `--steps`                | 3             |
| `--unreg/--reg`          | `--reg`       |
| `--loopsteps/--regular`  | `--regular`   |

```
python main.py --feat-ext='59' --cross-day --steps=3 --reg
```