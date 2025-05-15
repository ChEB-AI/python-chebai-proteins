
# 🧪 ChEB-AI Proteins

`python-chebai-proteins` repository for protein prediction and classification, built on top of the [`python-chebai`](https://github.com/ChEB-AI/python-chebai) codebase.


## 🔧 Installation


To install, follow these steps:

1. Clone the repository:
```
git clone https://github.com/ChEB-AI/python-chebai-proteins.git
```

2. Install the package:

```
cd python-chebai
pip install .
```

## 🗂 Recommended Folder Structure

To combine configuration files from both `python-chebai` and `python-chebai-proteins`, structure your project like this:

```
my_projects/
├── python-chebai/
│   ├── chebai/
│   ├── configs/
│   └── ...
└── python-chebai-proteins/
    ├── chebai_proteins/
    ├── configs/
    └── ...
```

This setup enables shared access to data and model configurations.



## 🚀 Training & Pretraining Guide

### ⚠️ Important Setup Instructions

Before running any training scripts, ensure the environment is correctly configured:

* Either:

  * Install the `python-chebai` repository as a package using:

    ```bash
    pip install .
    ```
* **OR**

  * Manually set the `PYTHONPATH` environment variable if working across multiple directories (`python-chebai` and `python-chebai-proteins`):

    * If your current working directory is `python-chebai-proteins`, set:

      ```bash
      export PYTHONPATH=path/to/python-chebai
      ```
      or vice versa.
      
    * If you're working within both repositories simultaneously or facing module not found errors,  we **recommend configuring both directories**:

      ```bash
      # Linux/macOS
      export PYTHONPATH=path/to/python-chebai:path/to/python-chebai-proteins

      # Windows (use semicolon instead of colon)
      set PYTHONPATH=path\to\python-chebai;path\to\python-chebai-proteins
      ```

> 🔎 See the [PYTHONPATH Explained](#-pythonpath-explained) section below for more details.


### 📊 SCOPE hierarchy prediction

Assuming your current working directory is `python-chebai-proteins`, run the following command to start training:
```bash
python -m chebai fit --trainer=../configs/training/default_trainer.yml --trainer.callbacks=../configs/training/default_callbacks.yml --trainer.logger.init_args.name=scope50  --trainer.accumulate_grad_batches=4 --trainer.logger=../configs/training/wandb_logger.yml --trainer.min_epochs=100 --trainer.max_epochs=100 --data=configs/data/scope/scope50.yml --data.init_args.batch_size=32  --data.init_args.num_workers=10 --model=../configs/model/electra.yml --model.train_metrics=../configs/metrics/micro-macro-f1.yml --model.test_metrics=../configs/metrics/micro-macro-f1.yml --model.val_metrics=../configs/metrics/micro-macro-f1.yml  --model.pass_loss_kwargs=false --model.criterion=../configs/loss/bce.yml --model.criterion.init_args.beta=0.99
```

Same command can be used for **DeepGO** just by changing the config path for data.







## 🧭 PYTHONPATH Explained

### What is `PYTHONPATH`?

`PYTHONPATH` is an environment variable that tells Python where to search for modules that aren't installed via `pip` or not in your current working directory.

### Why You Need It

If your config refers to a custom module like:

```yaml
class_path: chebai_proteins.preprocessing.datasets.scope.scope.SCOPe50
```

...and you're running the code from `python-chebai`, Python won't know where to find `chebai_proteins` (from another repo like `python-chebai-proteins/`) unless you add it to `PYTHONPATH`.


### How Python Finds Modules

Python looks for imports in this order:

1. Current directory
2. Standard library
3. Paths in `PYTHONPATH`
4. Installed packages (`site-packages`)

You can inspect the full search paths:

```bash
python -c "import sys; print(sys.path)"
```



### ✅ Setting `PYTHONPATH`

#### 🐧 Linux / macOS

```bash
export PYTHONPATH=/path/to/python-chebai-graph
echo $PYTHONPATH
```

#### 🪟 Windows CMD

```cmd
set PYTHONPATH=C:\path\to\python-chebai-graph
echo %PYTHONPATH%
```

> 💡 Note: This is temporary for your terminal session. To make it permanent, add it to your system environment variables.
