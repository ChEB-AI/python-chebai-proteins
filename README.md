
# 🧪 ChEB-AI Proteins

`python-chebai-proteins` repository for protein prediction and classification, built on top of the [`python-chebai`](https://github.com/ChEB-AI/python-chebai) codebase.


## 🔧 Installation


To install this repository, download [`python-chebai`](https://github.com/ChEB-AI/python-chebai) and this repository, then run

```
cd python-chebai
pip install .

cd python-chebai-proteins
pip install .
```

_Note for developers_: If you want to install the package in editable mode, use the following command instead:

```bash
pip install -e .
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



### 📊 SCOPE hierarchy prediction

Assuming your current working directory is `python-chebai-proteins`, run the following command to start training:
```bash
python -m chebai fit --trainer=../configs/training/default_trainer.yml --trainer.callbacks=../configs/training/default_callbacks.yml --trainer.logger.init_args.name=scope50  --trainer.accumulate_grad_batches=4 --trainer.logger=../configs/training/wandb_logger.yml --trainer.min_epochs=100 --trainer.max_epochs=100 --data=configs/data/scope/scope50.yml --data.init_args.batch_size=32  --data.init_args.num_workers=10 --model=../configs/model/electra.yml --model.train_metrics=../configs/metrics/micro-macro-f1.yml --model.test_metrics=../configs/metrics/micro-macro-f1.yml --model.val_metrics=../configs/metrics/micro-macro-f1.yml  --model.pass_loss_kwargs=false --model.criterion=../configs/loss/bce.yml --model.criterion.init_args.beta=0.99
```

Same command can be used for **DeepGO** just by changing the config path for data.
