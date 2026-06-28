# ChEB-AI Proteins

`python-chebai-proteins` is an extension package for
[`python-chebai`](https://github.com/ChEB-AI/python-chebai) that adapts the
ChEB-AI training and preprocessing stack to protein sequence data.

The repository provides dataset classes, readers, migration utilities, losses,
metrics, and example configs for protein classification experiments. Its main
focus is hierarchical, multi-label protein classification from amino-acid
sequences, with support for both tokenized sequence models and ESM2 protein
embeddings.

## Table of Contents

- [What This Repository Does](#what-this-repository-does)
- [Main Tasks](#main-tasks)
  - [Protein Function Prediction With Gene Ontology Labels](#protein-function-prediction-with-gene-ontology-labels)
  - [Protein Structural Classification With SCOPe Labels](#protein-structural-classification-with-scope-labels)
  - [Swiss-Prot Sequence Pretraining Data](#swiss-prot-sequence-pretraining-data)
- [Data Sources](#data-sources)
- [Protein Representations](#protein-representations)
- [Dataset Classes and Configs](#dataset-classes-and-configs)
- [Installation](#installation)
- [Training](#training)
- [Local Data Layout](#local-data-layout)
- [DeepGO Data Migration](#deepgo-data-migration)
- [Development](#development)
- [License](#license)

## What This Repository Does

This package turns public protein resources into the format expected by the
`chebai` training pipeline. In practice, it can:

- download and preprocess protein sequence datasets;
- parse ontology or classification hierarchies into directed graphs;
- select prediction targets according to dataset-specific thresholds;
- encode protein sequences as amino-acid tokens, n-grams, or ESM2 embeddings;
- build multi-hot label vectors for hierarchical protein classification;
- create dynamic train, validation, and test splits;
- load migrated DeepGO and DeepGO2 datasets into the same local data format;
- provide configuration files for model training with the `chebai` CLI.

The code is intentionally dataset-oriented: most of the project lives under
`chebai_proteins/preprocessing`, where raw biological resources are converted
into processed `data.pkl`, encoded `data.pt`, split files, and class lists.

## Main Tasks

### Protein Function Prediction With Gene Ontology Labels

The `deepGO` dataset classes build Gene Ontology (GO) prediction datasets from
Swiss-Prot proteins. Each protein is treated as one sample. The input is the
protein's amino-acid sequence, and the targets are GO terms.

This is a hierarchical multi-label classification task:

- one protein may have many GO labels;
- GO annotations are propagated through the ontology ancestors;
- the final label vector is multi-hot;
- classes are GO terms selected by annotation-frequency thresholds;
- experiments can target one GO namespace or all namespaces.

The supported GO branches are:

- `BP`: biological process;
- `MF`: molecular function;
- `CC`: cellular component;
- `all`: all three branches together.

For the generated GO-UniProt datasets, labels are selected with thresholded
classes such as:

- `GOUniProtOver50`: GO terms with at least 50 protein annotations;
- `GOUniProtOver250`: GO terms with at least 250 protein annotations.

Only Swiss-Prot records with reviewed protein entries are used. The parser keeps
GO annotations with experimental evidence codes such as `EXP`, `IDA`, `IPI`,
`IMP`, `IGI`, `IEP`, `TAS`, `IC`, and the high-throughput evidence codes used
by newer DeepGO2-style data. Proteins with no valid GO labels are not used for
the supervised GO task.

### Protein Structural Classification With SCOPe Labels

The `scope` dataset classes build classification datasets from SCOPe
(Structural Classification of Proteins - extended) and PDB sequence data. Here,
the samples are PDB chain sequences, and the labels are selected SCOPe hierarchy
classes.

This is also a hierarchical multi-label classification task:

- a PDB chain sequence can correspond to one or more SCOPe domains;
- each domain belongs to multiple hierarchy levels;
- labels are selected SCOPe SUN IDs at levels such as class, fold,
  superfamily, family, protein, and species;
- final targets are multi-hot vectors over selected hierarchy nodes.

The SCOPe hierarchy levels handled by the code are:

- `cl`: class;
- `cf`: fold;
- `sf`: superfamily;
- `fa`: family;
- `dm`: protein;
- `sp`: species;
- `px`: domain.

The included configs use SCOPe version `2.08` and target sets such as
`SCOPeOver50` and `SCOPeOver2000`, where the number indicates how many sequence
successors a hierarchy node must have to become a prediction class.

### Swiss-Prot Sequence Pretraining Data

The `SwissProteinPretrain` dataset prepares Swiss-Prot protein sequences for
pretraining-style experiments. It selects reviewed Swiss-Prot proteins that:

- have a sequence;
- are within the configured maximum sequence length;
- do not contain disallowed ambiguous amino-acid symbols;
- do not have a valid experimentally supported GO annotation.

This dataset yields protein sequences without supervised labels. It is useful
when pretraining a sequence encoder before supervised GO or SCOPe
classification.

## Data Sources

The repository works with three main biological data sources.

### Swiss-Prot / UniProtKB

Swiss-Prot supplies reviewed protein records, protein accessions, raw amino-acid
sequences, and GO cross references. The GO-UniProt and pretraining datasets use
the Swiss-Prot flat file:

```text
https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/complete/uniprot_sprot.dat.gz
```

The processed GO-UniProt table has this conceptual shape:

| Column | Meaning |
| --- | --- |
| `swiss_id` | Swiss-Prot entry name |
| `accession` | one or more UniProt accessions |
| `go_ids` | direct and propagated GO term IDs |
| `sequence` | amino-acid sequence |
| label columns | one boolean column per selected GO class |

### Gene Ontology

GO supplies the ontology graph used for function labels and ancestor
propagation. The code downloads `go-basic.obo`:

```text
https://purl.obolibrary.org/obo/go/go-basic.obo
```

The parser keeps non-obsolete GO terms and can restrict terms to `BP`, `MF`,
`CC`, or include all branches. GO term IDs are normalized internally from values
such as `GO:0003674` to integer IDs such as `3674`.

### SCOPe and PDB Sequences

SCOPe supplies protein structural classification metadata, and RCSB PDB supplies
chain-level sequences. The SCOPe preprocessing uses:

- SCOPe `CLA`, `HIE`, and `DES` parse files;
- PDB chain sequences from `pdb_seqres.txt.gz`.

The processed SCOPe table has this conceptual shape:

| Column | Meaning |
| --- | --- |
| `id` | generated sequence identifier |
| `sids` | SCOPe domain IDs associated with the sequence |
| `sequence` | PDB chain amino-acid sequence |
| label columns | one boolean column per selected SCOPe hierarchy node |

## Protein Representations

The repository currently supports two representation styles.

### Tokenized Amino-Acid Sequences

`ProteinDataReader` converts a raw sequence such as `MKTFF...` into integer
token IDs. By default it tokenizes at the single amino-acid level. It can also
tokenize into fixed-size n-grams by passing `n_gram`.

Allowed symbols are the standard amino-acid letters used by the project plus
`X`, which is treated as a valid unknown or masked amino-acid symbol. Some data
pipelines filter ambiguous amino acids such as `B`, `O`, `J`, `U`, `Z`, and
`*`; the DeepGO2 migration path instead replaces invalid symbols with `X`.

Token vocabularies are stored under:

```text
chebai_proteins/preprocessing/bin/protein_token/tokens.txt
chebai_proteins/preprocessing/bin/protein_token_3_gram/tokens.txt
```

### ESM2 Embeddings

`ESM2EmbeddingReader` can convert sequences into mean-pooled ESM2
representations. It loads pretrained ESM2 weights from a local cache when
available, otherwise from the FAIR ESM model URLs.

The default reader settings mirror DeepGO2-style usage:

- model: `esm2_t36_3B_UR50D`;
- representation layer: `36`;
- truncation length: `1022`;
- mean pooling over residue representations.

For debugging, the reader docstring recommends smaller ESM2 models such as
`esm2_t6_8M_UR50D` or `esm2_t12_35M_UR50D`.

## Dataset Classes and Configs

Important dataset classes:

| Class | Purpose |
| --- | --- |
| `GOUniProtOver50` | generated GO-UniProt dataset with GO terms annotated at least 50 times |
| `GOUniProtOver250` | generated GO-UniProt dataset with GO terms annotated at least 250 times |
| `DeepGO1MigratedData` | loads data migrated from the original DeepGO format |
| `DeepGO2MigratedData` | loads data migrated from DeepGO2-style data, optionally using stored ESM2 embeddings |
| `SwissProteinPretrain` | unlabeled Swiss-Prot sequence data for pretraining |
| `SCOPeOver50` | SCOPe dataset with hierarchy nodes having at least 50 sequence successors |
| `SCOPeOver2000` | SCOPe dataset with hierarchy nodes having at least 2000 sequence successors |
| `SCOPeOverPartial2000` | SCOPe subset under a chosen top class, using the 2000 threshold |

Example config files:

```text
configs/data/deepGO/go50.yml
configs/data/deepGO/go250.yml
configs/data/deepGO/deepgo_1_migrated_data.yml
configs/data/deepGO/deepgo_2_migrated_data.yml
configs/data/deepGO/deepgo2_esm2.yml
configs/data/scope/scope50.yml
configs/data/scope/scope50_esm.yml
configs/data/scope/scope2000.yml
configs/model/electra.yml
configs/loss/BCEWithLogitsLoss.yml
configs/metrics/MultilabelAUROC.yml
```

## Installation

This repository depends on `python-chebai`. A typical local development setup
keeps both repositories next to each other:

```text
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

Install both packages:

```bash
cd python-chebai
pip install .

cd ../python-chebai-proteins
pip install .
```

For editable development:

```bash
cd python-chebai-proteins
pip install -e .
```

Optional extras:

```bash
pip install -e ".[dev]"
pip install -e ".[plot]"
pip install -e ".[wandb]"
```

## Training

Training is launched through the `chebai` CLI from the parent project. The
example below trains an Electra-style model on the SCOPe 50-threshold dataset:

```bash
python -m chebai fit \
  --trainer=../configs/training/default_trainer.yml \
  --trainer.callbacks=../configs/training/default_callbacks.yml \
  --trainer.logger.init_args.name=scope50 \
  --trainer.accumulate_grad_batches=4 \
  --trainer.logger=../configs/training/wandb_logger.yml \
  --trainer.min_epochs=100 \
  --trainer.max_epochs=100 \
  --data=configs/data/scope/scope50.yml \
  --data.init_args.batch_size=32 \
  --data.init_args.num_workers=10 \
  --model=../configs/model/electra.yml \
  --model.train_metrics=../configs/metrics/micro-macro-f1.yml \
  --model.test_metrics=../configs/metrics/micro-macro-f1.yml \
  --model.val_metrics=../configs/metrics/micro-macro-f1.yml \
  --model.pass_loss_kwargs=false \
  --model.criterion=../configs/loss/bce.yml \
  --model.criterion.init_args.beta=0.99
```

To switch from SCOPe to a GO/DeepGO dataset, change the `--data` config, for
example:

```bash
--data=configs/data/deepGO/deepgo_2_migrated_data.yml
```

## Local Data Layout

Generated data is written under `data/` by the dataset classes. Important
locations include:

```text
data/GO_UniProt/
data/GO_UniProt/Pretraining/
data/SCOPe/version_<version>/
data/esm2_reader/
```

Typical generated artifacts include:

- raw downloaded files such as `go-basic.obo`, `uniprot_sprot.dat`,
  `cla.txt`, `hie.txt`, `des.txt`, and `pdb_sequences.txt`;
- `data.pkl`, a processed pandas table with readable features and boolean
  labels;
- `data.pt`, the encoded data consumed by the training pipeline;
- `classes.txt`, the selected class list;
- split files generated by the dynamic dataset machinery.

These data files can be large and are intentionally not part of the source tree.

## DeepGO Data Migration

The repository includes migration helpers for existing DeepGO and DeepGO2 data
formats:

```text
chebai_proteins/preprocessing/migration/deep_go/migrate_deep_go_1_data.py
chebai_proteins/preprocessing/migration/deep_go/migrate_deep_go_2_data.py
```

The migrated classes expect processed files to already exist in the target data
directory. If they are missing, the dataset class raises a `FileNotFoundError`
with a reminder to run the appropriate migration script first.

DeepGO1 migration:

- consumes train/test pickles and term files from the original DeepGO format;
- creates a validation split from the training split;
- preserves split assignments;
- writes GO labels into the local GO-UniProt-style layout.

DeepGO2 migration:

- consumes DeepGO2-style train, validation, and test pickles;
- truncates sequences to the configured maximum length, usually `1000`;
- replaces invalid amino-acid symbols with `X`;
- preserves stored ESM2 embeddings when available;
- writes data in the format consumed by `DeepGO2MigratedData`.

## Development

Run the unit tests with:

```bash
pytest
```

The tests cover the core reader behavior, GO ontology parsing, Swiss-Prot to GO
mapping, GO class selection thresholds, SCOPe preprocessing, and Swiss-Prot
pretraining data selection.

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

## License

This project is licensed under the AGPL-3.0 license. See `LICENSE` for details.
