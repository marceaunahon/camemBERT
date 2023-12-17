# camemBERT

## Authors

- Mayar Abdelgawad 
- Maxime Chalumeau
- Marceau Nahon
- Mehdi Noureddine

## Description

This repository contains the code and the data used for the Advanced Machine Learning project on the CamemBERT model. The goal of this project is to reproduce the results of the paper [CamemBERT: a Tasty French Language Model](https://arxiv.org/abs/1911.03894).

## Structure

The repository is structured as follows:

- `oscar.py` contains the code used to create the dataset from the OSCAR corpus. (`oscar_demo.ipynb` contains a demo of the methods used for the dataset)
- `train.py` contains the code used to train the model.
- `model.py` contains the code used to create the Transformer model.
- `train_test.ipynb` contains the code used to train and test the model.
- `Tokenization/` contains the files used to tokenize the data. (the database in text, the vocabulary and the tokenizer model)
- `Trained_models/` contains the different models that we trained during the project.
- `articles/` contains the articles used for the project.
- `Eval/` contains the scripts and metrics used for evaluating the model on different tasks after fine-tuning.
- `pos finetuning/` contains the scripts and datasets used for fine-tuning the model for part-of-speech tagging.

## Requirements

The code was tested on Python 3.11.4 with the following packages:

- torch==2.0.1
- sentencepiece==0.1.99
- tqdm==4.65.0
- datasets==2.12.0
- numpy==1.24.3
- matplotlib==3.7.1

For the POS tagging fine-tuning, the following packages are also required:
- conllu==4.5.3
- transformers==4.29.2

## Usage

### Training

To train the model, follow the instructions in `train_test.ipynb`. You can also use the `train.py` script.
To load a trained model, go to the last cell of `train_test.ipynb` and change the path to the model you want to load.

## Results and Observations

Unfortunately, the results of our model were not as expected. During the training process, we observed that the model was converging to a state where it was repeatedly generating the same token up to the maximum sequence length. This indicates that the model might be stuck in a local minimum during the optimization process, or there could be issues with the learning rate or the data.

We are currently investigating this issue and exploring different strategies to improve the model’s performance. This could involve adjusting the hyperparameters, changing the optimization algorithm, or modifying the model architecture. 