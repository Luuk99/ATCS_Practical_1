# ATCS_Practical_1
Practical 1 of Advanced Topics in Computational Semantics (first year master AI @ UvA).

## Content
In this project, we test multiple models proposed by [Conneau et al.](https://arxiv.org/pdf/1705.02364.pdf). The following models are considered:
* Baseline: averaging word embeddings to obtain sentence representations.
* Unidirectional LSTM applied on the word embeddings, where the last hidden state is considered as sentence representation.
* Simple bidirectional LSTM (BiLSTM), where the last hidden state of forward and backward layers are concatenated as sentence representations.
* BiLSTM with max pooling applied to the concatenation of word-level hidden states from both directions to retrieve sentence representations.

## Prerequisites
* Anaconda. Available at: https://www.anaconda.com/distribution/

## Getting Started
1. Open Anaconda prompt and clone this repository (or download and unpack zip):
```bash
git clone https://github.com/Luuk99/ATCS_Practical_1
```
2. Create the environment:
```bash
conda env create -f environment.yml
```
3. Activate the environment:
```bash
conda activate ATCS
```
4. View the notebook with the experimental results:
```bash
jupyter notebook results.ipynb
```

## Replicating Results
Training a model:
1. Do step 1-3 of the above section.
2. Download *en* from spacy for the tokenizer:
```bash
python -m spacy download en
```
3. Create a *.data* folder inside the root folder and place the SNLI data from the [SNLI website](https://nlp.stanford.edu/projects/snli/) in this folder.
4. Run the training of the models:
```bash
python main.py --model MODEL 
```

Running SentEval:
1. Clone the SentEval project:
```bash
git clone https://github.com/facebookresearch/SentEval.git
```
2. Navigate to the *SentEval* folder.
3. Install SentEval
```bash
python setup.py install
```
4. Open GitBash, navigate to the *data/downstream* folder and download data:
```bash
get_transfer_data.bash
```
5. Download Glove embeddings from the [Stanford website](http://nlp.stanford.edu/data/glove.840B.300d.zip).
6. Move the .zip file to the *SentEval/pretrained* folder and unzip here. (make sure the .txt file is in the *pretrained* folder directly)
7. Move the entire *SentEval* folder inside the *ATCS_Practical_1* folder.
8. Run SentEval from the *ATCS_Practical_1* folder:
```bash
python senteval.py --model MODEL
```

## Tips
* If you want to make use of the *--development* feature to run on a smaller dataset when making changes:
	1. Create a folder *.development_data* in the root folder.
	2. Copy the SNLI dataset from *.data* to *.development_data*.
	3. Limit the *.json* files to your taste. Since I used 64 as batch size, I use the following limits:
		* 64x400 for train
		* 64x100 for dev
		* 64x100 for test
* Add the *--progress_bar* argument to the training to see the training progress.
* If you want to use a checkpoint, use the *--checkpoint_dir* argument and provide the path to the checkpoint file. (add the *.ckpt* file at the end of the path)
* Use our trained models instead of training yourself (can take very long).
	1. Download the models from this [Drive folder](https://drive.google.com/drive/folders/1x2S5c_8n_zvXk1rXJ004_JhXAY3ldLmk?usp=sharing).
	2. Move the individual model folders inside your *pl_logs/lightning_logs/* folder.

## Using Lisa Cluster
* Use the *enviroment_Lisa.yml* file to create the correct environment.
* NO need to download *en* from spacy, this is done in the *.job* files.
* Run the provided *.job* files for the different models.
* If you alter the *.job* files, do keep in mind to not use *--progress_bar* as argument. This does not fare well on Lisa.

## Arguments
The models can be trained with the following command line arguments:
```bash
usage: main.py [-h] [--model MODEL] [--lr LR] [--lr_decay LR_DECAY]
		    [--lr_decrease_factor LR_DECREASE_FACTOR] [--lr_threshold LR_THRESHOLD] 
		    [--batch_size BATCH_SIZE] [--checkpoint_dir CHECKPOINT_DIR]
		    [--seed SEED] [--log_dir LOG_DIR] [--progress_bar] [--development]

optional arguments:
  -h, --help            			Show help message and exit.
  --model MODEL					What model to use. Options: ['AWE', 'UniLSTM', 'BiLSTM', 'BiLSTMMax']. Default is 'AWE'.
  --lr LR					Learning rate to use. Default is 0.1.
  --lr_decay LR_DECAY				Learning rate decay after each epoch. Default is 0.99.
  --lr_decrease_factor LR_DECREASE_FACTOR	Factor to divide learning rate by when dev accuracy decreases. Default is 5.
  --lr_threshold LR_THRESHOLD			Learning rate threshold to stop at. Default is 10e-5.
  --batch_size BATCH_SIZE			Minibatch size. Default is 64.
  --checkpoint_dir CHECKPOINT_DIR		Directory where the pretrained model checkpoint is located. Default is None (no checkpoint used).
  --seed SEED					Seed to use for reproducing results. Default is 1234.
  --log_dir LOG_DIR				Directory where the PyTorch Lightning logs should be created. Default is 'pl_logs'.
  --progress_bar				Use a progress bar indicator for interactive experimentation. Not to be used in conjuction with SLURM jobs.
  --development					Limit the size of the datasets in development.
```

## Authors
* Luuk Kaandorp - luuk.kaandorp@student.uva.nl

## Acknowledgements
* SentEval was cloned from the [original project GitHub](https://github.com/facebookresearch/SentEval).
* Pytorch Lightning implementation was developed using information available in the Deep Learning Course of the UvA (https://uvadlc.github.io/).
