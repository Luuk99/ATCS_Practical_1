# imports
from torchtext import datasets
from torchtext import vocab

# function that loads the SNLI dataset
def load_snli(device, batch_size=64):
    # load the SNLI dataset
    train_iter, dev_iter, test_iter = datasets.SNLI.iters(batch_size=batch_size, device=device)

    # return the datasets
    return train_iter, dev_iter, test_iter


# function that loads the GLOVE embeddings
def load_glove():
    # get the pre-trained embeddings
    embeddings = vocab.GloVe(name='840B', dim=300)

    # return the embeddings
    return embeddings
