# imports
from torchtext import datasets
from torchtext import vocab
from torchtext.data import Field, BucketIterator

# function that loads the SNLI dataset
def load_snli(device, batch_size=64):
    # load the glove embeddings
    glove_embeddings = load_glove()

    # load the SNLI dataset
    text_field = Field(tokenize='spacy', lower=True, batch_first=True)
    label_field = Field(sequential=False, batch_first=True, is_target=True)
    train_dataset, dev_dataset, test_dataset = datasets.SNLI.splits(text_field=text_field,
                                                                    label_field=label_field)

    # create the vocab
    text_field.build_vocab(train_dataset, vectors=glove_embeddings)
    label_field.build_vocab(train_dataset)

    # create batch iterators for the datasets
    train_iter, dev_iter, test_iter = BucketIterator.splits(
        (train_dataset, dev_dataset, test_dataset),
        batch_sizes=(batch_size, batch_size, batch_size),
        device=device)

    # return the datasets
    return train_iter, dev_iter, test_iter


# function that loads the GLOVE embeddings
def load_glove():
    # get the pre-trained embeddings
    embeddings = vocab.GloVe(name='840B', dim=300)

    # return the embeddings
    return embeddings
