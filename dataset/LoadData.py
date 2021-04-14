# imports
from torchtext import datasets
from torchtext import vocab
from torchtext.data import Field, BucketIterator

# function that loads the SNLI dataset
def load_snli(device=None, batch_size=64, development=False, return_label_vocab=False):
    """
    Inputs:
        device - Torchtext device to use. Default is None (use CUDA)
        batch_size - Size of the batches. Default is 64
        development - Whether to use development dataset. Default is False
        return_label_vocab - Whether to return the label vocab. Default is False
    Outputs:
        vocab - GloVe embedding vocabulary from the alignment
        train_iter - BucketIterator of training batches
        dev_iter - BucketIterator of validation/development batches
        test_iter - BucketIterator of test batches
    """

    # load the glove embeddings
    glove_embeddings = load_glove()

    # check whether to cut-off the datasets or not
    if development:
        data_root = '.development_data'
    else:
        data_root = '.data'

    # load the SNLI dataset
    text_field = Field(tokenize='spacy', lower=True, batch_first=True)
    label_field = Field(sequential=False, batch_first=True, is_target=True)
    train_dataset, dev_dataset, test_dataset = datasets.SNLI.splits(text_field=text_field,
                                                                    label_field=label_field,
                                                                    root=data_root)

    # create the vocab
    text_field.build_vocab(train_dataset, vectors=glove_embeddings)
    label_field.build_vocab(train_dataset, specials_first=False)
    vocab = text_field.vocab

    # create batch iterators for the datasets
    train_iter, dev_iter, test_iter = BucketIterator.splits(
        (train_dataset, dev_dataset, test_dataset),
        batch_sizes=(batch_size, batch_size, batch_size),
        device=device)

    # return the vocab and datasets
    if return_label_vocab:
        label_vocab = label_field.vocab
        return vocab, label_vocab, train_iter, dev_iter, test_iter
    return vocab, train_iter, dev_iter, test_iter


# function that loads the GLOVE embeddings
def load_glove():
    """
    Outputs:
        embeddings - GloVe 840B embeddings with dimension 300
    """

    # get the pre-trained embeddings
    embeddings = vocab.GloVe(name='840B', dim=300)

    # return the embeddings
    return embeddings
