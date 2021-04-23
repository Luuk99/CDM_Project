# imports
from datasets import load_dataset

# own imports
from utils import create_dataloader

# set Huggingface logging to error only
import datasets
datasets.logging.set_verbosity_error()


def load_sst2(args, tokenizer):
    """
    Function that loads the SST2 sentiment dataset.
    Inputs:
        args - Namespace object from the argument parser
        tokenizer - BERT tokenizer instance
    Outputs:
        train_set - Training dataset
        dev_set - Development dataset
        test_set - Test dataset
    """

    # load the sst dataset
    dataset = load_dataset("sst")

    # divide into train, dev and test
    train_set = dataset['train']
    dev_set = dataset['validation']
    test_set = dataset['test']

    # function that encodes the sentences
    def encode_sentence(examples):
         return tokenizer(examples['sentence'], truncation=True, padding='max_length')

    # tokenize the datasets
    train_set = train_set.map(encode_sentence, batched=False)
    dev_set = dev_set.map(encode_sentence, batched=False)
    test_set = test_set.map(encode_sentence, batched=False)

    # remove unnecessary columns
    train_set = train_set.remove_columns(['sentence', 'tokens', 'tree'])
    dev_set = dev_set.remove_columns(['sentence', 'tokens', 'tree'])
    test_set = test_set.remove_columns(['sentence', 'tokens', 'tree'])

    # rename the labels
    train_set = train_set.rename_column("label", "labels")
    dev_set = dev_set.rename_column("label", "labels")
    test_set = test_set.rename_column("label", "labels")

    # create dataloaders for the datasets
    train_set = create_dataloader(args, train_set, tokenizer)
    dev_set = create_dataloader(args, dev_set, tokenizer)
    test_set = create_dataloader(args, test_set, tokenizer)

    # return the datasets
    return train_set, dev_set, test_set
