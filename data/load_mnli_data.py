# imports
from datasets import load_dataset

# own imports
from utils import create_dataloader

# set Huggingface logging to error only
import datasets
datasets.logging.set_verbosity_error()


def LoadMNLI(args, tokenizer):
    """
    Function that loads the MultiNLI entailment dataset.
    Inputs:
        args - Namespace object from the argument parser
        tokenizer - BERT tokenizer instance
    Outputs:
        train_set - Training dataset
        dev_set - Development dataset
        test_set - Test dataset
    """

    # load the sst dataset
    dataset = load_dataset("multi_nli")

    # divide into train, dev and test
    train_set = dataset['train']
    dataset = dataset['validation_matched'].train_test_split(test_size=0.5, train_size=0.5, shuffle=True)
    dev_set = dataset['train']
    test_set = dataset['test']

    # function that encodes the sentences
    def encode_sentence(examples):
         return tokenizer(examples['premise'] + ' [SEP] ' + examples['hypothesis'] + ' [SEP]', truncation=True, padding='max_length')

    # tokenize the datasets
    train_set = train_set.map(encode_sentence, batched=False)
    dev_set = dev_set.map(encode_sentence, batched=False)
    test_set = test_set.map(encode_sentence, batched=False)

    # remove unnecessary columns
    train_set = train_set.remove_columns(['promptID', 'pairID', 'premise', 'premise_binary_parse', 'premise_parse',  'hypothesis', 'hypothesis_binary_parse', 'hypothesis_parse', 'genre'])
    dev_set = dev_set.remove_columns(['promptID', 'pairID', 'premise', 'premise_binary_parse', 'premise_parse',  'hypothesis', 'hypothesis_binary_parse', 'hypothesis_parse', 'genre'])
    test_set = test_set.remove_columns(['promptID', 'pairID', 'premise', 'premise_binary_parse', 'premise_parse',  'hypothesis', 'hypothesis_binary_parse', 'hypothesis_parse', 'genre'])

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
