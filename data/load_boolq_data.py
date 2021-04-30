# imports
from datasets import load_dataset

# own imports
from utils import create_dataloader

# set Huggingface logging to error only
import datasets
datasets.logging.set_verbosity_error()


def LoadBoolQ(args, tokenizer):
    """
    Function that loads the BoolQ question-answering dataset.
    Inputs:
        args - Namespace object from the argument parser
        tokenizer - BERT tokenizer instance
    Outputs:
        train_set - Training dataset
        dev_set - Development dataset
        test_set - Test dataset
    """

    # load the sst dataset
    dataset = load_dataset("boolq")

    # divide into train, dev and test
    train_set = dataset['train']
    dataset = dataset['validation'].train_test_split(test_size=0.5, train_size=0.5, shuffle=True)
    dev_set = dataset['train']
    test_set = dataset['test']

    # function that encodes the question and passage
    def encode_sentence(examples):
         return tokenizer('[CLS] ' + examples['question'] + ' [SEP] ' + examples['passage'] + ' [SEP]', truncation=True, padding='max_length')

    # tokenize the datasets
    train_set = train_set.map(encode_sentence, batched=False)
    dev_set = dev_set.map(encode_sentence, batched=False)
    test_set = test_set.map(encode_sentence, batched=False)

    # function to convert the answers to 0 (false) and 1 (true)
    def change_label(example):
        example['labels'] = 0 if (example['answer']) else 1
        return example

    # convert answers to labels
    train_set = train_set.map(change_label, batched=False)
    dev_set = dev_set.map(change_label, batched=False)
    test_set = test_set.map(change_label, batched=False)

    # remove unnecessary columns
    train_set = train_set.remove_columns(['question', 'passage', 'answer'])
    dev_set = dev_set.remove_columns(['question', 'passage', 'answer'])
    test_set = test_set.remove_columns(['question', 'passage', 'answer'])

    # create dataloaders for the datasets
    train_set = create_dataloader(args, train_set, tokenizer)
    dev_set = create_dataloader(args, dev_set, tokenizer)
    test_set = create_dataloader(args, test_set, tokenizer)

    # return the datasets
    return train_set, dev_set, test_set
