# imports
import numpy as np
from datasets import load_dataset

# own imports
from utils import create_dataloader

# set Huggingface logging to error only
import datasets
datasets.logging.set_verbosity_error()


def LoadIQAP(args, tokenizer):
    """
    Function that loads the Indirect Question-Answer Pairs dataset.
    Inputs:
        args - Namespace object from the argument parser
        tokenizer - BERT tokenizer instance
    Outputs:
        train_set - Training dataset
        dev_set - Development dataset
        test_set - Test dataset
    """

    # load the sst dataset
    dataset = load_dataset('csv', data_files='data/local_datasets/iqap/iqap-data.csv')['train']

    # function to convert majority votes to labels
    def assign_label(example):
        label = np.argmax([example['definite-yes'], example['probable-yes'], example['definite-no'], example['probable-no']])
        example['labels'] = label
        return example

    # assign labels to the dataset
    dataset = dataset.map(assign_label, batched=False)

    # divide into train, dev and test
    train_set = dataset.filter(lambda example: example['DevEval'] == 'DEVELOPMENT')
    dataset = dataset.filter(lambda example: example['DevEval'] == 'EVALUATION')
    dataset = dataset.train_test_split(test_size=0.5, train_size=0.5, shuffle=True)
    dev_set = dataset['train']
    test_set = dataset['test']

    # function that encodes the question and passage
    def encode_sentence(examples):
         return tokenizer('[CLS] ' + examples['Question'] + ' [SEP] ' + examples['Answer'] + ' [SEP]', truncation=True, padding='max_length')

    # tokenize the datasets
    train_set = train_set.map(encode_sentence, batched=False)
    dev_set = dev_set.map(encode_sentence, batched=False)
    test_set = test_set.map(encode_sentence, batched=False)

    # remove unnecessary columns
    train_set = train_set.remove_columns(['Answer', 'AnswerParse', 'Classification', 'DevEval', 'Item', 'Prefix', 'Question', 'QuestionParse', 'Source', 'definite-no', 'definite-yes', 'probable-no', 'probable-yes'])
    dev_set = dev_set.remove_columns(['Answer', 'AnswerParse', 'Classification', 'DevEval', 'Item', 'Prefix', 'Question', 'QuestionParse', 'Source', 'definite-no', 'definite-yes', 'probable-no', 'probable-yes'])
    test_set = test_set.remove_columns(['Answer', 'AnswerParse', 'Classification', 'DevEval', 'Item', 'Prefix', 'Question', 'QuestionParse', 'Source', 'definite-no', 'definite-yes', 'probable-no', 'probable-yes'])

    # create dataloaders for the datasets
    train_set = create_dataloader(args, train_set, tokenizer)
    dev_set = create_dataloader(args, dev_set, tokenizer)
    test_set = create_dataloader(args, test_set, tokenizer)

    # return the datasets
    return train_set, dev_set, test_set
