# imports
from datasets import load_dataset

# own imports
from utils import create_dataloader

# set Huggingface logging to error only
import datasets
datasets.logging.set_verbosity_error()


def LoadFilteredDataset():
    """
    Function that loads the Circa dataset and filters it.
    Outputs:
        dataset - Filtered Circa dataset
    """

    # load the dataset
    dataset = load_dataset('circa')['train']

    # filter out instances with label 'Other', 'I am not sure..' or 'N/A' (-1)
    dataset = dataset.filter(lambda example: example['goldstandard1'] != dataset.features['goldstandard1'].str2int('Other'))
    dataset = dataset.filter(lambda example: example['goldstandard1'] != dataset.features['goldstandard1'].str2int("I am not sure how X will interpret Y’s answer"))
    dataset = dataset.filter(lambda example: example['goldstandard1'] != -1)

    # return the dataset
    return dataset


def PrepareSets(args, tokenizer, train_set, dev_set, test_set):
    """
    Function that prepares the datasets for usage.
    Inputs:
        args - Namespace object from the argument parser
        tokenizer - BERT tokenizer instance
        train_set - Unprepared training set
        dev_set - Unprepared development set
        test_set - Unprepared test set
    Outputs:
        train_set - Prepared training set
        dev_set - Prepared development set
        test_set - Prepared test set
    """

    # function that encodes the questions and answers using the tokenizer
    def encode_qa(examples):
         return tokenizer('[CLS] ' + examples['question-X'] + ' [SEP] ' + examples['answer-Y'] + ' [SEP]', truncation=True, padding='max_length')

    # function that encodes only the questions using the tokenizer
    def encode_q(examples):
         return tokenizer('[CLS] ' + examples['question-X'] + ' [SEP]', truncation=True, padding='max_length')

    # function that encodes only the answers using the tokenizer
    def encode_a(examples):
         return tokenizer('[CLS] ' + examples['answer-Y'] + ' [SEP]', truncation=True, padding='max_length')

    # check which encoding function to use
    if args.model_version == "QA":
        encode_fn = encode_qa
    elif args.model_version == "A":
        encode_fn = encode_a
    elif args.model_version == 'Q':
        encode_fn = encode_q

    # tokenize the datasets
    train_set = train_set.map(encode_fn, batched=False)
    dev_set = dev_set.map(encode_fn, batched=False)
    test_set = test_set.map(encode_fn, batched=False)

    # check which labels to use
    if args.labels == "strict":
        use_labels = "goldstandard1"
    else:
        use_labels = "goldstandard2"

    # set the labels
    train_set = train_set.rename_column(use_labels, "labels")
    dev_set = dev_set.rename_column(use_labels, "labels")
    test_set = test_set.rename_column(use_labels, "labels")

    # remove unnecessary columns
    if args.labels == 'strict':
        remove_gold_label = 'goldstandard2'
    else:
        remove_gold_label = 'goldstandard1'
    train_set = train_set.remove_columns(['answer-Y', 'canquestion-X', 'context', remove_gold_label, 'judgements', 'question-X'])
    dev_set = dev_set.remove_columns(['answer-Y', 'canquestion-X', 'context', remove_gold_label, 'judgements', 'question-X'])
    test_set = test_set.remove_columns(['answer-Y', 'canquestion-X', 'context', remove_gold_label, 'judgements', 'question-X'])

    # return the prepared datasets
    return train_set, dev_set, test_set


def LoadCircaMatched(args, tokenizer):
    """
    Function to load the Circa dataset for the matched setting.
    Inputs:
        args - Namespace object from the argument parser
        tokenizer - BERT tokenizer instance
    Outputs:
        train_set - Training dataset containing 60% of the data
        dev_set - Development dataset containing 20% of the data
        test_set - Test dataset containing 20% of the data
    """

    # load the filtered dataset
    dataset = LoadFilteredDataset()

    # create the dictionary for the labels
    if args.labels == "strict":
        label_dict = {'Yes': dataset.features['goldstandard1'].str2int('Yes'),
        'Probably yes / sometimes yes': dataset.features['goldstandard1'].str2int('Probably yes / sometimes yes'),
        'Yes, subject to some conditions': dataset.features['goldstandard1'].str2int('Yes, subject to some conditions'),
        'No': dataset.features['goldstandard1'].str2int('No'),
        'Probably no': dataset.features['goldstandard1'].str2int('Probably no'),
        'In the middle, neither yes nor no': dataset.features['goldstandard1'].str2int('In the middle, neither yes nor no')}
    else:
        label_dict = {'Yes': dataset.features['goldstandard2'].str2int('Yes'),
        'No': dataset.features['goldstandard2'].str2int('No'),
        'In the middle, neither yes nor no': dataset.features['goldstandard2'].str2int('In the middle, neither yes nor no'),
        'Yes, subject to some conditions': dataset.features['goldstandard2'].str2int('Yes, subject to some conditions')}

    # split the dataset into train, dev and test
    dataset = dataset.train_test_split(test_size=0.4, train_size=0.6, shuffle=True)
    train_set = dataset['train']
    dataset = dataset['test'].train_test_split(test_size=0.5, train_size=0.5, shuffle=True)
    dev_set = dataset['train']
    test_set = dataset['test']

    # prepare the data
    train_set, dev_set, test_set = PrepareSets(args, tokenizer, train_set, dev_set, test_set)

    # create dataloaders for the datasets
    train_set = create_dataloader(args, train_set, tokenizer)
    dev_set = create_dataloader(args, dev_set, tokenizer)
    test_set = create_dataloader(args, test_set, tokenizer)

    # return the datasets and label dict
    return train_set, dev_set, test_set, label_dict


def LoadCircaUnmatched(args, tokenizer, test_scenario):
    """
    Function to load the Circa dataset for the unmatched setting.
    Inputs:
        args - Namespace object from the argument parser
        tokenizer - BERT tokenizer instance
        test_scenario - Scenario to reserve for testing
    Outputs:
        train_set - Training dataset containing 80% of left scenarios
        dev_set - Development dataset containing 20% of left scenarios
        test_set - Test dataset containing 1 scenario
    """

    # load the filtered dataset
    dataset = LoadFilteredDataset()

    # create the dictionary for the labels
    if args.labels == "strict":
        label_dict = {'Yes': dataset.features['goldstandard1'].str2int('Yes'),
        'Probably yes / sometimes yes': dataset.features['goldstandard1'].str2int('Probably yes / sometimes yes'),
        'Yes, subject to some conditions': dataset.features['goldstandard1'].str2int('Yes, subject to some conditions'),
        'No': dataset.features['goldstandard1'].str2int('No'),
        'Probably no': dataset.features['goldstandard1'].str2int('Probably no'),
        'In the middle, neither yes nor no': dataset.features['goldstandard1'].str2int('In the middle, neither yes nor no')}
    else:
        label_dict = {'Yes': dataset.features['goldstandard2'].str2int('Yes'),
        'No': dataset.features['goldstandard2'].str2int('No'),
        'In the middle, neither yes nor no': dataset.features['goldstandard2'].str2int('In the middle, neither yes nor no'),
        'Yes, subject to some conditions': dataset.features['goldstandard2'].str2int('Yes, subject to some conditions')}

    # create the test, dev and train sets
    test_set = dataset.filter(lambda example: example['context'] == test_scenario)
    left_set = dataset.filter(lambda example: example['context'] != test_scenario)
    left_set = left_set.train_test_split(test_size=0.2, train_size=0.8, shuffle=True)
    train_set = left_set['train']
    dev_set = left_set['test']

    # prepare the data
    train_set, dev_set, test_set = PrepareSets(args, tokenizer, train_set, dev_set, test_set)

    # create dataloaders for the datasets
    train_set = create_dataloader(args, train_set, tokenizer)
    dev_set = create_dataloader(args, dev_set, tokenizer)
    test_set = create_dataloader(args, test_set, tokenizer)

    # return the datasets and label dict
    return train_set, dev_set, test_set, label_dict
