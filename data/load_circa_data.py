# imports
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# own imports
from utils import create_dataloader
from data.annotate_circa_data import *

# set Huggingface logging to error only
import datasets
datasets.logging.set_verbosity_error()

def processCircaDataset(doAnnotateImportantWords = False, preloadImportantWords = True,
                          doAnnotateTopics = False, preloadTopics = True, traverseTopicLemmas = True):
    """
    Function that processes the Circa dataset by filtering and annotating it.
    
    Outputs:
        dataset - Filtered/annotated Circa dataset
    """

    # load the dataset
    dataset = load_dataset('circa')['train']
    
    if doAnnotateImportantWords:
        mostImportantWords = annotateImportantWords(dataset, preload = preloadImportantWords)
        dataset = dataset.add_column('most_important_word_in_answer', mostImportantWords)
        
        # nested because important word must be present
        if doAnnotateTopics:
            topicsForAnswer = annotateWordNetTopics(dataset, preload = preloadTopics, traverseAll = traverseTopicLemmas)
            dataset = dataset.add_column('topic', topicsForAnswer)
    
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
    dataset = processCircaDataset(doAnnotateImportantWords = args.impwords, preloadImportantWords = args.npimpwords,
                                    doAnnotateTopics = args.topics, preloadTopics = args.nptopics, traverseTopicLemmas = args.traversetopics)

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

    # return the datasets
    return train_set, dev_set, test_set


def LoadCircaUnmatched(args, tokenizer, test_scenario, dev_scenario):
    """
    Function to load the Circa dataset for the unmatched setting.
    Inputs:
        args - Namespace object from the argument parser
        tokenizer - BERT tokenizer instance
        test_scenario - Scenario to reserve for testing
        dev_scenario - Scenario to reserve for development
    Outputs:
        train_set - Training dataset containing 8 scenarios
        dev_set - Development dataset containing 1 scenario
        test_set - Test dataset containing 1 scenario
    """

    # load the filtered dataset
    dataset = processCircaDataset(doAnnotateImportantWords = args.impwords, preloadImportantWords = args.npimpwords,
                                    doAnnotateTopics = args.topics, preloadTopics = args.nptopics, traverseTopicLemmas = args.traversetopics)

    # create the test, dev and train sets
    test_set = dataset.filter(lambda example: example['context'] == test_scenario)
    dev_set = dataset.filter(lambda example: example['context'] == dev_scenario)
    train_set = dataset.filter(lambda example: example['context'] != test_scenario)
    train_set = train_set.filter(lambda example: example['context'] != dev_scenario)

    # prepare the data
    train_set, dev_set, test_set = PrepareSets(args, tokenizer, train_set, dev_set, test_set)

    # create dataloaders for the datasets
    train_set = create_dataloader(args, train_set, tokenizer)
    dev_set = create_dataloader(args, dev_set, tokenizer)
    test_set = create_dataloader(args, test_set, tokenizer)

    # return the datasets
    return train_set, dev_set, test_set