# imports
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# own imports
from utils import create_dataloader
from data.annotate_circa_data import *

# set Huggingface logging to error only
import datasets
datasets.logging.set_verbosity_error()

def processCircaDataset(doAnnotateImportantWords = False, preloadImportantWords = True, doAnnotateTopics = False, preloadTopics = True, 
                        traverseTopicLemmas = True, tfidf = False, hybrid = False, topic_depth = None, label_density = None, impwordsfile = None, topicsfile = None, topiclabelsfile = None):
    """
    Function that processes the Circa dataset by filtering and annotating it.
    
    Inputs:
        doAnnotateImportantWords - if True, most important words will be extracted from answers where possible
        preloadImportantWords    - if True, these important words will be pre-loaded from an existing file
        impwordsfile             - filename of saved important words, if None, default filename (specified in annotate_circa_data.py) will be used
        tfidf                    - if True, noun with highest TF-IDF scores will be considered as the most important word
                                   if False, last noun of the sentence will be considered the most important
        hybrid                   - if True, most important words will be determined by TF-IDF values ONLY if there is no last noun
        doAnnotateTopics         - if True, a column will be added to the dataset with topic labels inferred from every answer
        preloadTopics            - if True, these topics will be pre-loaded from an existing file
        topicsfile               - filename of saved topics, default filename (specified in annotate_circa_data.py) will be used
        topicslabelsfile         - filename of saved topic labels, default filename (specified in annotate_circa_data.py) will be used
        traverseTopicLemmas      - if True, the whole synset will be traversed for the most important word in the answer, and all lemmas extracted
                                   if False, the first lemma at a certain top-down tree-depth level will be taken (the most left synonym in the tree will always be taken)
        topic_depth              - top-down depth level (starting at 0 from root node) at which topic is sampled if traverseTopicLemmas is False
        label_density            - Controls the number of allowed topic class labels (>= total topic classes)
    Outputs:
        dataset - Filtered/annotated Circa dataset
    """

    # load the dataset
    dataset = load_dataset('circa')['train']
    
    usedTopicLabels = None
    
    if doAnnotateImportantWords:
        mostImportantWords = annotateImportantWords(dataset, preload = preloadImportantWords, context = tfidf, hybrid = hybrid, annotationFileName = impwordsfile)
        #dataset = dataset.add_column('most_important_word_in_answer', mostImportantWords)
        
        # nested because important word must be present
        if doAnnotateTopics:
            topicsForAnswer, labelsForAnswer, usedTopicLabels = annotateWordNetTopics(dataset, importantWordsColumn = mostImportantWords, preload = preloadTopics, traverseAll = traverseTopicLemmas,
                                                                     topic_depth = topic_depth, label_density = label_density, annotationFileName = topicsfile, annotationLabelsFileName = topiclabelsfile)
            #dataset = dataset.add_column('topic_of_most_important_word', topicsForAnswer)
            dataset = dataset.add_column('topic_label', labelsForAnswer)
    
    # filter out instances with label 'Other', 'I am not sure..' or 'N/A' (-1)
    dataset = dataset.filter(lambda example: example['goldstandard1'] != dataset.features['goldstandard1'].str2int('Other'))
    dataset = dataset.filter(lambda example: example['goldstandard1'] != dataset.features['goldstandard1'].str2int("I am not sure how X will interpret Y’s answer"))
    dataset = dataset.filter(lambda example: example['goldstandard1'] != -1)

    # return the dataset
    return dataset, usedTopicLabels

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
    train_set = train_set.remove_columns(['canquestion-X', 'answer-Y', 'context', remove_gold_label, 'judgements', 'question-X'])
    dev_set = dev_set.remove_columns(['canquestion-X', 'answer-Y', 'context', remove_gold_label, 'judgements', 'question-X'])
    test_set = test_set.remove_columns(['canquestion-X', 'answer-Y', 'context', remove_gold_label, 'judgements', 'question-X'])

    # return the prepared datasets
    return train_set, dev_set, test_set

def LoadCirca(args, tokenizer, test_scenario = None):
    """
    Function to load the Circa dataset for both matched and unmatched settings.
    Inputs:
        args - Namespace object from the argument parser
        tokenizer - BERT tokenizer instance
        test_scenario - Scenario to reserve for testing (only if UNMATCHED! Otherwise MATCHED settings are loaded)
    Outputs:
        train_set - Training dataset containing 60% of the data if matched; 80% of left scenarios if unmatched
        dev_set - Development dataset containing 20% of the data if matched; 20% of left scenarios if unmatched
        test_set - Test dataset containing 20% of the data if matched; 1 scenario of unmatched
    """
    
    # load the filtered and annotated dataset
    dataset, usedTopicLabels = processCircaDataset(doAnnotateImportantWords = args.impwords,
                                                    preloadImportantWords = args.npimpwords,
                                                    doAnnotateTopics = args.topics,
                                                    preloadTopics = args.nptopics,
                                                    traverseTopicLemmas = args.traversetopics,
                                                    tfidf = args.tfidf, hybrid = args.hybrid,
                                                    topic_depth = args.topic_depth,
                                                    label_density = args.label_density,
                                                    impwordsfile = args.impwordsfile,
                                                    topicsfile = args.topicsfile,
                                                    topiclabelsfile = args.topiclabelsfile)
    
    # create the dictionary for the labels
    if args.labels == "strict":
        circa_labels = {'Yes': dataset.features['goldstandard1'].str2int('Yes'),
        'Probably yes / sometimes yes': dataset.features['goldstandard1'].str2int('Probably yes / sometimes yes'),
        'Yes, subject to some conditions': dataset.features['goldstandard1'].str2int('Yes, subject to some conditions'),
        'No': dataset.features['goldstandard1'].str2int('No'),
        'Probably no': dataset.features['goldstandard1'].str2int('Probably no'),
        'In the middle, neither yes nor no': dataset.features['goldstandard1'].str2int('In the middle, neither yes nor no')}
    else:
        circa_labels = {'Yes': dataset.features['goldstandard2'].str2int('Yes'),
        'No': dataset.features['goldstandard2'].str2int('No'),
        'In the middle, neither yes nor no': dataset.features['goldstandard2'].str2int('In the middle, neither yes nor no'),
        'Yes, subject to some conditions': dataset.features['goldstandard2'].str2int('Yes, subject to some conditions')}
        
    # split the dataset into train, dev and test
    if not test_scenario: # matched
        dataset = dataset.train_test_split(test_size=0.4, train_size=0.6, shuffle=True)
        train_set = dataset['train']
        dataset = dataset['test'].train_test_split(test_size=0.5, train_size=0.5, shuffle=True)
        dev_set = dataset['train']
        test_set = dataset['test']
    else: # unmatched
        test_set = dataset.filter(lambda example: example['context'] == test_scenario)
        left_set = dataset.filter(lambda example: example['context'] != test_scenario)
        left_set = left_set.train_test_split(test_size=0.2, train_size=0.8, shuffle=True)
        train_set = left_set['train']
        dev_set = left_set['test']
        
    # prepare the data
    train_set_o, dev_set_o, test_set_o = PrepareSets(args, tokenizer, train_set, dev_set, test_set)

    # create dataloaders for the datasets
    train_set_dict = {'Circa': create_dataloader(args, train_set_o, tokenizer)}
    dev_set_dict = {'Circa': create_dataloader(args, dev_set_o, tokenizer)}
    test_set_dict = {'Circa': create_dataloader(args, test_set_o, tokenizer)}
    
    label_dict = {'Circa': circa_labels}

    if 'TOPICS' in args.aux_tasks:
        # we use only the answer for topic extracting
        orgModelVersion = args.model_version
        argsModified = vars(args)
        argsModified['model_version'] = 'A'
        
        train_set_t, dev_set_t, test_set_t = PrepareSets(args, tokenizer, train_set, dev_set, test_set)
    
        # rename labels
        train_set_t = train_set_t.remove_columns(['labels'])
        dev_set_t = dev_set_t.remove_columns(['labels'])
        test_set_t = test_set_t.remove_columns(['labels'])
        
        train_set_t = train_set_t.rename_column("topic_label", "labels")
        dev_set_t = dev_set_t.rename_column("topic_label", "labels")
        test_set_t = test_set_t.rename_column("topic_label", "labels")
        
        # for the separate aux task, remove samples without topic annotation
        if '' in usedTopicLabels:
            emptyLabelIndex = usedTopicLabels.index('') # actually, we know that position is last element, just to be sure, not costly
            #emptyLabelIndex = len(usedTopicLabels) - 1
            train_set_t = train_set_t.filter(lambda x: x['labels'] != emptyLabelIndex)
            dev_set_t = dev_set_t.filter(lambda x: x['labels'] != emptyLabelIndex)
            test_set_t = test_set_t.filter(lambda x: x['labels'] != emptyLabelIndex)
        
        print('After removing empty topics, we have %d; %d; and %d samples for (respectively) train, dev, test sets for topic aux task' % (len(train_set_t), len(dev_set_t), len(test_set_t)))
        
        # add to dict
        train_set_dict['TOPICS'] = create_dataloader(args, train_set_t, tokenizer)
        dev_set_dict['TOPICS'] = create_dataloader(args, dev_set_t, tokenizer)
        test_set_dict['TOPICS'] = create_dataloader(args, test_set_t, tokenizer)
        label_dict['TOPICS'] = usedTopicLabels[:-1]
        
        argsModified['model_version'] = orgModelVersion

    # return the datasets (and label dict)
    return train_set_dict, dev_set_dict, test_set_dict, label_dict
