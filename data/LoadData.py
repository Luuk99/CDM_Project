# imports
from datasets import load_dataset


def load_circa(args, tokenizer):
    """
    Function to load the Circa dataset.
    Inputs:
        args - Namespace object from the argument parser
        tokenizer - BERT tokenizer instance
    Outputs:
        train_set - Training dataset containg 60% of the data
        dev_set - Development dataset containg 20% of the data
        test_set - Test dataset containg 20% of the data
    """

    # load the dataset
    dataset = load_dataset('circa')['train']

    # filter out instances with label 'Other', 'I am not sure..' or 'N/A' (-1)
    dataset = dataset.filter(lambda example: example['goldstandard1'] != dataset.features['goldstandard1'].str2int('Other'))
    dataset = dataset.filter(lambda example: example['goldstandard1'] != dataset.features['goldstandard1'].str2int("I am not sure how X will interpret Y’s answer"))
    dataset = dataset.filter(lambda example: example['goldstandard1'] != -1)

    # split the dataset into train, dev and test
    dataset = dataset.train_test_split(test_size=0.4, train_size=0.6, shuffle=True)
    train_set = dataset['train']
    dataset = dataset['test'].train_test_split(test_size=0.5, train_size=0.5, shuffle=True)
    dev_set = dataset['train']
    test_set = dataset['test']

    # function that encodes the questions and answers using the tokenizer
    def encode_qa(examples):
         return tokenizer(examples['question-X'], examples['answer-Y'], truncation=True, padding='max_length')

    # function that encodes only the questions using the tokenizer
    def encode_q(examples):
         return tokenizer(examples['question-X'], truncation=True, padding='max_length')

    # function that encodes only the answers using the tokenizer
    def encode_a(examples):
         return tokenizer(examples['answer-Y'], truncation=True, padding='max_length')

    # check which encoding function to use
    if args.model_version == "QA":
        encode_fn = encode_qa
    elif args.model_version == "A":
        encode_fn = encode_a
    else:
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

    # function that handles negative labels
    def handle_labels(examples):
        examples[use_labels] = abs(examples[use_labels])
        return examples

    # handle the labels
    #train_set = train_set.map(handle_labels, batched=False)
    #dev_set = dev_set.map(handle_labels, batched=False)
    #test_set = test_set.map(handle_labels, batched=False)

    # set the labels
    train_set.rename_column_(use_labels, "labels")
    dev_set.rename_column_(use_labels, "labels")
    test_set.rename_column_(use_labels, "labels")

    # return the datasets
    return train_set, dev_set, test_set
