# imports
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# huggingface imports
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers import AdamW
from transformers import Trainer, TrainingArguments

# own imports
from data.LoadData import load_circa


def perform_epoch(model_version, model, tokenizer, optimizer, data_iter, device):
    """
    Function to perform an epoch of the BERT model.
    Inputs:
        model_version - Which version of the model to use
            'Q' - Question only
            'A' - Answer only
            'Q&A' - Question and answer
        model - BERT model instance
        tokenizer - BERT tokenizer instance
        optimizer - Huggingface optimizer
        data_iter - Iterator of the data (can be train, dev, or test)
        device - PyTorch device to use
    Outputs:
        mean_loss - Mean loss over all batches
        accuracy - Accuracy over all batches
    """

    # set the model to training
    model.train()

    # loop over the training batches
    losses = []
    predictions = []
    labels = []
    for batch in data_iter:
        model.zero_grad()

        # check which version to use
        if (model_version == 'Q'):
            model_input = batch.questionX
        elif (model_version == 'A'):
            model_input = batch.answerY
        else:
            model_input = [q + " " + a for q,a in zip(batch.questionX, batch.answerY)]

        # tokenize the batch
        encoding = tokenizer(model_input, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # forward through the model
        outputs = model(input_ids, attention_mask=attention_mask, labels=batch.goldstandard2)
        loss = outputs.loss
        losses.append(loss)
        predictions.append(outputs.logits)
        labels.append(batch.goldstandard2)

        # set a step with the optimizer
        loss.backward()
        optimizer.step()

    # average the training loss for the epoch
    mean_loss = torch.mean(torch.stack(losses, dim=0), dim=0)

    # calculate the training accuracy
    predictions = torch.argmax(torch.cat(predictions, dim=0), dim=1)
    labels = torch.cat(labels, dim=0)
    accuracy = torch.true_divide(torch.sum(predictions == labels), torch.tensor(labels.shape[0], device=labels.device))

    # return the loss and accuracy
    return mean_loss, accuracy

# function to test the model
def test_model(model_version, model, tokenizer, data_iter, device):
    """
    Function to test the trained BERT model.
    Inputs:
        model_version - Which version of the model to use
            'Q' - Question only
            'A' - Answer only
            'Q&A' - Question and answer
        model - BERT model instance
        tokenizer - BERT tokenizer instance
        data_iter - Iterator of the data (can be train, dev, or test)
        device - PyTorch device to use
    Outputs:
        mean_loss - Mean loss over all batches
        accuracy - Accuracy over all batches
    """

    # set the model to evaluation
    model.eval()

    # loop over the training batches
    losses = []
    predictions = []
    labels = []
    for batch in data_iter:
        model.zero_grad()

        # check which version to use
        if (model_version == 'Q'):
            model_input = batch.questionX
        elif (model_version == 'A'):
            model_input = batch.answerY
        else:
            model_input = [q + " " + a for q,a in zip(batch.questionX, batch.answerY)]

        # tokenize the batch
        encoding = tokenizer(model_input, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # forward through the model
        outputs = model(input_ids, attention_mask=attention_mask, labels=batch.goldstandard2)
        loss = outputs.loss
        losses.append(loss)
        predictions.append(outputs.logits)
        labels.append(batch.goldstandard2)

    # average the training loss for the epoch
    mean_loss = torch.mean(torch.stack(losses, dim=0), dim=0)

    # calculate the training accuracy
    predictions = torch.argmax(torch.cat(predictions, dim=0), dim=1)
    labels = torch.cat(labels, dim=0)
    accuracy = torch.true_divide(torch.sum(predictions == labels), torch.tensor(labels.shape[0], device=labels.device))

    # return the loss and accuracy
    return mean_loss, accuracy

# TEST
def qa_perform_epoch(model_version, model, tokenizer, optimizer, data_iter, device):
    # set the model to training
    model.train()

    # loop over the training batches
    losses = []
    predictions = []
    labels = []
    for batch in data_iter:
        model.zero_grad()

        # tokenize the batch
        q_encoding = tokenizer(batch.questionX, return_tensors='pt', padding=True, truncation=True)
        q_input_ids = q_encoding['input_ids'].to(device)
        q_attention_mask = q_encoding['attention_mask'].to(device)
        a_encoding = tokenizer(batch.answerY, return_tensors='pt', padding=True, truncation=True)
        a_input_ids = a_encoding['input_ids'].to(device)
        a_attention_mask = a_encoding['attention_mask'].to(device)
        c_encoding = tokenizer(batch.context, return_tensors='pt', padding=True, truncation=True)
        c_input_ids = c_encoding['input_ids'].to(device)
        c_attention_mask = c_encoding['attention_mask'].to(device)

        # forward through the model
        q_outputs = model(q_input_ids, attention_mask=q_attention_mask, labels=batch.goldstandard2)
        a_outputs = model(a_input_ids, attention_mask=a_attention_mask, labels=batch.goldstandard2)
        c_outputs = model(c_input_ids, attention_mask=c_attention_mask, labels=batch.goldstandard2)
        loss = torch.sum(torch.stack([q_outputs.loss, a_outputs.loss, c_outputs.loss], dim=0), dim=0)
        losses.append(loss)
        batch_predictions = torch.sum(torch.stack([q_outputs.logits, a_outputs.logits, c_outputs.logits], dim=0), dim=0)
        predictions.append(batch_predictions)
        labels.append(batch.goldstandard2)

        # set a step with the optimizer
        loss.backward()
        optimizer.step()

    # average the training loss for the epoch
    mean_loss = torch.mean(torch.stack(losses, dim=0), dim=0)

    # calculate the training accuracy
    predictions = torch.argmax(torch.cat(predictions, dim=0), dim=1)
    labels = torch.cat(labels, dim=0)
    accuracy = torch.true_divide(torch.sum(predictions == labels), torch.tensor(labels.shape[0], device=labels.device))

    # return the loss and accuracy
    return mean_loss, accuracy

# TEST
def qa_test_model(model_version, model, tokenizer, data_iter, device):
    # set the model to evaluation
    model.eval()

    # loop over the training batches
    losses = []
    predictions = []
    labels = []
    for batch in data_iter:
        model.zero_grad()

        # tokenize the batch
        q_encoding = tokenizer(batch.questionX, return_tensors='pt', padding=True, truncation=True)
        q_input_ids = q_encoding['input_ids'].to(device)
        q_attention_mask = q_encoding['attention_mask'].to(device)
        a_encoding = tokenizer(batch.answerY, return_tensors='pt', padding=True, truncation=True)
        a_input_ids = a_encoding['input_ids'].to(device)
        a_attention_mask = a_encoding['attention_mask'].to(device)

        # forward through the model
        q_outputs = model(q_input_ids, attention_mask=q_attention_mask, labels=batch.goldstandard2)
        a_outputs = model(a_input_ids, attention_mask=a_attention_mask, labels=batch.goldstandard2)
        loss = torch.sum(torch.stack([q_outputs.loss, a_outputs.loss], dim=0), dim=0)
        losses.append(loss)
        batch_predictions = torch.sum(torch.stack([q_outputs.logits, a_outputs.logits], dim=0), dim=0)
        predictions.append(batch_predictions)
        labels.append(batch.goldstandard2)

    # average the training loss for the epoch
    mean_loss = torch.mean(torch.stack(losses, dim=0), dim=0)

    # calculate the training accuracy
    predictions = torch.argmax(torch.cat(predictions, dim=0), dim=1)
    labels = torch.cat(labels, dim=0)
    accuracy = torch.true_divide(torch.sum(predictions == labels), torch.tensor(labels.shape[0], device=labels.device))

    # return the loss and accuracy
    return mean_loss, accuracy

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# function to train the BERT model
def train_model(args):
    """
    Function for training and testing the BERT model.
    Inputs:
        args - Namespace object from the argument parser
    """

    # set the seed
    torch.manual_seed(args.seed)

    # check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print the model parameters
    print('-----TRAINING PARAMETERS-----')
    print('Model version: {}'.format(args.model_version))
    print('Labels: {}'.format(args.labels))
    print('PyTorch device: {}'.format(device))
    print('Num epochs: {}'.format(args.num_epochs))
    print('Learning rate: {}'.format(args.lr))
    print('Batch size: {}'.format(args.batch_size))
    print('Logging directory: {}'.format(args.log_dir))
    print('Results directory: {}'.format(args.results_dir))
    print('-----------------------------')

    # check how many labels to use
    if args.labels == 'strict':
        num_labels = 6
    else:
        num_labels = 4

    # load the model
    print('Loading model..')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
        num_labels=num_labels
    ).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print('Model loaded')

    # set the model to training mode
    model.train()

    # load the optimizer
    #optimizer = AdamW(model.parameters(), lr=args.lr)

    # load the dataset
    print('Loading dataset..')
    train_set, dev_set, test_set = load_circa(args, tokenizer)
    print('Dataset loaded')

    # training arguments
    training_args = TrainingArguments(
        output_dir=args.results_dir + '/' + args.model_version + '_' + args.labels,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type='constant',
        warmup_steps=50,
        logging_dir=args.log_dir + '/' + args.model_version + '_' + args.labels,
        evaluation_strategy='epoch',
        do_train =True,
        do_eval = True,
        gradient_accumulation_steps=2,
        eval_accumulation_steps=1,
        save_strategy='epoch',
        save_total_limit=1,
    )

    # create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # train the model
    print('Starting training..')
    trainer.train()
    print('Training finished')

    # test the model
    print('Starting testing..')
    test_output = trainer.predict(test_set)
    print(test_output.metrics)
    print('Testing finished')

    # # train the model for the specified number of epochs
    # print('Starting training..')
    # for epoch in range(1, args.num_epochs+1):
    #     # perform a training epoch
    #     train_loss, train_acc = qa_perform_epoch(args.model_version, model, tokenizer, optimizer, train_iter, device)
    #
    #     # perform a validation epoch
    #     with torch.no_grad():
    #         dev_loss, dev_acc = qa_test_model(args.model_version, model, tokenizer, dev_iter, device)
    #
    #     # print the epoch measures
    #     print('Epoch {}: train_loss {} & train_acc {} & dev_loss {} & dev_acc {}'.format(
    #             epoch,
    #             round(train_loss.item(), 4),
    #             round(train_acc.item(), 4),
    #             round(dev_loss.item(), 4),
    #             round(dev_acc.item(), 4),
    #         )
    #     )
    # print('Training finished')
    #
    # # test the model
    # print('Starting testing..')
    # with torch.no_grad():
    #     test_loss, test_acc = qa_test_model(args.model_version, model, tokenizer, test_iter, device)
    # print('Testing finished: test_loss {} & test_acc {}'.format(
    #         round(test_loss.item(), 4),
    #         round(test_acc.item(), 4),
    #     )
    # )


# command line arguments parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # model hyperparameters
    parser.add_argument('--model_version', default='Q&A', type=str,
                        help='What model version to use. Default is Q&A (Question and Answer)',
                        choices=['Q&A', 'Q', 'A'])
    parser.add_argument('--labels', default='strict', type=str,
                        help='What labels to use. Default is strict',
                        choices=['strict', 'relaxed'])

    # training hyperparameters
    parser.add_argument('--num_epochs', default=3, type=int,
                        help='Number of epochs to train for. Default is 3')

    # optimizer hyperparameters
    parser.add_argument('--lr', default=3e-5, type=float,
                        help='Learning rate to use. Default is 3e-5')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Minibatch size. Default is 8')

    # loading hyperparameters
    parser.add_argument('--checkpoint_dir', default=None, type=str,
                        help='Directory where the model checkpoint is located. Default is None (no checkpoint used)')

    # other hyperparameters
    parser.add_argument('--seed', default=1234, type=int,
                        help='Seed to use for reproducing results. Default is 1234')
    parser.add_argument('--log_dir', default='./train_logs', type=str,
                        help='Directory where the training logs should be created. Default is ./train_logs')
    parser.add_argument('--results_dir', default='./results', type=str,
                        help='Directory where the training results should be created. Default is ./results')

    # parse the arguments
    args = parser.parse_args()

    # train the model
    train_model(args)
