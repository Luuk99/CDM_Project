# imports
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel


class MLTBertForSequenceClassification(BertPreTrainedModel):
    """
    Adaption of the BertForSequenceClassification model from huggingface
    for multitask learning.
    """

    def __init__(self, config):
        """
        Init function that initializes the model.
        Inputs:
            config - Configuration of the model
        """

        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_labels_list = [config.num_labels]
        self.hidden_size = config.hidden_size

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # create a list of classifiers
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.classifiers = [self.classifier]

        self.init_weights()

    def add_aux_classifiers(self, aux_num_labels):
        """
        Function to create the auxilary classifiers for the additional tasks.
        Inputs:
            aux_num_labels - List of number of labels for the additional tasks
        """

        for task_num_labels in aux_num_labels:
            self.num_labels_list.append(task_num_labels)
            self.classifiers.append(nn.Linear(self.hidden_size, task_num_labels).to(self.device))

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
            position_ids=None, head_mask=None, inputs_embeds=None, labels=None, task_idx=None,
            output_attentions=None, output_hidden_states=None, return_dict=None,
        ):
        """
        Forward pass function of the model.

        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifiers[task_idx](pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels_list[task_idx] == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels_list[task_idx]), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
