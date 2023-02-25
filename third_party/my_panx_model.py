import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import XLMPreTrainedModel, XLMModel, RobertaForTokenClassification, BertPreTrainedModel, \
    RobertaConfig, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, RobertaModel, XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP

from xlm_roberta import XLMRobertaConfig, XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP

NO_LABEL = -100

# Version: transformer 2.3
class CustomXLMRobertaConfig(XLMRobertaConfig):
    langs = ['en', 'de', 'fr']
    num_labels_list = [7, 7, 7]

# Version: transformer 2.3
class CustomXLMRoBertaForTokenClassification(RobertaForTokenClassification):
    config_class = CustomXLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP

    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.ModuleDict(
            {config.langs[i]: nn.Linear(config.hidden_size, config.num_labels) for i, num_labels in
             enumerate(config.num_labels_list)}
        )  # reload，don't change the name "self.classifier"

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, langs=None, task=None, predict_head=None):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        # Training stage
        if task:
            classifier = self.select_classifier(task)
            logits = classifier(sequence_output)
        # Testing stage, average all classifier heads' output
        else:
            if predict_head == 'mean':
                logit_list = torch.stack([self.classifier[k](sequence_output) for k in self.classifier.keys()])
                logits = torch.mean(logit_list, dim=0)
            else:  # single-head predict
                classifier = self.select_classifier(predict_head)
                logits = classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

    def select_classifier(self, lang):
        return self.classifier[lang]



class PseudoXLMRobertaConfig(RobertaConfig):
    pretrained_config_archive_map = XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "xlm-roberta"

class PseudoXLMRobertaForTokenClassification(BertPreTrainedModel):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

        # self.af = config.af
        self.af = 0.3

    def create_loss_fn(self, loss_type):
        criterion = None
        if loss_type == 'soft':
            # for pytorch 0.4.0
            # criterion = nn.CrossEntropyLoss(ignore_index=NO_LABEL, reduce=False)
            # for pytorch 0.4.1
            # criterion = nn.CrossEntropyLoss(ignore_index=NO_LABEL, reduction='none')
            criterion = nn.CrossEntropyLoss(ignore_index=NO_LABEL, reduction='mean')
        return criterion

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        in_batch_task_id=None
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # loss_fct = self.create_loss_fn(loss_type='soft')
            # Only keep active parts of the loss

            def pseudo_loss_cal(outputs):
                with torch.no_grad():
                    pseudo_labeled = outputs.max(1)[1]

                unlabeled_loss = loss_fct(outputs, pseudo_labeled)
                loss = 0.3 * unlabeled_loss
                return loss

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                # logits, labels = active_logits.view(-1, self.num_labels), active_labels.view(-1)

                if in_batch_task_id and in_batch_task_id > 3:   # unlabeled training data
                    loss = pseudo_loss_cal(active_logits)
                else:
                    loss = loss_fct(active_logits, active_labels)
            else:
                logits, labels = logits.view(-1, self.num_labels), labels.view(-1)

                if in_batch_task_id and in_batch_task_id > 3:   # unlabeled training data
                    loss = pseudo_loss_cal(logits)
                else:
                    loss = loss_fct(logits, labels)

                # loss = labeled_loss + self.unlabeled_weight() * unlabeled_loss
                # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

    def unlabeled_weight(self):
        alpha = 0.0
        if self.epoch > self.T1:
            alpha = (self.epoch-self.T1) / (self.T2-self.T1)*self.af
            if self.epoch > self.T2:
                alpha = self.af
        return alpha

if __name__ == '__main__':
    model = CustomXLMRoBertaForTokenClassification.from_pretrained('xlm-roberta-base')
    print(model)
