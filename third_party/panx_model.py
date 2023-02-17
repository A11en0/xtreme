import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import XLMPreTrainedModel, XLMModel, RobertaForTokenClassification

from xlm_roberta import XLMRobertaConfig, XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


class CustomXLMRobertaConfig(XLMRobertaConfig):
    langs = ['en', 'de', 'fr']
    num_labels_list = [7, 7, 7]

class CustomXLMRoBertaForTokenClassification(RobertaForTokenClassification):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.ModuleDict(
            {config.langs[i]: nn.Linear(config.hidden_size, config.num_labels) for i, num_labels in enumerate(config.num_labels_list)}
        )

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, langs=None, task=None):

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
        classifier = self.select_classifier(task)
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


if __name__ == '__main__':
    model = CustomXLMRoBertaForTokenClassification.from_pretrained('xlm-roberta-base', langs=['en', 'de', 'fr'], num_labels_list=[7, 7, 7])
    print(model)


