from typing import Optional, Union, Tuple

import torch
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import AutoModel, PreTrainedModel, PretrainedConfig, RobertaForSequenceClassification
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput


class CodeT5pEmbForSequenceClassificationConfig(PretrainedConfig):
    model_type = "t5-emb-for-sequence-classification"

    def __init__(self,
                 encoder_name: Optional[str] = None,
                 num_labels: Optional[int] = 2,
                 classifier_dropout: Optional[float] = 0.1,
                 **kwargs):
        self.encoder_name = encoder_name
        self.num_labels = num_labels
        self.classifier_dropout = classifier_dropout
        super(CodeT5pEmbForSequenceClassificationConfig, self).__init__(**kwargs)


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self,
                 size: int,
                 num_labels: int,
                 classifier_dropout: [float] = 0.1):
        super().__init__()
        self.dense = nn.Linear(size, size)
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(size, num_labels)

    def forward(self, x, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class CodeT5pEmbForSequenceClassification(PreTrainedModel):
    config_class = CodeT5pEmbForSequenceClassificationConfig

    def __init__(self, config: CodeT5pEmbForSequenceClassificationConfig):
        super(CodeT5pEmbForSequenceClassification, self).__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.encoder = AutoModel.from_pretrained(config.encoder_name, trust_remote_code=True)
        self.classifier = ClassificationHead(256, config.num_labels,
                                             config.classifier_dropout)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask
        )
        enc_out = outputs

        logits = self.classifier(outputs)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # if not return_dict:
        #    print('encoder out:', enc_out)
        #    out = (logits,) + enc_out[2:]
        #    return ((loss,) + out) if loss is not None else out

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
