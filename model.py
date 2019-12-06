import torch.nn as nn
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel, BertOnlyMLMHead, BertPreTrainingHeads


class BertForEmotionClassification(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForEmotionClassification, self).__init__(config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc = nn.Linear(config.hidden_size, self.num_labels)
        self.apply(self.init_weights)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                masked_lm_labels=None, classification_label=None):
        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                token_type_ids=token_type_ids,
                                head_mask=head_mask)

        # pooled_output : Last layer hidden-state of the first token of the sequence ([CLS] token)
        pooled_output = bert_output[1]
        pooled_output = self.dropout(pooled_output)
        output = self.fc(pooled_output)

        if classification_label is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(output.view(-1, self.num_labels), classification_label.view(-1))
            output = (output, loss)

        return output


class BertForMLM(BertPreTrainedModel):

    def __init__(self, config):
        ''' 추가적인 Pretraining을 진행하기 위한 모듈 '''
        super(BertForMLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                masked_lm_labels=None, classification_label=None):
        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                token_type_ids=token_type_ids,
                                head_mask=head_mask)

        # sequence_output : Last layer hidden-state sequence
        sequence_output = bert_output[0]
        # logits of each vocab
        output = self.cls(sequence_output)

        if masked_lm_labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fn(output.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            output = (output, loss)

        return output


class BertForMLMwithClassification(BertPreTrainedModel):
    r"""
    Masked Language Model과 classification을 동시에 수행합니다.
    """
    def __init__(self, config):
        super(BertForMLMwithClassification, self).__init__(config)
        self.bert = BertModel(config)

        self.num_labels = config.num_labels
        self.cls = BertPreTrainingHeads(config)
        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                masked_lm_labels=None, classification_label=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output, pooled_output = outputs[:2]
        output = self.cls(sequence_output, pooled_output)

        if masked_lm_labels is not None and classification_label is not None:
            prediction_scores, sentiment_scores = output
            loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fn(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            sentiment_loss = loss_fn(sentiment_scores.view(-1, 2), classification_label.view(-1))
            loss = masked_lm_loss + sentiment_loss
            output = (output, loss)

        return output

