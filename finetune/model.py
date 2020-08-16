import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertPreTrainedModel, RobertaModel, RobertaConfig
from transformers.modeling_bert import BertLayerNorm, gelu


class ClsHead(nn.Module):
    def __init__(self, config, num_labels, dropout=0.3):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, num_labels, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_labels), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x



class CoLAKEForRE(BertPreTrainedModel):
    base_model_prefix = "roberta"

    def __init__(self, config, num_types, ent_emb):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.rel_head = ClsHead(config, num_types)
        self.apply(self._init_weights)
        self.ent_embeddings_n = nn.Embedding.from_pretrained(ent_emb)
        self.num_types = num_types
    
    def tie_rel_weights(self, rel_cls_weight):
        # rel_index: num_types
        self.rel_head.decoder.weight.data = rel_cls_weight
        if getattr(self.rel_head.decoder, "bias", None) is not None:
            self.rel_head.decoder.bias.data = torch.nn.functional.pad(
                self.rel_head.decoder.bias.data,
                (0, self.rel_head.decoder.weight.shape[0] - self.rel_head.decoder.bias.shape[0],),
                "constant",
                0,
            )

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            n_word_nodes=None,
            n_entity_nodes=None, 
            target=None 
    ):
        n_word_nodes = n_word_nodes[0]
        word_embeddings = self.roberta.embeddings.word_embeddings(
            input_ids[:, :n_word_nodes])  # batch x n_word_nodes x hidden_size

        ent_embeddings = self.ent_embeddings_n(
            input_ids[:, n_word_nodes:])

        inputs_embeds = torch.cat([word_embeddings, ent_embeddings],
                                  dim=1)  # batch x seq_len x hidden_size

        outputs = self.roberta(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooler_output = outputs[0][:, 0, :]  # batch x hidden_size
        logits = self.rel_head(pooler_output)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), target.view(-1))
        return {'loss': loss, 'pred': torch.argmax(logits, dim=-1)}


class CoLAKEForTyping(BertPreTrainedModel):
    base_model_prefix = "roberta"

    def __init__(self, config, num_types, ent_emb):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.cls_head = ClsHead(config, num_types)
        self.apply(self._init_weights)
        self.ent_embeddings_n = nn.Embedding.from_pretrained(ent_emb)
        self.num_types = num_types

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            n_word_nodes=None, 
            n_entity_nodes=None, 
            target=None 
    ):
        n_word_nodes = n_word_nodes[0]
        word_embeddings = self.roberta.embeddings.word_embeddings(
            input_ids[:, :n_word_nodes])  # batch x n_word_nodes x hidden_size

        ent_embeddings = self.ent_embeddings_n(
            input_ids[:, n_word_nodes:])

        inputs_embeds = torch.cat([word_embeddings, ent_embeddings],
                                  dim=1)  # batch x seq_len x hidden_size

        outputs = self.roberta(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooler_output = outputs[0][:, 0, :]  # batch x hidden_size
        logits = self.cls_head(pooler_output)
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.num_types), target.view(-1, self.num_types))
        return {'loss': loss, 'pred': logits}