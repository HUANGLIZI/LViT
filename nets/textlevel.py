import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings

'''text-level module'''


class TextLevelModule(nn.Module):
    def __init__(self, config):
        super(TextLevelModule, self).__init__()
        self.bert_config = BertConfig(
           vocab_size=config["vocab_size"],
           hidden_size=config["hidden_size"],
           num_hidden_layers=config["num_layers"],
           num_attention_heads=config["num_heads"],
           intermediate_size=config["hidden_size"]*config["mlp_ration"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )
        self.text_embeddings = BertEmbeddings(self.bert_config)
        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])

    def forward(self, x):
        x = self.text_embeddings(x)
        return x
