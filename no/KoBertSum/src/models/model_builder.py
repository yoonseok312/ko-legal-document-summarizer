import copy
import os
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F

from models.decoder import TransformerDecoder
from models.encoder import Classifier, ExtTransformerEncoder
from models.optimizers import Optimizer
from models.BERTmodel.transformer import TransformerBlock
from models.BERTmodel.embedding import BERTEmbedding
from models.BERTmodel.bert import BERT

def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim'] #[0]  ['optims'] [0]   원래 ['optim'][0]이었는데 오류남
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))


    return optim

def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim

def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim


def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator

# class Bert(nn.Module):
#     def __init__(self, large, temp_dir, finetune=False):
#         super(Bert, self).__init__()
#         # if(large):
#         #     self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
#         # else:
#         self.model = BertModel.from_pretrained("monologg/kobert", cache_dir=temp_dir)
#
#         self.finetune = finetune
#
#     def forward(self, x, segs, mask):
#         if(self.finetune):
#             print("x", x.shape)
#             print("token", segs.shape)
#             print("mask", mask.shape)
#             top_vec, _ = self.model(x, token_type_ids=segs, attention_mask=mask)
#             #print(last_hiddens, last_pooling_hiddens, hiddens)
#             #top_vec = hiddens[-1]
#         else:
#             self.eval()
#             with torch.no_grad():
#                 top_vec, _ = self.model(x, token_type_ids=segs, attention_mask=mask)
#         return top_vec

class Bert(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, large, temp_dir, finetune=False, hidden=256, n_layers=8, attn_heads=8, vocab_size=363426, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        # print(vocab_size, hidden, n_layers, attn_heads, dropout)
        self.model = BERT(vocab_size, hidden, n_layers, attn_heads, dropout)
        print(os.getcwd())
        self.model.load_state_dict(torch.load('models/bert.model.ep100'))
        # model.eval()

        self.finetune = finetune
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
        print(attn_heads, hidden)
    def forward(self, x, segment_info, mask):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        if self.finetune:
        # embedding the indexed sequence to sequence of vectors
        #     x = self.embedding(x, segment_info)
        #
        #     # running over multiple transformer blocks
        #     for transformer in self.transformer_blocks:
        #         x = transformer.forward(x, mask)
        #
        #     return x
            top_vec = self.model(x, segment_info, mask)
            return top_vec
        else:
            self.eval()
            with torch.no_grad():
                top_vec = self.model(x, segment_info, mask)
            return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        self.ext_layer = ExtTransformerEncoder(256, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, args.ext_layers)

        self.lstm_layer = nn.LSTM(
            input_size=256, hidden_size=256
        )
        self.wo = nn.Linear(256, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.generator = Generator(256)

        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size,
                                     num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads, intermediate_size=args.ext_ff_size)
            self.bert.model = BertModel(bert_config)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)

        # need this?
        # if(args.max_pos>512):
        #     my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
        #     my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
        #     my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
        #     self.bert.model.embeddings.position_embeddings = my_pos_embeddings


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

        self.to(device)


    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores, add_lstm = self.ext_layer(sents_vec, mask_cls) # .squeeze(-

        # if add_lstm:
        #     sent_scores_lstm, _ = self.lstm_layer(
        #         sent_scores
        #     )
        #     sent_scores_lstm2 = self.sigmoid(self.wo(sent_scores_lstm))
        # # sent_scores_lstm2 = self.generator(sent_scores_lstm)
        #     sent_scores_lstm3 = sent_scores_lstm2.squeeze(-1) * mask_cls.float()
        #     sent_scores_lstm4 = sent_scores_lstm3.squeeze(-1)
        #     return sent_scores_lstm4, mask_cls
        # # generator_output = self.generator(sent_scores_lstm4)
        # else:
            # final_sent_scores = sent_scores_lstm4 * 0.3 + sent_scores * 0.7
        return sent_scores, mask_cls

        # else:
        #     return sent_scores, mask_cls

class Generator(nn.Module):
    def __init__(self, d_model, hp=None):
        super().__init__()
        self.fc_1 = nn.Linear(d_model, int(d_model / 2))
        self.fc_2 = nn.Linear(int(d_model / 2), int(d_model / 4))
        self.fc_3 = nn.Linear(int(d_model / 4), 1)

        self.ln_1 = nn.LayerNorm(int(d_model / 2))
        self.ln_2 = nn.LayerNorm(int(d_model / 4))

    def forward(self, x):
        h = self.fc_1(x)
        h = self.ln_1(h)
        h = F.relu(h)

        h = self.fc_2(h)
        h = self.ln_2(h)
        h = F.relu(h)

        h = self.fc_3(h)
        h = torch.sigmoid(h)
        return h


class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        if bert_from_extractive is not None:
            self.bert.model.load_state_dict(
                dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)

        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
                                     num_hidden_layers=args.enc_layers, num_attention_heads=8,
                                     intermediate_size=args.enc_ff_size,
                                     hidden_dropout_prob=args.enc_dropout,
                                     attention_probs_dropout_prob=args.enc_dropout)
            self.bert.model = BertModel(bert_config)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if(args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        return decoder_outputs, None
