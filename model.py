from transformers import ElectraForPreTraining
import torch.nn.functional as F
import numpy as np
import torch
from einops import rearrange
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.nn as nn
from prediction import Prediction,find_context_question_for_batch

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None):
        """
        Implementation of multi-head attention layer of the original transformer model.
        einsum and einops.rearrange is used whenever possible
        Args:
            dim: token's dimension, i.e. word embedding vector size
            heads: the number of distinct representations to learn
            dim_head: the dim of the head. In general dim_head<dim.
            However, it may not necessary be (dim/heads)
        """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        # self.to_qvk = nn.Linear(dim, _dim , bias=False)
        self.to_q = nn.Linear(dim,_dim)
        self.to_k = nn.Linear(dim, _dim)
        self.to_v = nn.Linear(dim, _dim)
        self.W_0 = nn.Linear( _dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5

    def forward(self, q,k,v, mask=None):
        # qkv = self.to_qvk(x)
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        q = rearrange(q, 'b t (d h) ->  b h t d ', h=self.heads)
        k = rearrange(k, 'b t (d h) ->  b h t d ', h=self.heads)
        v = rearrange(v, 'b t (d h) ->  b h t d ', h=self.heads)
        scaled_dot_prod = torch.einsum('b h i d , b h j d -> b h i j', q, k) * self.scale_factor
        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[2:]
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)
        attention = torch.softmax(scaled_dot_prod, dim=-1)
        out = torch.einsum('b h i j , b h j d -> b h i d', attention, v)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.W_0(out)

class TransformerLayer(nn.Module):
   def __init__(self, dim, heads=8, dim_head=None, dim_linear_block=1024, dropout=0.1):
       super().__init__()
       self.mhsa = MultiHeadSelfAttention(dim=dim, heads=heads, dim_head=dim_head)
       self.drop = nn.Dropout(dropout)
       self.norm_1 = nn.LayerNorm(dim)
       self.norm_2 = nn.LayerNorm(dim)

       self.linear = nn.Sequential(
           nn.Linear(dim, dim_linear_block),
           nn.ReLU(),
           nn.Dropout(dropout),
           nn.Linear(dim_linear_block, dim),
           nn.Dropout(dropout)
       )
   def forward(self, q,k,v, mask=None):
       y = self.norm_1(self.drop(self.mhsa(q,k,v, mask)) + q)
       return self.norm_2(self.linear(y) + y)

class TransformerBlock(nn.Module):
    def __init__(self,n_layer):
        super().__init__()
        self.n_layers = n_layer
        self.block = nn.ModuleList([TransformerLayer(dim=768, heads=8) for _ in range(self.n_layers)])
    def forward(self,q,k,v):
        for i in range(self.n_layers):
            output = self.block[i](q,k,v)
            q = output
        return output

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2, weight=None, ignore_index=-100):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.weight = weight
#         self.ignore_index=ignore_index
#
#     def forward(self, input, target):
#         logpt = F.log_softmax(input, dim=1)
#         pt = torch.exp(logpt)
#         logpt = (1-pt)**self.gamma * logpt
#         loss = F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)
#         return loss
def mean_pooling(model_output,attention_mask):
    token_embedding = model_output
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embedding.size()).float()
    return torch.sum(token_embedding*input_mask_expanded,1)/torch.clamp(input_mask_expanded.sum(1),min=1e-9)

class SentenceModel(nn.Module):
    def __init__(self,out_dim,pretrained_model):
        super(SentenceModel,self).__init__()
        self.encoder = ElectraForPreTraining.from_pretrained(pretrained_model).electra
        self.decoder = torch.nn.Linear(768, out_dim)
        self.dropout = nn.Dropout(0.3)
    def forward(self,input_ids ,attention_mask):
        output_sentence = self.encoder(input_ids=input_ids,
                                                attention_mask=attention_mask).last_hidden_state
        mean_pooling_output_sentence = mean_pooling(output_sentence,attention_mask)
        mean_pooling_output_sentence = self.dropout(mean_pooling_output_sentence)
        has_ans_logits = self.decoder(mean_pooling_output_sentence)
        return has_ans_logits

class BiLSTM(nn.Module):
    """
    input size: (batch, seq_len, input_size)
    output size: (batch, seq_len, num_directions * hidden_size) last layer
    h_0: (num_layers * num_directions, batch, hidden_size)
    c_0: (num_layers * num_directions, batch, hidden_size)
    """
    def __init__(self, n_layers, input_size, hidden_size,dropout=0.3):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size  # TODO: verify the input size
        self.pdrop = dropout
        self.n_layers = n_layers
        self.bilstm = nn.LSTM(self.input_size, self.hidden_size, self.n_layers, batch_first=True, bidirectional=True)
        self.h0 = None
        self.c0 = None

    def forward(self, X, X_mask, dropout=True, use_packing=False):
        if use_packing:
            N, max_len = X.shape[0], X.shape[-2]
            lens = torch.sum(X_mask, dim=-1)
            lens += torch.eq(lens, 0).long()  # Avoid length 0
            lens, indices = torch.sort(lens, descending=True)
            _, rev_indices = torch.sort(indices, descending=False)

            X = pack(X[indices], lens, batch_first=True)
            H, (h0, c0) = self.bilstm(X)
            H, _ = unpack(H, total_length=max_len, batch_first=True)

            # h0: [2, N, hidden_size]
            # H : [N, maxlen, hidden_size*2]
            H = H[rev_indices]
            h0 = torch.transpose(h0, 0, 1)
            h0 = h0[rev_indices].view(N, 2 * self.hidden_size)
            c0 = torch.transpose(c0, 0, 1)
            c0 = c0[rev_indices].view(N, 2 * self.hidden_size)
        else:
            H, (h0, c0) = self.bilstm(X)

        if X_mask is not None:
            H = H * X_mask.unsqueeze(-1).float()
        if dropout:
            H = F.dropout(H, self.pdrop, training=self.training)
        return H, (h0, c0)

class QA_Model(nn.Module):
    def __init__(self,dr_rate=0.5,pretrained_model="FPTAI/velectra-base-discriminator-cased",alpha1=0.5,alpha2=0.5):
        super(QA_Model,self).__init__()
        self.sentence_encoder = SentenceModel(2,pretrained_model)
        self.sentence_encoder.load_state_dict(torch.load("./saved_model/cross-encoder-reranking-model.pt"))
        self.encoder = self.sentence_encoder.encoder
        self.encoder_output_dim = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dr_rate)
        self.linear = nn.Linear(self.encoder_output_dim,2)
        # self.lm = nn.Linear(self.encoder_output_dim,n_vocab,bias=False)
        # embed_weight = self.encoder.embeddings.word_embeddings.weight
        # self.lm.weight = embed_weight
        # self.lm_bias = nn.Parameter(torch.zeros(n_vocab))
        self.transformer_block = TransformerBlock(n_layer=3)
        self.bilstm = BiLSTM(2,self.encoder_output_dim,self.encoder_output_dim//2)
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def forward(self,batch,return_logits=False,train=True,pred_list=None,tokenizer=None,input_feature_list=None):
        if train:
            input_ids,token_type_ids,attention_mask, \
               start_positions,end_positions,has_answer,question_ids,question_mask,question_token_types,sentence_ans_ids,masked_sentence_ans_ids\
                 = batch["input_ids"],batch["token_type_ids"],batch["attention_mask"], \
                batch["start_position"],batch["end_position"],batch["has_answer"],batch["question_ids"],batch["question_mask"],batch["question_token_types"], \
            batch["sentence_ans_ids"],batch["masked_sentence_ans_ids"]

            all_outputs = self.encoder(input_ids,attention_mask,token_type_ids)[0]
            all_outputs = self.dropout(all_outputs)
            question_embed = self.encoder.embeddings.word_embeddings(question_ids)
            question_output,_ = self.bilstm(question_embed,question_mask)

            outputs = self.transformer_block(all_outputs,question_output,question_output)

            outputs = self.dropout(outputs)
            logits = self.linear(outputs)

            start_logits,end_logits = logits.split(1,dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()

            assert start_positions is not None and end_positions is not None
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)
            cls_na_start_positions = start_positions*has_answer
            cls_na_end_positions = end_positions*has_answer
            cls_na_start_positions = cls_na_start_positions.clamp(0,ignored_index)
            cls_na_end_positions = cls_na_end_positions.clamp(0,ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits,cls_na_start_positions)
            end_loss = loss_fct(end_logits, cls_na_end_positions)

            has_ans_logits = self.sentence_encoder(sentence_ans_ids,masked_sentence_ans_ids)
            loss_has_ans = loss_fct(has_ans_logits,has_answer.view(-1))

            if return_logits:
                return start_logits,end_logits,has_ans_logits,(start_loss+end_loss)/2

            # masked_pos = masked_pos[:, :, None].expand(-1, -1, outputs.size(-1))
            # h_masked = torch.gather(outputs, 1, masked_pos)
            # ce = nn.CrossEntropyLoss()
            # logits_lm = self.lm(h_masked) + self.lm_bias
            # loss_lm = ce(logits_lm.transpose(1, 2), masked_tokens)
            # loss_lm = (loss_lm.float()).mean()

            total_loss = self.alpha1*(end_loss+start_loss)/2+self.alpha2*(loss_has_ans)

            return total_loss,(start_loss+end_loss)/2,(loss_has_ans)
        else:
            ids, input_ids, attention_mask, token_type_ids, question_ids, question_mask, question_token_types = \
                batch["ids"], batch["input_ids"], batch["attention_mask"], \
                batch["token_type_ids"], batch[
                    "question_ids"], batch["question_mask"], batch["question_token_types"]
            questions,contexts,offset_mappings = find_context_question_for_batch(input_feature_list,ids)
            all_outputs = self.encoder(input_ids, attention_mask, token_type_ids)[0]

            question_embed = self.encoder.embeddings.word_embeddings(question_ids)
            question_output, _ = self.bilstm(question_embed, question_mask)

            outputs = self.transformer_block(all_outputs, question_output, question_output)

            outputs = self.dropout(outputs)
            logits = self.linear(outputs)

            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()

            start_indexes = torch.argmax(start_logits, dim=-1)
            start_scores = start_logits.gather(1, start_indexes.unsqueeze(-1))
            end_indexes = torch.argmax(end_logits, dim=-1)
            end_scores = end_logits.gather(1, end_indexes.unsqueeze(-1))

            for ind, (start_score, start_index, end_score, end_index) in enumerate(
                    zip(start_scores, start_indexes, end_scores, end_indexes)):
                if end_index < start_index:
                    if start_index == end_index:
                        start_char = offset_mappings[ind][start_index][0]
                        end_char = offset_mappings[ind][end_index][1]
                        pred_ans = contexts[ind][start_char:end_char]
                        if pred_ans == "":
                            na_score = (start_score + end_score) / 2
                            pred_list.append(
                                Prediction(ids[ind], pred_ans, na_score=na_score.cpu().tolist()[0]))
                    else:
                        pred_ans = ""
                        na_score = (start_score + end_score) / 2
                        pred_list.append(
                            Prediction(ids[ind], pred_ans, na_score=na_score.cpu().tolist()[0]))
                else:
                    start_char = offset_mappings[ind][start_index][0]
                    end_char = offset_mappings[ind][end_index][1]
                    pred_ans = contexts[ind][start_char:end_char]
                    sentence_ans_len = 256 - len(pred_ans)
                    side_sentece_ans_len = sentence_ans_len // 2
                    start_sentence = -1
                    end_sentence = -1
                    for i in range(start_index - side_sentece_ans_len, start_index):
                        if contexts[ind][i:i + 1] == " ":
                            start_sentence = i + 1
                            break
                    for j in range(start_index + len(pred_ans) + side_sentece_ans_len, start_index + len(pred_ans), -1):
                        if contexts[ind][j:j + 1] == " ":
                            end_sentence = j - 1
                            break
                    sentence_ans = tokenizer(questions[ind] + " " + contexts[ind][
                                                                    start_sentence:end_sentence].strip() + ' [SEP] ' + pred_ans)
                    sentence_input_ids = torch.LongTensor(sentence_ans["input_ids"]).to(input_ids.device)
                    sentence_mask_attention = torch.LongTensor(sentence_ans["attention_mask"]).to(input_ids.device)
                    has_ans_logits = self.sentence_encoder(sentence_input_ids.unsqueeze(0),
                                                           sentence_mask_attention.unsqueeze(0))
                    clf = torch.argmax(torch.nn.Softmax(dim=-1)(has_ans_logits), dim=-1)
                    # if clf == 1:
                        # if pred_ans == "":
                            # print(contexts[ind])
                            # print(start_char)
                            # print(end_char)
                            # print(start_index)
                            # print(end_index)
                    has_score = (start_score + end_score)/2
                    pred_list.append(Prediction(ids[ind], pred_ans, has_score=has_score.cpu().tolist()[0],score_verifier=has_ans_logits.squeeze(0).cpu().tolist()))
                    # else:
                    #     pred_ans = ""
                    #     na_score = (start_score + end_score)/2
                    #     pred_list.append(
                    #         Prediction(ids[ind], pred_ans, na_score=na_score.cpu().tolist()[0], score_verifier=has_ans_logits.squeeze(0).cpu().tolist()))
            return pred_list




