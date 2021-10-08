from transformers import ElectraForPreTraining
import torch.nn.functional as F
import numpy as np
import torch
from einops import rearrange
from torch import nn


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

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index=ignore_index

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)
        return loss

class QA_Model(nn.Module):
    def __init__(self,dr_rate=0.3,n_vocab =32055,pretrained_model="FPTAI/velectra-base-discriminator-cased",alpha1=0.5,alpha2=0.5):
        super(QA_Model,self).__init__()
        self.encoder = ElectraForPreTraining.from_pretrained(pretrained_model).electra
        self.encoder_output_dim = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dr_rate)
        self.linear = nn.Linear(self.encoder_output_dim,2)
        self.has_answer = nn.Linear(self.encoder_output_dim,1)
        self.lm = nn.Linear(self.encoder_output_dim,n_vocab,bias=False)
        embed_weight = self.encoder.embeddings.word_embeddings.weight
        self.lm.weight = embed_weight
        self.lm_bias = nn.Parameter(torch.zeros(n_vocab))
        self.transformer = TransformerLayer(dim=768,heads=8)
        self.alpha1 = 0.5
        self.alpha2 = 0.5
    def forward(self,batch,return_logits=False):
        masked_input_ids,token_type_ids,attention_mask, \
           start_positions,end_positions,has_answer,masked_tokens,masked_pos,question_ids,question_mask,question_token_types \
             = batch["masked_input_ids"],batch["token_type_ids"],batch["attention_mask"], \
            batch["start_position"],batch["end_position"],batch["has_answer"],batch["masked_tokens"],\
               batch["masked_pos"],batch["question_ids"],batch["question_mask"],batch["question_token_types"]
        batch_size = masked_input_ids.size(0)
        all_outputs = self.encoder(masked_input_ids,attention_mask,token_type_ids)[0]
        question_output = self.encoder(question_ids,question_mask,question_token_types)[0]
        outputs = self.transformer(all_outputs,question_output,question_output)
        outputs = self.dropout(outputs)
        logits = self.linear(outputs)
        start_logits,end_logits = logits.split(1,dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        
        total_loss = None
        assert start_positions is not None and end_positions is not None
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0,ignored_index)
        end_positions = end_positions.clamp(0,ignored_index)

        loss_fct = FocalLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits,start_positions)
        end_loss = loss_fct(end_logits, end_positions)

        masked_pos = masked_pos[:,:,None].expand(-1,-1,outputs.size(-1))
        h_masked = torch.gather(outputs,1,masked_pos)
        ce = nn.CrossEntropyLoss()
        logits_lm = self.lm(h_masked) + self.lm_bias
        loss_lm = ce(logits_lm.transpose(1,2),masked_tokens)
        loss_lm = (loss_lm.float()).mean()
        bce = nn.BCEWithLogitsLoss()
        sum_ans_vec = torch.cat([outputs[:,[0]*batch_size,:],outputs[:,start_positions,:],outputs[:,end_positions,:]],dim=1).sum(1)
        has_ans_logits = self.has_answer(sum_ans_vec)
        loss_has_ans = bce(has_ans_logits.squeeze(-1),has_answer.float())
        total_loss = self.alpha1*(end_loss+start_loss)/2+self.alpha2*(loss_has_ans+loss_lm)/2
        if return_logits:
            return start_logits,end_logits,has_ans_logits,(start_loss+end_loss)/2
        return total_loss





