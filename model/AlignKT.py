import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_
from .TransformerLayer import TransformerEncoderLayer, TransformerDecoderLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AlignKT(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, drop_out, n_blocks, **dkwargs):
        """
            data_kwargs: used for initializing the variables related to the dataset
        """
        super().__init__()
        # 1 get all params
        self.model_name = "AlignKT"
        self.n_question = dkwargs['num_c']
        self.n_pid = dkwargs['num_q']
        self.d_model, self.n_heads, self.d_ff, self.drop_out, self.n_blocks = d_model, n_heads, d_ff, drop_out, n_blocks

        # ratio of l2 regularization
        self.l2 = 1e-5

        # adjust the ratio of questions and concepts in method _embedding
        self.a1, self.a2 = 0.8, 0.5

        # ratio of constractive loss
        self.l_cl = 0.05
        self.temprerature = 0.03

        # 2 embedding layers
        self.difficult_param = nn.Embedding(self.n_pid+1, 1)  #
        self.pid_diff_proj = nn.Linear(1, self.d_model)  #

        self.q_embed_var = nn.Embedding(self.n_question+1, self.d_model)  # question emb
        self.qa_embed_var = nn.Embedding(2 * (self.n_question+1), self.d_model)  # interaction emb

        self.q_embed = nn.Embedding(self.n_question+1, self.d_model)
        self.qa_embed = nn.Embedding(2 * (self.n_question+1), self.d_model)  # interaction emb

        # 3 vanilla transformer sub-model
        self.model = Transformer(n_blocks=n_blocks, d_model=d_model, d_ff=d_ff, n_heads=n_heads, drop_out=drop_out)

        # 4 initialize the ideal states that represents the a ideal state of the student: students know all concepts base on provided dataset
        self.ideal_states_idx = torch.arange(self.n_question+1, (self.n_question+1)*2).unsqueeze(0).to(device)  # shape=(1, n_concepts), and dim=0 will be expand to batch_size

        # 5 personal states retriever
        self.ideal_states_encoder = TransformerEncoderLayer(d_model, d_model, d_model, d_model, n_heads, d_ff, drop_out)  # sa
        self.personal_states_retriever = TransformerEncoderLayer(d_model, d_model, d_model, d_model, n_heads, d_ff, drop_out)  # mha

        # 6 output layer
        self.out = nn.Sequential(
            nn.Linear(self.d_model + self.d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.drop_out),
            nn.Linear(self.d_ff, self.d_model),
            nn.GELU(),
            nn.Dropout(self.drop_out),
            nn.Linear(self.d_model, 1)
        )

        constant_(self.difficult_param.weight, 0.)


    def _embedding(self, q_data, target, q_shft, pid_data=None, pid_shft=None):  # c, r, cshft, q, qshft
        q_embed_data = self.q_embed(q_shft)  # (bs, seqlen, d_model), c_ct
        qa_data = q_data + (self.n_question+1) * target  # +1 to adapt the [mask] question
        qa_embed_data = self.qa_embed(qa_data)

        # rasch model
        q_embed_var_data = self.q_embed_var(q_shft)  # d_ct
        pid_shft_diff_data = self.difficult_param(pid_shft)  # μq_shft
        pid_shft_diff_proj = self.pid_diff_proj(pid_shft_diff_data)
        # q_embed_data = q_embed_data + (pid_shft_embed_data * q_embed_var_data)  # (μ_q * d_ct) + c_ct, ori-rasch model
        q_embed_data = (1-self.a1) * q_embed_data + self.a1 * pid_shft_diff_data * (q_embed_var_data + pid_shft_diff_proj)  # μ_q * (q_t + d_ct) + c_ct, modified-rasch model

        qa_embed_var_data = self.qa_embed_var(qa_data)  # f_(ct,rt)
        pid_diff_data = self.difficult_param(pid_data)  # μq
        pid_diff_proj = self.pid_diff_proj(pid_diff_data)  # q_t
        # qa_embed_data = qa_embed_data + (pid_embed_data * qa_embed_var_data)  # (μ_q * f_(ct, rt)) + e_(ct, rt)
        qa_embed_data = (1-self.a2) * qa_embed_data + self.a2 * pid_diff_data * (qa_embed_var_data + pid_diff_proj)  # μ_q * (q_t + f_(ct,rt)) + e_(ct,rt)

        return q_embed_data, qa_embed_data, pid_diff_data


    def _cal_personal_states(self, states):
        batch_size, seq_len, concepts_len = states.shape[0], states.shape[1], self.ideal_states_idx.shape[1]

        # 1 construct the ideal_states with problems(questions) info
        ideal_states_idx = self.ideal_states_idx.repeat(batch_size, 1)  # shape=(batch_size, n_concepts)
        ideal_states = self.qa_embed(ideal_states_idx)  # shape=(batch_size, n_concepts, d_model)

        # 2 get the personal_states
        # both the benath masks are all ones(no a causal mask)
        idealState_src_mask = torch.ones((concepts_len, concepts_len)).to(device) == 1
        personalState_src_mask = torch.ones((seq_len, concepts_len)).to(device) == 1

        for i in range(self.n_blocks):  # sa
            ideal_states, _ = self.ideal_states_encoder(ideal_states, ideal_states, ideal_states, mask=idealState_src_mask, is_effect=False)

        for i in range(self.n_blocks):  # mha
            states, personal_score = self.personal_states_retriever(states, ideal_states, ideal_states, mask=personalState_src_mask, is_effect=False)

        return states, ideal_states, personal_score


    def _cal_InfoNCE(self, q_data, r_data, q_shft, sm, pid_data=None, pid_shft=None):  # c, r, cshft, q, qshft
        batch_size, seq_len = q_data.shape[0], q_data.shape[1]
        feature_len = self.d_model

        # 1 construct a mask, shape=(batch_size, seq_len, feature_len) used for cl_loss
        mask_list = []
        for idx in range(batch_size):
            valid_len = sm[idx].sum()
            mask_list.append(torch.cat([torch.ones((valid_len, feature_len)), torch.zeros((seq_len-valid_len, feature_len))], dim=0))
        src_mask = torch.stack(mask_list, dim=0).to(device)  # shape=(batch_size, seq_len, feature_len), tackle the padding

        causal_mask = torch.triu(torch.ones((seq_len, seq_len)).to(device), 1) == 0  # used for encoder

        # 2
        q = q_data.clone().detach()  # concepts
        qshft = q_shft.clone().detach()
        r = r_data.clone().detach()  # answers
        p = pid_data.clone().detach()  # questions
        pshft = pid_shft.clone().detach()

        # 3 get the ori_states
        q_embed_data_ori, qa_embed_data_ori, _ = self._embedding(q, r, qshft, p, pshft)

        for block in self.model.knowledge_encoder:
            qa_embed_data_ori, _ = block(qa_embed_data_ori, qa_embed_data_ori, qa_embed_data_ori, mask=causal_mask,
                                         is_effect=False)

        for block in self.model.states_retriever:
            q_embed_data_ori, _ = block.masked_sa(q_embed_data_ori, q_embed_data_ori, q_embed_data_ori, mask=causal_mask,
                                                  is_effect=False)

        qa_embed_data_ori = qa_embed_data_ori * src_mask
        q_embed_data_ori = q_embed_data_ori * src_mask

        # 4 part A: positive, order-permute and [mask] q
        q_, r_, qshft_, p_, pshft_ = (q_data.clone().detach(), r_data.clone().detach(), q_shft.clone().detach(),
                                      pid_data.clone().detach(), pid_shft.clone().detach())
        for bs in range(batch_size):
            if sm[bs].sum() >= 4:
                len_list = [max(1, int((sm[bs].sum() - 2) * 0.6)), max(1, int((sm[bs].sum() - 2) * 0.4))]
                idx_list = [np.random.choice(range(1, sm[bs].sum() - 1), length, replace=False) for length in len_list]
                # idx = random.sample(range(1, sm[bs].sum()-1), max(1, int((sm[bs].sum()-2) * 0.9)))
                for i in idx_list[0]:
                    q_[bs, i], q_[bs, i + 1] = q_[bs, i + 1], q_[bs, i]
                    r_[bs, i], r_[bs, i + 1] = r_[bs, i + 1], r_[bs, i]
                    qshft_[bs, i - 1], qshft_[bs, i] = qshft_[bs, i], qshft_[bs, i - 1]
                    p_[bs, i], p_[bs, i + 1] = p_[bs, i + 1], p_[bs, i]
                    pshft_[bs, i - 1], pshft_[bs, i] = pshft_[bs, i], pshft_[bs, i - 1]

                # [mask] q
                # sub_idx = random.sample(range(len(idx)), max(1, int(len(idx) * 0.5)))
                for i in idx_list[1]:
                    q_[bs, i] = self.n_question
                    qshft_[bs, i - 1] = self.n_question
                    p_[bs, i] = self.n_pid
                    pshft_[bs, i - 1] = self.n_pid

        q_embed_data_A, qa_embed_data_A, _ = self._embedding(q_, r_, qshft_, p_, pshft_)

        for block in self.model.knowledge_encoder:
            qa_embed_data_A, _ = block(qa_embed_data_A, qa_embed_data_A, qa_embed_data_A, mask=causal_mask,
                                       is_effect=False)

        for block in self.model.states_retriever:
            q_embed_data_A, _ = block.masked_sa(q_embed_data_A, q_embed_data_A, q_embed_data_A, mask=causal_mask,
                                                is_effect=False)

        qa_embed_data_A = qa_embed_data_A * src_mask
        q_embed_data_A = q_embed_data_A * src_mask

        # 5 part B: hard-neg
        q_, r_, qshft_, p_, pshft_ = (q_data.clone().detach(), r_data.clone().detach(), q_shft.clone().detach(),
                                      pid_data.clone().detach(), pid_shft.clone().detach())
        for bs in range(batch_size):
            idx = random.sample(range(sm[bs].sum()), max(1, int(sm[bs].sum() * 0.9)))
            for i in idx:
                r_[bs, i] = 1 - r_[bs, i]

        q_embed_data_B, qa_embed_data_B, _ = self._embedding(q_, r_, qshft_, p_, pshft_)

        for block in self.model.knowledge_encoder:
            qa_embed_data_B, _ = block(qa_embed_data_B, qa_embed_data_B, qa_embed_data_B, mask=causal_mask,
                                       is_effect=False)

        for block in self.model.states_retriever:
            q_embed_data_B, _ = block.masked_sa(q_embed_data_B, q_embed_data_B, q_embed_data_B, mask=causal_mask,
                                                is_effect=False)

        qa_embed_data_B = qa_embed_data_B * src_mask
        q_embed_data_B = q_embed_data_B * src_mask

        # 6 cal the contrastive loss
        cl_input_states = F.cosine_similarity(qa_embed_data_ori, qa_embed_data_A, dim=-1) / self.temprerature
        cl_hard_neg_states = F.cosine_similarity(qa_embed_data_ori, qa_embed_data_B, dim=-1) / self.temprerature
        cl_loss_states = (-torch.log(cl_input_states.exp() / (cl_hard_neg_states.exp() + cl_input_states.exp()))).mean()

        cl_input_concepts = F.cosine_similarity(q_embed_data_ori, q_embed_data_A, dim=-1) / self.temprerature
        cl_hard_neg_concepts = F.cosine_similarity(q_embed_data_ori, q_embed_data_B, dim=-1) / self.temprerature
        cl_loss_concepts = (-torch.log(cl_input_concepts.exp() / (cl_hard_neg_concepts.exp() + cl_input_concepts.exp()))).mean()

        return cl_loss_states + cl_loss_concepts

    def forward(self, q_data, r_data, q_shft, sm, pid_data=None, pid_shft=None, qtest=False):  # c, r, cshft, q, qshft

        # 1 get the embeddings and l2_loss
        q_embed_data, qa_embed_data, pid_diff_data = self._embedding(q_data, r_data, q_shft, pid_data, pid_shft)
        c_reg_loss = (pid_diff_data ** 2.).sum() * self.l2  # rasch: l2_loss

        # 2 get the states and concepts by the sub-model: transformer
        states, concepts = self.model(q_embed_data, qa_embed_data)

        # 3 get the personal_states and ideal_states
        personal_states, ideal_states, personal_score = self._cal_personal_states(states)
        if qtest:
            return personal_states, ideal_states, personal_score

        # 4 cal the InfoNCE loss
        cl_loss = 0
        if self.training:
            loss = self._cal_InfoNCE(q_data, r_data, q_shft, sm, pid_data, pid_shft)
            cl_loss = loss * self.l_cl

        concat_q = torch.cat([personal_states, concepts], dim=-1)
        logit = self.out(concat_q).squeeze(-1)

        return logit, concat_q, c_reg_loss + cl_loss


class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, drop_out, n_blocks):
        super().__init__()

        # 1 Transformer
        self.knowledge_encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model, d_model, d_model, d_model, n_heads, d_ff, drop_out)
            for _ in range(n_blocks)
        ])
        self.states_retriever = nn.ModuleList([
            TransformerDecoderLayer(d_model, d_model, d_model, d_model, n_heads, d_ff, drop_out)
            for _ in range(n_blocks)
        ])

    def forward(self, q_embed_data, qa_embed_data):
        states = qa_embed_data
        concepts = q_embed_data
        seq_len = states.shape[1]

        # mha
        causal_mask = torch.triu(torch.ones((seq_len, seq_len)).to(device), 1) == 0

        for block in self.knowledge_encoder:
            states, states_attn = block(states, states, states, mask=causal_mask, is_effect=True)

        for block in self.states_retriever:
            concepts, concepts_attn = block(concepts, states, states, mask_sa=causal_mask, mask_mha=causal_mask, is_effect=True)

        return states, concepts

