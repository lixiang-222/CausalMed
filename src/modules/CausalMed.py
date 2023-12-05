import numpy as np
import torch
import torch.nn as nn

from .hetero_effect_graph import hetero_effect_graph
from .homo_relation_graph import homo_relation_graph


class CausalMed(torch.nn.Module):
    def __init__(
            self,
            causal_graph,
            tensor_ddi_adj,
            emb_dim,
            voc_size,
            dropout,
            device=torch.device('cpu'),
    ):
        super(CausalMed, self).__init__()
        self.device = device
        self.emb_dim = emb_dim

        # Embedding of all entities
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(voc_size[0], emb_dim),
            torch.nn.Embedding(voc_size[1], emb_dim),
            torch.nn.Embedding(voc_size[2], emb_dim)
        ])

        if dropout > 0 and dropout < 1:
            self.rnn_dropout = torch.nn.Dropout(p=dropout)
        else:
            self.rnn_dropout = torch.nn.Sequential()

        self.causal_graph = causal_graph

        self.homo_graph = nn.ModuleList([
            homo_relation_graph(emb_dim, device),
            homo_relation_graph(emb_dim, device),
            homo_relation_graph(emb_dim, device)
        ])

        self.hetero_graph = hetero_effect_graph(emb_dim, emb_dim, device)

        # Isomeric and isomeric addition parameters
        self.rho = nn.Parameter(torch.ones(3, 2))

        self.seq_encoders = torch.nn.ModuleList([
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True)
        ])

        # Convert patient information to drug score
        self.query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 6, voc_size[2])
        )

        self.tensor_ddi_adj = tensor_ddi_adj
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

    def forward(self, patient_data):

        seq_diag, seq_proc, seq_med = [], [], []
        for adm_id, adm in enumerate(patient_data):
            idx_diag = torch.LongTensor(adm[0]).to(self.device)
            idx_proc = torch.LongTensor(adm[1]).to(self.device)
            emb_diag = self.rnn_dropout(self.embeddings[0](idx_diag)).unsqueeze(0)
            emb_proc = self.rnn_dropout(self.embeddings[1](idx_proc)).unsqueeze(0)

            # Isomorphic graph hierarchical representation
            graph_diag = self.causal_graph.get_graph(adm[3], "Diag")
            graph_proc = self.causal_graph.get_graph(adm[3], "Proc")
            emb_diag1 = self.homo_graph[0](graph_diag, emb_diag)
            emb_proc1 = self.homo_graph[1](graph_proc, emb_proc)

            if adm == patient_data[0]:
                emb_med1 = torch.zeros(1, 1, self.emb_dim).to(self.device)
            else:
                adm_last = patient_data[adm_id - 1]
                idx_med = torch.LongTensor([adm_last[2]]).to(self.device)
                emb_med = self.rnn_dropout(self.embeddings[2](idx_med))
                med_graph = self.causal_graph.get_graph(adm_last[3], "Med")
                emb_med1 = self.homo_graph[2](med_graph, emb_med)

            # Heterogeneous graph representation
            if adm == patient_data[0]:
                idx_med = torch.LongTensor([5000]).to(self.device)  # 用5000作为一个占位值，表示没有节点
                emb_med = torch.zeros(1, 1, self.emb_dim).to(self.device)
            else:
                adm_last = patient_data[adm_id - 1]
                idx_med = torch.LongTensor(adm_last[2]).to(self.device)
                emb_med = self.rnn_dropout(self.embeddings[2](idx_med)).unsqueeze(0)

            # Initialize two two-dimensional arrays to zero
            diag_med_weights = np.zeros((len(idx_diag), len(idx_med)))
            proc_med_weights = np.zeros((len(idx_proc), len(idx_med)))

            if 5000 not in idx_med:
                # padding diag_med_weights
                for i, med in enumerate(idx_med):
                    for j, diag in enumerate(idx_diag):
                        effect = self.causal_graph.get_effect(diag, med, "Diag", "Med")
                        diag_med_weights[j, i] = effect

                # padding proc_med_weights
                for i, med in enumerate(idx_med):
                    for j, proc in enumerate(idx_proc):
                        effect = self.causal_graph.get_effect(proc, med, "Proc", "Med")
                        proc_med_weights[j, i] = effect

            emb_diag2, emb_proc2, emb_med2 = \
                self.hetero_graph(emb_diag, emb_proc, emb_med, diag_med_weights, proc_med_weights)

            emb_diag3 = self.rho[0, 0] * emb_diag1 + self.rho[0, 1] * emb_diag2
            emb_proc3 = self.rho[1, 0] * emb_proc1 + self.rho[1, 1] * emb_proc2
            emb_med3 = self.rho[2, 0] * emb_med1 + self.rho[2, 1] * emb_med2

            seq_diag.append(torch.sum(emb_diag3, keepdim=True, dim=1))
            seq_proc.append(torch.sum(emb_proc3, keepdim=True, dim=1))
            seq_med.append(torch.sum(emb_med3, keepdim=True, dim=1))

        seq_diag = torch.cat(seq_diag, dim=1)
        seq_proc = torch.cat(seq_proc, dim=1)
        seq_med = torch.cat(seq_med, dim=1)
        output_diag, hidden_diag = self.seq_encoders[0](seq_diag)
        output_proc, hidden_proc = self.seq_encoders[1](seq_proc)
        output_med, hidden_med = self.seq_encoders[2](seq_med)
        seq_repr = torch.cat([hidden_diag, hidden_proc, hidden_med], dim=-1)
        last_repr = torch.cat([output_diag[:, -1], output_proc[:, -1], output_med[:, -1]], dim=-1)
        patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()])

        score = self.query(patient_repr).unsqueeze(0)

        neg_pred_prob = torch.sigmoid(score)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return score, batch_neg
