import torch
from torch.nn.init import xavier_normal_
from torch import nn


class ComplexEx(torch.nn.Module):
    def __init__(self, param):
        super(ComplexEx, self).__init__()
        self.dropout = torch.nn.Dropout(param["dropout"])
        self.embedding_dim = param["embedding_dim"]
        self.num_entities = param["num_entities"]
        self.num_relations = param["num_relations"]

        ##Embeddings
        self.Er = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.Ei = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.Rr = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        self.Ri = torch.nn.Embedding(self.num_relations, self.embedding_dim)

        self.bn_r = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_i = torch.nn.BatchNorm1d(self.embedding_dim)
        self.loss = torch.nn.BCELoss()  # Binary Cross Entropy

    def initialize_weights(self):
        xavier_normal_(self.Er.weight.data)
        xavier_normal_(self.Rr.weight.data)
        xavier_normal_(self.Ei.weight.data)
        xavier_normal_(self.Ri.weight.data)

    def forward(self, e1_idx, rel_idx, targets):
        e1r = self.Er(e1_idx)  # NxD
        e1i = self.Ei(e1_idx)
        rr = self.Rr(rel_idx)
        ri = self.Ri(rel_idx)

        # Regularization
        e1r = self.bn_r(e1r)
        e1r = self.dropout(e1r)
        e1i = self.bn_i(e1i)
        e1i = self.dropout(e1i)
        rr = self.bn_r(rr)
        rr = self.dropout(rr)
        ri = self.bn_i(ri)
        ri = self.dropout(ri)

        # NxD X DxNUM_ENTITY => NxNUM_ENTITY
        score = (
            torch.mm(e1r * rr, self.Er.weight.transpose(1, 0))
            + torch.mm(e1r * ri, self.Ei.weight.transpose(1, 0))
            + torch.mm(e1i * rr, self.Ei.weight.transpose(1, 0))
            - torch.mm(e1i * ri, self.Er.weight.transpose(1, 0))
        )
        logits = torch.nn.functional.softmax(score, dim=1)
        # logits  = torch.sigmoid(score)
        loss = self.loss(logits, targets)
        return logits, loss

    def predict(self, e1_idx, rel_idx, e2_idx, obj=True):
        e1r = self.Er(e1_idx)
        e1i = self.Ei(e1_idx)
        rr = self.Rr(rel_idx)
        ri = self.Ri(rel_idx)
        e2r = self.Er(e2_idx)
        e2i = self.Ei(e2_idx)

        # Regularization
        e1r = self.bn_r(e1r)
        e1r = self.dropout(e1r)
        e1i = self.bn_i(e1i)
        e1i = self.dropout(e1i)
        rr = self.bn_r(rr)
        rr = self.dropout(rr)
        ri = self.bn_i(ri)
        ri = self.dropout(ri)

        rr_r = (e1r * rr * e2r).sum(dim=1)
        ri_i = (e1r * ri * e2i).sum(dim=1)
        ir_i = (e1i * rr * e2i).sum(dim=1)
        ii_r = (e1i * ri * e2r).sum(dim=1)

        score = rr_r + ri_i + ir_i - ii_r
        logits = torch.nn.functional.softmax(score, dim=0)
        return logits
