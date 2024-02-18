import torch
from torch.nn.init import xavier_normal_


class ComplexEx(torch.nn.Module):
    def __init__(self, param):
        super(ComplexEx, self).__init__()
        self.name = "ComplexEx"
        self.param = param
        self.embedding_dim = self.param["embedding_dim"]
        self.num_entities = self.param["num_entities"]
        self.num_relations = self.param["num_relations"]

        # Embeddings of Entity and relations
        self.Er = torch.nn.Embedding(
            self.num_entities, self.embedding_dim, padding_idx=0
        )
        self.Rr = torch.nn.Embedding(
            self.num_relations, self.embedding_dim, padding_idx=0
        )
        self.Ei = torch.nn.Embedding(
            self.num_entities, self.embedding_dim, padding_idx=0
        )
        self.Ri = torch.nn.Embedding(
            self.num_relations, self.embedding_dim, padding_idx=0
        )

        self.input_dropout = torch.nn.Dropout(self.param["input_dropout"])
        self.bn0 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn1 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.loss = torch.nn.BCELoss()

    def initialize_weights(self):
        xavier_normal_(self.Er.weight.data)
        xavier_normal_(self.Rr.weight.data)
        xavier_normal_(self.Ei.weight.data)
        xavier_normal_(self.Ri.weight.data)

    def forward_subject_batch(self, e1_idx, rel_idx):
        e1r = self.Er(e1_idx)  # 1xD
        e1i = self.Ei(e1_idx)
        rr = self.Rr(rel_idx)
        ri = self.Ri(rel_idx)
        e1r = self.bn0(e1r)
        e1r = self.input_dropout(e1r)
        e1i = self.bn1(e1i)
        e1i = self.input_dropout(e1i)
        # (1xD)x(DxN) = (1xN)
        pred = (
            torch.mm(e1r * rr, self.Er.weight.transpose(1, 0))
            + torch.mm(e1r * ri, self.Ei.weight.transpose(1, 0))
            + torch.mm(e1i * rr, self.Ei.weight.transpose(1, 0))
            - torch.mm(e1i * ri, self.Er.weight.transpose(1, 0))
        )
        pred = torch.sigmoid(pred)
        return pred

    def forward_subject_and_loss(self, e1_idx, rel_idx, targets):
        return self.loss(
            self.forward_subject_batch(e1_idx=e1_idx, rel_idx=rel_idx), targets
        )

    def forward_triples(self, e1_idx, rel_idx, e2_idx):
        # print("E1,", e2_idx.is_cuda)
        # print("capE", self.Er.weight.is_cuda)
        e1r = self.Er(e1_idx)
        # print("E1R", e1r.is_cuda)
        rr = self.Rr(rel_idx)
        e1i = self.Ei(e1_idx)
        ri = self.Ri(rel_idx)
        e1r = self.bn0(e1r)
        e1r = self.input_dropout(e1r)
        e1i = self.bn1(e1i)
        e1i = self.input_dropout(e1i)

        e2r = self.Er(e2_idx)
        e2i = self.Ei(e2_idx)

        real_real_real = (e1r * rr * e2r).sum(dim=1)
        real_imag_imag = (e1r * ri * e2i).sum(dim=1)
        imag_real_imag = (e1i * rr * e2i).sum(dim=1)
        imag_imag_real = (e1i * ri * e2r).sum(dim=1)

        pred = real_real_real + real_imag_imag + imag_real_imag - imag_imag_real
        pred = torch.sigmoid(pred)
        return pred

    def forward_triples_and_loss(self, *args, **kwargs):
        raise NotImplementedError("Negative Sampling is not implemented for Complex")
