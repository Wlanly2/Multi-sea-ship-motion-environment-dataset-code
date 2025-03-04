import numpy as np
import torch
import torch.nn as nn


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),dtype=np.int32)
    return labels_onehot



def edge2node(self, x, rel_rec):
    # NOTE: Assumes that we have the same graph across all samples.
    incoming = torch.matmul(rel_rec.t(), x)
    return incoming / incoming.size(1)


def node2edge(self, x, rel_rec, rel_send):
    # NOTE: Assumes that we have the same graph across all samples.
    receivers = torch.matmul(rel_rec, x)#128,42,50
    senders = torch.matmul(rel_send, x)#128,42,50
    edges = torch.cat([senders, receivers], dim=2)#128,42,100
    return edges


off_diag = np.ones([9, 9])
rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)

