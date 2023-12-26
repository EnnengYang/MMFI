import torch
import numpy as np
from itertools import combinations

class EmbeddingLayer(torch.nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class InnerProductLayer(torch.nn.Module):
    """ output: product_sum_pooling (bs x 1),
                Bi_interaction_pooling (bs * dim),
                inner_product (bs x f2/2),
                elementwise_product (bs x f2/2 x emb_dim)
    """
    def __init__(self, num_fields=None, output="product_sum_pooling"):
        super(InnerProductLayer, self).__init__()
        self.num_fields = num_fields
        self._output_type = output
        if output not in ["product_sum_pooling", "Bi_interaction_pooling", "inner_product", "elementwise_product", "all_product", "matrix_product"]:
            raise ValueError("InnerProductLayer output={} is not supported.".format(output))
        if num_fields is None:
            if output in ["inner_product", "elementwise_product", "all_product"]:
                raise ValueError("num_fields is required when InnerProductLayer output={}.".format(output))
        else:
            p, q = zip(*list(combinations(range(num_fields), 2)))
            self.field_p = torch.nn.Parameter(torch.LongTensor(p), requires_grad=False)
            self.field_q = torch.nn.Parameter(torch.LongTensor(q), requires_grad=False)
            self.interaction_units = int(num_fields * (num_fields - 1) / 2)
            self.upper_triange_mask = torch.nn.Parameter(torch.triu(torch.ones(num_fields, num_fields), 1).type(torch.ByteTensor),
                                                   requires_grad=False)
            all = torch.LongTensor(range(num_fields))
            l = all.unsqueeze(0).expand(num_fields, -1).clone()
            r = all.unsqueeze(1).expand(-1, num_fields).clone()
            self.field_l = torch.nn.Parameter(l, requires_grad=False)
            self.field_r = torch.nn.Parameter(r, requires_grad=False)
            self.interaction_units = int(num_fields * (num_fields - 1) / 2)

    def forward(self, feature_emb, matrix=None):
        if self._output_type in ["product_sum_pooling", "Bi_interaction_pooling"]:
            sum_of_square = torch.sum(feature_emb, dim=1) ** 2
            square_of_sum = torch.sum(feature_emb ** 2, dim=1)
            bi_interaction = (sum_of_square - square_of_sum) * 0.5
            if self._output_type == "Bi_interaction_pooling":
                return bi_interaction
            else:
                return bi_interaction.sum(dim=-1, keepdim=True)
        elif self._output_type == "elementwise_product":
            emb1 = torch.index_select(feature_emb, 1, self.field_p)
            emb2 = torch.index_select(feature_emb, 1, self.field_q)
            return emb1 * emb2
        elif self._output_type == "matrix_product" and matrix is not None:
            emb1 = torch.index_select(feature_emb, 1, self.field_p)
            emb1 = torch.matmul(emb1.unsqueeze(2), matrix).squeeze(2)
            emb2 = torch.index_select(feature_emb, 1, self.field_q)
            return emb1 * emb2
        elif self._output_type == "all_product":
            emb1 = [torch.index_select(feature_emb, 1, self.field_l[f]) for f in range(self.num_fields)]
            emb2 = [torch.index_select(feature_emb, 1, self.field_r[f]) for f in range(self.num_fields)]
            return [emb1[f] * emb2[f] for f in range(self.num_fields)]
        elif self._output_type == "inner_product":
            inner_product_matrix = torch.bmm(feature_emb, feature_emb.transpose(1, 2))
            flat_upper_triange = torch.masked_select(inner_product_matrix, self.upper_triange_mask)
            return flat_upper_triange.view(-1, self.interaction_units)

class LinearLayer(torch.nn.Module):
    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias

class CrossInteractionLayer(torch.nn.Module):
    def __init__(self, input_dim):
        super(CrossInteractionLayer, self).__init__()
        self.weight = torch.nn.Linear(input_dim, 1, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(input_dim))

    def forward(self, X_0, X_i):
        interaction_out = self.weight(X_i) * X_0 + self.bias
        return interaction_out

class CrossNet(torch.nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNet, self).__init__()
        self.num_layers = num_layers
        self.cross_net = torch.nn.ModuleList(CrossInteractionLayer(input_dim) for _ in range(self.num_layers))

    def forward(self, X_0):
        X_i = X_0
        for i in range(self.num_layers):
            X_i = X_i + self.cross_net[i](X_0, X_i)
        return X_i

class CrossNetV2(torch.nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNetV2, self).__init__()
        self.num_layers = num_layers
        self.cross_layers = torch.nn.ModuleList(torch.nn.Linear(input_dim, input_dim) for _ in range(self.num_layers))

    def forward(self, X_0):
        X_i = X_0
        for i in range(self.num_layers):
            X_i = X_i + X_0 * self.cross_layers[i](X_i)
        return X_i