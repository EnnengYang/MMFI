import numpy as np
import torch
from torch import nn
from .layers import EmbeddingLayer, MultiLayerPerceptron, InnerProductLayer, LinearLayer

class MMFIModel(nn.Module):
    def __init__(self, task_num, embed_dim, numerical_num, categorical_field_dims, expert_num=4, user_numerical_dim=23, item_numerical_dim=40, scenario_num=4, scenario_indicator_dim=4, task_indicator_dim=4, hard_sparse=True):
        super(MMFIModel, self).__init__()
        self.embedding = EmbeddingLayer(np.delete(categorical_field_dims, [0]), embed_dim)
        self.user_numerical_layer = torch.nn.Linear(user_numerical_dim, embed_dim)
        self.item_numerical_layer = torch.nn.Linear(item_numerical_dim, embed_dim)

        self.user_numerical_linear = torch.nn.Linear(user_numerical_dim, embed_dim)
        self.item_numerical_linear = torch.nn.Linear(item_numerical_dim, embed_dim)

        self.interact_dim = int((len(categorical_field_dims)+1) * len(categorical_field_dims) / 2)
        self.num_fields = int(len(categorical_field_dims)+1)
        self.embed_dim = embed_dim
        self.embed_output_dim = embed_dim*2
        self.user_numerical_dim = user_numerical_dim
        self.item_numerical_dim = item_numerical_dim

        self.scenario_num = scenario_num
        self.task_num = task_num
        self.scenario_embedding = EmbeddingLayer([self.scenario_num], scenario_indicator_dim)
        self.task_embedding = EmbeddingLayer([task_num], task_indicator_dim)

        self.scenario_task_weights = nn.Parameter(torch.empty((task_num, scenario_num, 1), dtype=torch.float32))
        nn.init.xavier_normal_(self.scenario_task_weights)

        self.interact_weights = nn.Parameter(torch.empty((1, self.interact_dim, 1), dtype=torch.float32))
        nn.init.xavier_normal_(self.interact_weights)

        self.scenario_linear = LinearLayer([self.scenario_num], output_dim=scenario_indicator_dim)
        self.task_linear = LinearLayer([task_num], output_dim=task_indicator_dim)

        self.product_layer = InnerProductLayer(num_fields=int(len(categorical_field_dims)-1+2), output='elementwise_product')

        self.linear_layer = EmbeddingLayer(np.delete(categorical_field_dims, [0]), embed_dim)
        self.bias = torch.nn.Parameter(torch.zeros((self.task_num, scenario_num, embed_dim), requires_grad=True))

        self.hard_sparse = hard_sparse
        self.mid_size = 4

        self.share_bottom = nn.Linear(embed_dim, 14)
        self.expert_num = expert_num
        self.linear_share_bottom = nn.Linear(embed_dim, 14)
        self.share_experts = nn.ModuleList([nn.Linear(14, self.mid_size) for _ in range(self.expert_num)])
        self.specific_experts = nn.ModuleList(
            [nn.ModuleList([nn.ModuleList([nn.Linear(14, self.mid_size) for _ in range(int(self.expert_num / 2))])
                            for _ in range(self.task_num)]) for _ in range(self.scenario_num)])
        self.linear_share_experts = nn.ModuleList([nn.Linear(14, self.mid_size) for _ in range(self.expert_num)])
        self.linear_specific_experts = nn.ModuleList([nn.ModuleList([nn.ModuleList([nn.Linear(14, self.mid_size)
                                                       for _ in range(int(self.expert_num / 2))])
                                                       for _ in range(self.task_num)]) for _ in range(self.scenario_num)])

        self.gates = nn.ModuleList([nn.ModuleList([nn.Sequential(
            nn.Linear(scenario_indicator_dim + embed_dim, int(3 * self.expert_num / 2)),
            nn.Softmax(dim=2)) for _ in range(self.task_num)]) for _ in range(scenario_num)])
        self.linear_gates = nn.ModuleList([nn.ModuleList([nn.Sequential(
            nn.Linear(scenario_indicator_dim + embed_dim, int(3 * self.expert_num / 2)),
            nn.Softmax(dim=2)) for _ in range(self.task_num)]) for _ in range(scenario_num)])
        self.linear_tower = nn.ModuleList([nn.ModuleList(
            [nn.Sequential(nn.Linear(self.mid_size * self.num_fields, self.num_fields), nn.BatchNorm1d(self.num_fields),
                           temperature(1), nn.Softmax(dim=1)) for _ in range(self.task_num)]) for _ in range(self.scenario_num)])

        if hard_sparse:
            self.masks = nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Linear(self.mid_size * self.interact_dim, self.interact_dim),
                                                                     nn.BatchNorm1d(self.interact_dim), nn.Tanh(),
                                                                     nn.ReLU()) for _ in range(task_num)]) for _ in range(self.scenario_num)])
            self.attention_tower = nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Linear(self.interact_dim, 1, bias=False))
                                                                 for _ in range(self.task_num)]) for _ in range(self.scenario_num)])
        else:
            self.attention_tower = nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Linear(self.mid_size * self.interact_dim, self.interact_dim),
                                                                               nn.BatchNorm1d(self.interact_dim),
                               temperature(1), nn.Softmax(dim=1)) for _ in range(self.task_num)]) for _ in range(self.scenario_num)])

        self.last = nn.ModuleList([nn.Sequential(nn.Linear(self.embed_output_dim, 40), nn.BatchNorm1d(40), nn.ReLU(),
                                                 nn.Linear(40, 32), nn.BatchNorm1d(32), nn.ReLU(),
                                                 nn.Linear(32, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 1),
                                                 nn.Sigmoid()) for _ in range(self.task_num)])

    def forward(self, categorical_x, numerical_x, device):
        batch_size = categorical_x.shape[0]
        task_indicators = []
        scenario_task_interactions = []
        task_ids = []

        scenario_ids = categorical_x[:, 0].unsqueeze(1).to(device)
        scenario_idxs = [nn.functional.one_hot(scenario_ids.clone().detach(), self.scenario_num).to(device)]

        scenario_indicator = self.scenario_embedding(scenario_ids).squeeze()

        for t in range(self.task_num):
            task_id = torch.ones([batch_size, 1], dtype=torch.long) * t
            task_id = task_id.to(device)
            task_ids.append(task_id)

            task_indicator = self.task_embedding(task_id).squeeze()
            task_indicators.append(task_indicator)
            scenario_task_weights = torch.einsum("ij, jk->ik",  [torch.squeeze(scenario_idxs[0].float()), self.scenario_task_weights[t]])
            scenario_task_interactions.append(scenario_indicator * task_indicator * scenario_task_weights)

        categorical_emb = self.embedding(categorical_x[:, 1:])
        user_numerical_emb = self.user_numerical_layer(numerical_x[:, :self.user_numerical_dim]).unsqueeze(1)
        item_numerical_emb = self.item_numerical_layer(numerical_x[:, self.user_numerical_dim:]).unsqueeze(1)
        categorical_emb = torch.cat([categorical_emb, user_numerical_emb, item_numerical_emb], dim=1)
        categorical_emb = self.product_layer(categorical_emb) * self.interact_weights

        attention_inputs = [torch.cat([(scenario_task_interactions[t]).unsqueeze(1).expand(-1, self.interact_dim, -1),
                                       categorical_emb], dim=2) for t in range(self.task_num)]

        # interaction
        experts_in = self.share_bottom(categorical_emb)
        share_out = torch.stack([self.share_experts[e](experts_in) for e in range(int(self.expert_num))])
        specific_out = [torch.stack([torch.stack([self.specific_experts[s][t][e](experts_in) for e in range(int(self.expert_num / 2))])
                                    for t in range(self.task_num)]) for s in range(self.scenario_num)]
        attention_weights = [[torch.cat([share_out, specific_out[s][t]], dim=0)
                              for t in range(self.task_num)] for s in range(self.scenario_num)]
        attention_gates = [torch.stack([self.gates[s][t](attention_inputs[t]) for t in range(self.task_num)])
                           for s in range(self.scenario_num)]
        attention_gates = [[attention_gates[s][t].permute(2, 0, 1) for t in range(self.task_num)]
                           for s in range(self.scenario_num)]
        attention_weights = [[torch.sum(attention_weights[s][t] * attention_gates[s][t].unsqueeze(3), dim=0)
                              for t in range(self.task_num)] for s in range(self.scenario_num)]

        if self.hard_sparse:
            attention_weights = [[self.masks[s][t](attention_weights[s][t].view(-1, self.mid_size * self.interact_dim))
                                  for t in range(self.task_num)] for s in range(self.scenario_num)]
            attention_weights = [[(attention_weights[s][t].unsqueeze(2) * categorical_emb).transpose(1, 2) for t in range(self.task_num)]
                                   for s in range(self.scenario_num)]
            attention_weights = [torch.stack([self.attention_tower[s][t](attention_weights[s][t]).squeeze()
                                              for s in range(self.scenario_num)]) for t in range(self.task_num)]
            interaction_out = [torch.einsum('ji, ijk->jk', [scenario_idxs[0].squeeze().float(), attention_weights[t]])
                             for t in range(self.task_num)]

        else:
            attention_weights = [torch.stack([self.attention_tower[s][t](attention_weights[s][t].view(-1, self.mid_size * self.interact_dim))
                                 for s in range(self.scenario_num)]) for t in range(self.task_num)]
            attention_weights = [torch.einsum('ji, ijk->jk', [scenario_idxs[0].squeeze().float(), attention_weights[t]])
                                 for t in range(self.task_num)]
            interaction_out = [torch.sum(attention_weights[t].unsqueeze(2) * categorical_emb, dim=1) for t in  range(self.task_num)]

        linear_out = self.linear_layer(categorical_x[:, 1:])
        user_numerical_linear = self.user_numerical_linear(numerical_x[:, :self.user_numerical_dim]).unsqueeze(1)
        item_numerical_linear = self.item_numerical_linear(numerical_x[:, self.user_numerical_dim:]).unsqueeze(1)
        linear_out = torch.cat([linear_out, user_numerical_linear, item_numerical_linear], dim=1)

        attention_inputs_ = [torch.cat([(scenario_task_interactions[t]).unsqueeze(1).expand(-1, self.num_fields, -1),
                                        linear_out], dim=2) for t in range(self.task_num)]

        # first order
        experts_in_ = self.linear_share_bottom(linear_out)
        share_out_ = torch.stack([self.linear_share_experts[e](experts_in_) for e in
                                  range(int(self.expert_num))])
        specific_out_ = [torch.stack([torch.stack([self.linear_specific_experts[s][t][e](experts_in_) for e in
                                                   range(int(self.expert_num / 2))]) for t in range(self.task_num)]) for
                                                    s in range(self.scenario_num)]
        attention_weights_ = [[torch.cat([share_out_, specific_out_[s][t]], dim=0) for t in range(self.task_num)]
                              for s in range(self.scenario_num)]
        attention_gates_ = [torch.stack([self.gates[s][t](attention_inputs_[t]) for t in range(self.task_num)])
                            for s in range(self.scenario_num)]
        attention_gates_ = [[attention_gates_[s][t].permute(2, 0, 1) for t in range(self.task_num)]
                            for s in range(self.scenario_num)]
        attention_weights_ = [[torch.sum(attention_weights_[s][t] * attention_gates_[s][t].unsqueeze(3), dim=0)
                               for t in range(self.task_num)] for s in range(self.scenario_num)]
        attention_weights_ = [torch.stack([self.linear_tower[s][t](attention_weights_[s][t].view(-1, self.num_fields * self.mid_size))
                                for s in range(self.scenario_num)]) for t in range(self.task_num)]
        attention_weights_ = [torch.einsum('ji, ijk->jk', [scenario_idxs[0].squeeze().float(), attention_weights_[t]])
                              for t in range(self.task_num)]
        linear_out = [torch.sum(attention_weights_[t].unsqueeze(2) * linear_out, dim=1) for t in range(self.task_num)]

        bias = [torch.einsum("ij, jk->ik", [torch.squeeze(scenario_idxs[0].float()), self.bias[t]])
                for t in range(self.task_num)]

        linear_out = [linear_out[t] + bias[t] for t in range(self.task_num)]
        embeddings = [torch.cat([interaction_out[t], linear_out[t]], 1) for t in range(self.task_num)]
        outputs = [t(ti).squeeze() for (t, ti) in zip(self.last, embeddings)]
        return outputs

class squeeze(nn.Module):
    def __init__(self):
        super(squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze()

class temperature(nn.Module):
    def __init__(self, tem):
        super(temperature, self).__init__()
        self.tem = tem

    def forward(self, x):
        return x/self.tem