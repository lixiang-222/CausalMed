import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F

class hetero_effect_graph(nn.Module):
    def __init__(self, in_channels, out_channels, device, diag_med_levels=5, proc_med_levels=5):
        super(hetero_effect_graph, self).__init__()

        self.device = device

        # 等级数量，用于划分不同权重级别的边
        self.diag_med_levels = diag_med_levels
        self.proc_med_levels = proc_med_levels

        # 总关系数 最后加2代表生成一种虚拟的边 用于没有药物嵌入的时候
        num_relations = diag_med_levels + proc_med_levels + 2

        # 边类型映射字典
        self.edge_type_mapping = {}
        self.initialize_edge_type_mapping(num_relations)

        # 定义两个RGCN卷积层
        self.conv1 = RGCNConv(in_channels, out_channels, num_relations)
        self.conv2 = RGCNConv(out_channels, out_channels, num_relations)

    def initialize_edge_type_mapping(self, num_relations):
        # 分配整数值给每种边类型
        j = 0
        for i in range(self.diag_med_levels + 1):
            edge_type = ('Med', f'connected_to_diag_{i}', 'Diag')
            self.edge_type_mapping[edge_type] = j
            j += 1

        for i in range(self.proc_med_levels + 1):
            edge_type = ('Med', f'connected_to_proc_{i}', 'Proc')
            self.edge_type_mapping[edge_type] = j
            j += 1

    def create_hetero_graph(self, diag_emb, proc_emb, med_emb, diag_med_weights, proc_med_weights):
        # 创建异构图数据结构
        data = HeteroData()

        # 分配节点嵌入
        data['Diag'].x = diag_emb.squeeze(0)
        data['Proc'].x = proc_emb.squeeze(0)
        data['Med'].x = med_emb.squeeze(0)

        # 如果全部是0向量不用分层
        if np.all(diag_med_weights == 0):
            src = torch.zeros(diag_med_weights.size, dtype=torch.int64)
            dst = torch.arange(0, diag_med_weights.size, dtype=torch.int64)
            edge_index = torch.stack([src, dst])
            data['Med', f'connected_to_diag_{0}', 'Diag'].edge_index = edge_index
        else:
            # 根据权重为关系分配不同的关系类型
            for i in range(1, self.diag_med_levels + 1):
                mask = (diag_med_weights > (i / self.diag_med_levels)) & \
                       (diag_med_weights <= ((i + 1) / self.diag_med_levels))
                edge_index = torch.from_numpy(np.vstack(mask.nonzero()))

                if edge_index.size(0) > 0:
                    # 不需要具体的权重，知道属于第几类边就可以了
                    # data[f'Med', f'connected_to_diag_{i}', 'Diag'].edge_attr = diag_med_weights[edge_index[0], edge_index[1]]
                    edge_index = edge_index.flip([0])
                    data['Med', f'connected_to_diag_{i}', 'Diag'].edge_index = edge_index

        # 如果全部是0向量不用分层
        if np.all(proc_med_weights == 0):
            src = torch.zeros(proc_med_weights.size, dtype=torch.int64)
            dst = torch.arange(0, proc_med_weights.size, dtype=torch.int64)
            edge_index = torch.stack([src, dst])
            data['Med', f'connected_to_proc_{0}', 'Proc'].edge_index = edge_index
        else:
            # 根据权重为关系分配不同的关系类型
            for i in range(1, self.proc_med_levels + 1):
                mask = (proc_med_weights > (i / self.proc_med_levels)) & \
                       (proc_med_weights <= ((i + 1) / self.proc_med_levels))
                edge_index = torch.from_numpy(np.vstack(mask.nonzero()))

                if edge_index.size(0) > 0:
                    # 不需要具体的权重，知道属于第几类边就可以了
                    # data[f'Med', f'connected_to_proc_{i}', 'Proc'].edge_attr = proc_med_weights[edge_index[0], edge_index[1]]
                    edge_index = edge_index.flip([0])
                    data['Med', f'connected_to_proc_{i}', 'Proc'].edge_index = edge_index

        return data

    def hetero_to_homo(self, data):
        # 统一编码所有节点
        diag_offset = 0
        proc_offset = diag_offset + data['Diag'].x.size(0)
        med_offset = proc_offset + data['Proc'].x.size(0)

        # 合并所有节点特征，x_all是所有节点的嵌入
        x_all = torch.cat([data['Diag'].x, data['Proc'].x, data['Med'].x], dim=0)

        # 创建整张图的edge_index和edge_type
        edge_index_list = []
        edge_type_list = []

        # range+1为了适配虚拟类
        for i in range(self.diag_med_levels + 1):
            key = ('Med', f'connected_to_diag_{i}', 'Diag')
            if key in data.edge_types:
                src, dst = data[key].edge_index
                edge_index_list.append(torch.stack([src + med_offset, dst + diag_offset], dim=0))
                edge_type_list.append(torch.full((len(src),), self.edge_type_mapping[key]))

        # range+1为了适配虚拟类
        for i in range(self.proc_med_levels + 1):
            key = ('Med', f'connected_to_proc_{i}', 'Proc')
            if key in data.edge_types:
                src, dst = data[key].edge_index
                edge_index_list.append(torch.stack([src + med_offset, dst + proc_offset], dim=0))
                edge_type_list.append(torch.full((len(src),), self.edge_type_mapping[key]))

        # Concatenate edge_index from different edge types
        edge_index = torch.cat(edge_index_list, dim=1).to(self.device)

        # Concatenate edge_type from different edge types
        edge_type = torch.cat(edge_type_list, dim=0).to(self.device)

        return x_all, edge_index, edge_type

    def forward(self, emb_diag, emb_proc, emb_med, diag_med_weights, proc_med_weights):
        # 创建异构图
        data = self.create_hetero_graph(emb_diag, emb_proc, emb_med, diag_med_weights, proc_med_weights)

        # 从异构图转换到同构图
        x, edge_index, edge_type = self.hetero_to_homo(data)

        # 卷积
        out1 = self.conv1(x, edge_index, edge_type)
        out1 = F.relu(out1)
        out = self.conv2(out1, edge_index, edge_type)

        # 根据偏移量切割张量，分解出每种类型的嵌入
        diag_offset = 0
        proc_offset = diag_offset + data['Diag'].x.size(0)
        med_offset = proc_offset + data['Proc'].x.size(0)

        out_emb_diag = out[diag_offset:proc_offset]
        out_emb_proc = out[proc_offset:med_offset]
        out_emb_med = out[med_offset:]

        return out_emb_diag.unsqueeze(0), out_emb_proc.unsqueeze(0), out_emb_med.unsqueeze(0)


if __name__ == '__main__':
    # 示例数据
    torch.manual_seed(1203)
    np.random.seed(2048)

    diag_emb = torch.randn(5, 8)
    proc_emb = torch.randn(3, 8)
    med_emb = torch.randn(4, 8)

    diag_med_weights = np.random.rand(5, 4)
    proc_med_weights = np.random.rand(3, 4)

    # 创建模型并计算输出
    model = hetero_effect_graph(8, 8, torch.device("cpu"))
    out = model(diag_emb, proc_emb, med_emb, diag_med_weights, proc_med_weights)
