import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import dgl
import dgl.function as fn
from dgl.nn.pytorch import SAGEConv, GATConv
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


def get_local_seq(full_seq, kernel_size, mean, std, device=torch.device('cpu')):
    seq_len = full_seq.size()[1]

    indices = torch.LongTensor(seq_len).to(device)

    torch.arange(0, seq_len, out=indices)

    indices = Variable(indices, requires_grad=False)

    first_seq = torch.index_select(full_seq, dim=1, index=indices[kernel_size - 1:])
    second_seq = torch.index_select(full_seq, dim=1, index=indices[:-kernel_size + 1])

    local_seq = first_seq - second_seq

    local_seq = (local_seq - mean) / std

    return local_seq


class Attr(nn.Module):
    def __init__(self, embed_dims, data_feature):
        super(Attr, self).__init__()

        self.embed_dims = embed_dims
        self.data_feature = data_feature

        for name, dim_in, dim_out in self.embed_dims:
            self.add_module(name + '_em', nn.Embedding(dim_in, dim_out))

    def out_size(self):
        sz = 0
        for _, _, dim_out in self.embed_dims:
            sz += dim_out
        # append total distance
        return sz + 1

    def forward(self, batch):
        em_list = []
        for name, _, _ in self.embed_dims:
            embed = getattr(self, name + '_em')
            attr_t = batch[name]

            attr_t = torch.squeeze(embed(attr_t))

            em_list.append(attr_t)

        dist_mean = self.data_feature.get("dist_mean", 9.578281194509781)
        dist_std = self.data_feature.get("dist_std", 3.9656010701306283)
        dist = (batch["dist"] - dist_mean) / dist_std
        dist = (dist - dist_mean) / dist_std
        em_list.append(dist)

        return torch.cat(em_list, dim=1)


class DotProductPredictor(nn.Module):
    # def __init__(self, hidden_features, out_features, device=torch.device('cpu')):
    # super(DotProductPredictor, self).__init__()
    # self.pred = nn.Linear(out_features, out_features, bias=True)
    def forward(self, graph, h):
        # h是从5.1节的GNN模型中计算出的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h
            # -->得到边的特征
            graph.apply_edges(fn.u_mul_v('h', 'h', 'score'))
            # print("graph.edata['score'] =", graph.edata['score'], graph.edata['score'].shape)
            # print.show()
            return graph.edata['score']


class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, device=torch.device('cpu')):
        super(Model, self).__init__()
        self.device = device
        # 第1层节点特征
        self.sage_1 = SAGEConv(in_features, out_features, 'pool')
        # self.gat_1 = GATConv((in_features, in_features), out_features, num_heads=8)
        # self.gat = GATConv((in_feature, out_feature), out_feature, num_heads=8)
        # 第2层节点特征
        # self.sage_2 = SAGEConv(hidden_features, out_features, 'pool')
        # 边的聚合操作
        self.sage_3 = SAGEConv(out_features, out_features, 'pool')
        # self.sage_3 = GATConv(out_features, out_features)
        # self.sage_4 = SAGEConv(hidden_features, out_features, 'pool')
        self.pred = DotProductPredictor()

    def forward(self, g, x, flag):
        if flag == 'node':
            h = self.sage_1(g, x)
            return self.pred(g, h)
        else:
            h = self.sage_3(g, x)
            # h = self.sage_4(g, h1)
            return self.pred(g, h)


class Model_inGeo(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, device=torch.device('cpu')):
        super(Model_inGeo, self).__init__()
        self.device = device
        # 第1层节点特征
        self.sage_1 = SAGEConv(in_features, hidden_features, 'pool')
        # 第2层节点特征
        self.sage_2 = SAGEConv(hidden_features, out_features, 'pool')

    def forward(self, g, x):
        h1 = self.sage_1(g, x)
        h = self.sage_2(g, h1)
        return h


class GAT(nn.Module):
    def __init__(self, in_features, data_feature={}, device=torch.device('cpu')):
        super(GAT, self).__init__()
        self.data_feature = data_feature
        self.device = device
        self.gatconv1 = GATConv((in_features, in_features), in_features, 8, residual=True).to(self.device)

    def forward(self, batch, loc):
        # GPS序列长度
        # loc.size()[0]-->一个batch的样本数-->10个
        # loc.size()[1]-->一天gps序列节点个数-->不定
        # loc.size()[2]-->节点特征维度-->4个
        # print("loc =", loc, loc.size())
        nodelenght = loc.size()[1]  # 节点个数
        # print("nodelenght =", nodelenght)
        nodeid = np.arange(0, nodelenght)  # 节点ID
        src = nodeid[:len(nodeid) - 1]  # 源节点
        dst = nodeid[1:len(nodeid)]  # 目标节点
        u = src
        # print("u =", u)
        v = (dst - 1)
        # print("v =", v)
        # 同时建立反向边-->有向图
        # edge_pred_graph = dgl.graph((np.concatenate([src, dst]), np.concatenate([dst, src])))
        g = dgl.heterograph({('A', 'r', 'B'): (u, v)}).to(self.device)
        # print("loc =", loc, loc.shape)
        # print("loc[0] =", loc[0]) # --第0条gps序列
        # print("loc[0][0] =", loc[0][0]) # 第0条gps序列的第0个点
        # print("loc[1][1] =", loc[1][1])
        # print.show()
        for i in range(0, loc.size()[0]):
            # 建立点和边特征，以及边的标签-->直接用出来的节点特征
            # edge_pred_graph.ndata['feature'] = torch.randn(nodel enght, 4)
            # print("torch.randn(nodelenght, 4) =", torch.randn(nodelenght, 4))
            # 节点特征,第i条gps序列上点的特征
            # print("loc[i] =", loc[i], loc[i].shape)
            u_feat = loc[i][:len(nodeid) - 1].to(self.device)
            v_feat = loc[i][1:len(nodeid)].to(self.device)
            # we use gat to update destination
            res = self.gatconv1(g, (u_feat, v_feat)).to(self.device)

            # print("res =", res, res.size())
            res = res.mean(axis=1, keepdim=False)
            # print("res =", res, res.size())
            # print("loc[i][0] =", loc[i][0])
            origin = loc[i][0].clone()
            # print("origin =", origin)
            origin = origin.unsqueeze(0)
            # print("origin =", origin)
            pred = torch.cat((origin, res), 0)
            # print("pred =", pred, pred.size())
            # print("pred =", pred, pred.size())
            # print.show()
            if i == 0:
                # 初始值
                pred_new = pred
                # 增加一个维度
                pred_new = pred_new.unsqueeze(0)
                # print("pred_new =", pred_new, pred_new.shape)

            else:
                # 拼接特征
                # 增加一个维度
                pred = pred.unsqueeze(0)
                pred_new = torch.cat((pred_new, pred), 0)
            # dist_gap_mean = self.data_feature.get("dist_gap_mean", 0.274716042312)
            # dist_gap_std = self.data_feature.get("dist_gap_std", 0.127051674693)
            # current_dis = (batch["current_dis"] - dist_gap_mean) / dist_gap_std
            # calculate the dist for local paths
            # local_dist = get_local_seq(current_dis, 3, dist_gap_mean, dist_gap_std, self.device)
            # local_dist = torch.unsqueeze(local_dist, dim=2)

            # pred_new = torch.cat((pred_new, local_dist), dim=2)
            # print("pred_new =", pred_new, pred_new.shape)
            # print(show)
        # print("pred_new =", pred_new, pred_new.shape)
        # print.show
        return pred_new


class EdgeGAT(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, data_feature={}, device=torch.device('cpu')):
        super(EdgeGAT, self).__init__()
        self.data_feature = data_feature
        self.device = device
        self.gatconv1 = GATConv((in_features, in_features), in_features, 8, residual=True).to(self.device)
        self.pred = DotProductPredictor()

    def forward(self, batch, loc):
        # GPS序列长度
        # loc.size()[0]-->一个batch的样本数-->10个
        # loc.size()[1]-->一天gps序列节点个数-->不定
        # loc.size()[2]-->节点特征维度-->4个
        # print("loc =", loc, loc.size())
        nodelenght = loc.size()[1]  # 节点个数
        # print("nodelenght =", nodelenght)
        nodeid = np.arange(0, nodelenght)  # 节点ID
        src = nodeid[:len(nodeid) - 1]  # 源节点
        dst = nodeid[1:len(nodeid)]  # 目标节点
        u = src
        # print("u =", u)
        v = (dst - 1)
        # print("v =", v)
        # 同时建立反向边-->有向图
        # edge_pred_graph = dgl.graph((np.concatenate([src, dst]), np.concatenate([dst, src])))
        g = dgl.heterograph({('A', 'r', 'B'): (u, v)}).to(self.device)
        # print("loc =", loc, loc.shape)
        # print("loc[0] =", loc[0]) # --第0条gps序列
        # print("loc[0][0] =", loc[0][0]) # 第0条gps序列的第0个点
        # print("loc[1][1] =", loc[1][1])
        # print.show()
        for i in range(0, loc.size()[0]):
            # 建立点和边特征，以及边的标签-->直接用出来的节点特征
            # edge_pred_graph.ndata['feature'] = torch.randn(nodel enght, 4)
            # print("torch.randn(nodelenght, 4) =", torch.randn(nodelenght, 4))
            # 节点特征,第i条gps序列上点的特征
            # print("loc[i] =", loc[i], loc[i].shape)
            u_feat = loc[i][:len(nodeid) - 1].to(self.device)
            v_feat = loc[i][1:len(nodeid)].to(self.device)
            # we use gat to update destination
            res = self.gatconv1(g, (u_feat, v_feat)).to(self.device)

            # print("res =", res, res.size())
            res = res.mean(axis=1, keepdim=False)
            # print("res =", res, res.size())
            # print("loc[i][0] =", loc[i][0])
            origin = loc[i][0].clone()
            # print("origin =", origin)
            origin = origin.unsqueeze(0)
            # print("origin =", origin)
            pred = torch.cat((origin, res), 0)
            # print("pred =", pred, pred.size())
            # print("pred =", pred, pred.size())
            # print.show()
            self.pred(g, pred)
            if i == 0:
                # 初始值
                pred_new = pred
                # 增加一个维度
                pred_new = pred_new.unsqueeze(0)
                # print("pred_new =", pred_new, pred_new.shape)

            else:
                # 拼接特征
                # 增加一个维度
                pred = pred.unsqueeze(0)
                pred_new = torch.cat((pred_new, pred), 0)
            # dist_gap_mean = self.data_feature.get("dist_gap_mean", 0.274716042312)
            # dist_gap_std = self.data_feature.get("dist_gap_std", 0.127051674693)
            # current_dis = (batch["current_dis"] - dist_gap_mean) / dist_gap_std
            # calculate the dist for local paths
            # local_dist = get_local_seq(current_dis, 3, dist_gap_mean, dist_gap_std, self.device)
            # local_dist = torch.unsqueeze(local_dist, dim=2)

            # pred_new = torch.cat((pred_new, local_dist), dim=2)
            # print("pred_new =", pred_new, pred_new.shape)
            # print(show)
        # print("pred_new =", pred_new, pred_new.shape)
        # print.show
        return pred_new


# -->个人感觉局部距离起决定性作用？先验证一下！应该很有用<-加一下？-->有用
# GNN->路网图
# GeoCov->轨迹图
# 如何合并两者？需要进一步考虑
class GNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, data_feature={}, device=torch.device('cpu')):
        super(GNN, self).__init__()
        self.data_feature = data_feature
        self.device = device
        self.model = Model(in_features, hidden_features, out_features, device=self.device).to(device)
        # self.sage = SAGEConv(in_features, out_features, 'pool')
        # self.pred = DotProductPredictor()

    def forward(self, batch, loc, attr_t):
        '''
        'current_longi': 'float', 'current_lati': 'float',
        'current_tim': 'float', 'current_dis': 'float',
        'current_state': 'float',
        'uid': 'int',
        'weekid': 'int',
        'timeid': 'int',
        'dist': 'float',
        'time': 'float',
        '''
        # print("current_longi =", batch["current_longi"], batch["current_longi"].size())
        # print("current_lati =", batch["current_lati"], batch["current_lati"].size())
        # print("current_tim =", batch["current_tim"], batch["current_tim"].size())#

        time_gap_mean = self.data_feature.get("time_gap_mean", 43.8756927994)
        time_gap_std = self.data_feature.get("time_gap_std", 51.4811932987)
        # print("timeid =", batch["timeid"], batch["timeid"].size())

        current_tim = ((batch["timeid"] + batch["current_tim"]) - time_gap_mean) / time_gap_std
        # print("current_tim =", current_tim)
        # print(show)
        # current_tim = torch.unsqueeze(current_tim, dim=2)
        # print("current_tim =", current_tim, current_tim.size())

        # print("current_dis =", batch["current_dis"], batch["current_dis"].size())
        # print(show)
        # print("current_state =", batch["current_state"], batch["current_state"].size())
        # print("uid =", batch["uid"], batch["uid"].size())
        # print("weekid =", batch["weekid"], batch["weekid"].size())
        # print("timeid =", batch["timeid"], batch["timeid"].size())
        # print("dist =", batch["dist"], batch["dist"].size())
        # print("time =", batch["time"], batch["time"].size())

        # longi_mean = self.data_feature.get("longi_mean", 104.05810954320589)
        # longi_std = self.data_feature.get("longi_std", 0.04988770679679998)
        # 经度操作
        # current_longi = (batch["current_longi"] - longi_mean) / longi_std
        # print("current_longi =", current_longi, current_longi.shape)
        # lngs = torch.unsqueeze(current_longi, dim=2)
        # print("lngs =", lngs, lngs.shape)
        # print("loc =", loc, loc.size())
        # print.show

        # print("attr_t =", attr_t, attr_t.shape)
        # attr_t = torch.unsqueeze(attr_t, dim=1)
        # print("attr_t =", attr_t, attr_t.shape)
        # print("loc =", loc, loc.shape)

        # print("attr_t =", attr_t, attr_t.shape)
        # expand_attr_t = attr_t.expand(loc.size()[:2] + (attr_t.size()[-1],))
        # print("expand_attr_t =", expand_attr_t, expand_attr_t.shape)
        # loc = torch.cat((loc, current_tim), dim=2)
        # print("weekid =", batch["weekid"], batch["weekid"].size())
        # weekid = torch.unsqueeze(batch["weekid"], dim=1)
        # weekid = weekid.expand(loc.size()[:2] + (weekid.size()[-1],))
        # print("loc =", loc, loc.size())
        # loc = torch.cat((loc, weekid), dim=2)
        # print("loc =", loc, loc.size())
        # print.show()
        # concat the loc_conv and the attributes
        # -->一条GPS序列上的点的特征是一样的？

        # GPS序列长度
        # loc.size()[0]-->一个batch的样本数-->10个
        # loc.size()[1]-->一天gps序列节点个数-->不定
        # loc.size()[2]-->节点特征维度-->4个
        # print("loc =", loc, loc.size())
        nodelenght = loc.size()[1]  # 节点个数
        # print("nodelenght =", nodelenght)
        nodeid = np.arange(0, nodelenght)  # 节点ID
        src = nodeid[:len(nodeid) - 1]  # 源节点
        dst = nodeid[1:len(nodeid)]  # 目标节点
        # 同时建立反向边-->有向图
        # edge_pred_graph = dgl.graph((np.concatenate([src, dst]), np.concatenate([dst, src])))
        edge_pred_graph = dgl.graph((src, dst)).to(self.device)
        # print("loc =", loc, loc.shape)
        # print("loc[0] =", loc[0]) # --第0条gps序列
        # print("loc[0][0] =", loc[0][0]) # 第0条gps序列的第0个点
        # print("loc[1][1] =", loc[1][1])
        # print.show()
        for i in range(0, loc.size()[0]):
            # 建立点和边特征，以及边的标签-->直接用出来的节点特征
            # edge_pred_graph.ndata['feature'] = torch.randn(nodel enght, 4)
            # print("torch.randn(nodelenght, 4) =", torch.randn(nodelenght, 4))
            # 节点特征,第i条gps序列上点的特征
            # print("loc[i] =", loc[i], loc[i].shape)
            edge_pred_graph.ndata['feature'] = loc[i].to(self.device)
            # print("loc[i] =", loc[i])
            # print(show)
            # edge_pred_graph.edata['feature'] = torch.randn(1000, 10)
            # 标签得话，用实际时间
            # edge_pred_graph.edata['label'] = torch.randn((nodelenght-1)*2)
            node_features = edge_pred_graph.ndata['feature'].to(self.device)
            # edge_label = edge_pred_graph.edata['label']
            # train_mask = edge_pred_graph.edata['train_mask']

            # -->得到的是边的特征，其实边也可以看成点
            edge_pred = self.model(edge_pred_graph, node_features, flag='node')

            #######  边->图的操作
            new_nodelenght = edge_pred.size()[0]  # 边的条数->新构建图的节点个数
            # print("nodelenght =", new_nodelenght)
            new_nodeid = np.arange(0, new_nodelenght)  # 节点ID
            new_src = new_nodeid[:len(new_nodeid) - 1]  # 源节点
            new_dst = new_nodeid[1:len(new_nodeid)]  # 目标节点
            # 同时建立反向边-->有向图
            # edge_pred_graph = dgl.graph((np.concatenate([src, dst]), np.concatenate([dst, src])))
            new_edge_pred_graph = dgl.graph((new_src, new_dst)).to(self.device)

            new_edge_pred_graph.ndata['feature'] = edge_pred.to(self.device)
            # edge_pred_graph.edata['feature'] = torch.randn(1000, 10)
            # 标签得话，用实际时间
            # edge_pred_graph.edata['label'] = torch.randn((nodelenght-1)*2)
            new_node_features = new_edge_pred_graph.ndata['feature'].to(self.device)
            # print(new_node_features, new_node_features.size())
            # print.show()
            # -->得到的是边的特征，其实边也可以看成点
            pred = self.model(new_edge_pred_graph, new_node_features, flag='edge')
            if i == 0:
                # 初始值
                pred_new = pred
                # 增加一个维度
                pred_new = pred_new.unsqueeze(0)
                # print("pred_new =", pred_new, pred_new.shape)
            else:
                # 拼接特征
                # 增加一个维度
                pred = pred.unsqueeze(0)
                pred_new = torch.cat((pred_new, pred), 0)
                # print("pred_new =", pred_new, pred_new.shape)
                # print.show

        # print("pred_new =", pred_new, pred_new.shape)

        # print("pred_new =", pred_new, pred_new.shape)
        # 维度重新排序
        # pred_new = pred_new.permute(1, 0, 2)
        # print("pred_new =", pred_new, pred_new.shape)
        # print.show()
        # pred_new = pred_new.permute(0, 2, 1)
        dist_gap_mean = self.data_feature.get("dist_gap_mean", 0.274716042312)
        dist_gap_std = self.data_feature.get("dist_gap_std", 0.127051674693)
        current_dis = (batch["current_dis"] - dist_gap_mean) / dist_gap_std
        # calculate the dist for local paths
        local_dist = get_local_seq(current_dis, 3, dist_gap_mean, dist_gap_std, self.device)
        local_dist = torch.unsqueeze(local_dist, dim=2)
        # print("local_dist =", local_dist, local_dist.size())
        # print("pred_new =", pred_new, pred_new.size())
        # print.show()
        # 路网特征+路段距离
        # print("pred_new =", pred_new, pred_new.size())
        # print("local_dist =", local_dist, local_dist.size())
        pred_new = torch.cat((pred_new, local_dist), dim=2)
        return pred_new


class GNN_inGeo(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, data_feature={}, device=torch.device('cpu')):
        super(GNN_inGeo, self).__init__()
        self.data_feature = data_feature
        self.device = device
        self.model = Model_inGeo(in_features, hidden_features, out_features, device=self.device).to(device)
        # self.sage = SAGEConv(in_features, out_features, 'pool')
        # self.pred = DotProductPredictor()

    def forward(self, batch, loc):
        # GPS序列长度
        # loc.size()[0]-->一个batch的样本数-->10个
        # loc.size()[1]-->一天gps序列节点个数-->不定
        # loc.size()[2]-->节点特征维度-->4个
        nodelenght = loc.size()[1]  # 节点个数
        # print("nodelenght =", nodelenght)
        nodeid = np.arange(0, nodelenght)  # 节点ID
        src = nodeid[:len(nodeid) - 1]  # 源节点
        dst = nodeid[1:len(nodeid)]  # 目标节点
        # 同时建立反向边-->有向图
        # edge_pred_graph = dgl.graph((np.concatenate([src, dst]), np.concatenate([dst, src])))
        edge_pred_graph = dgl.graph((src, dst)).to(self.device)
        # print("loc =", loc, loc.shape)
        # print("loc[0] =", loc[0]) # --第0条gps序列
        # print("loc[0][0] =", loc[0][0]) # 第0条gps序列的第0个点
        # print("loc[1][1] =", loc[1][1])
        # print.show()
        for i in range(0, loc.size()[0]):
            # 建立点和边特征，以及边的标签-->直接用出来的节点特征
            # edge_pred_graph.ndata['feature'] = torch.randn(nodel enght, 4)
            # print("torch.randn(nodelenght, 4) =", torch.randn(nodelenght, 4))
            # 节点特征,第i条gps序列上点的特征
            # print("loc[i] =", loc[i], loc[i].shape)
            edge_pred_graph.ndata['feature'] = loc[i].to(self.device)
            # print("loc[i] =", loc[i])
            # print(show)
            # edge_pred_graph.edata['feature'] = torch.randn(1000, 10)
            # 标签得话，用实际时间
            # edge_pred_graph.edata['label'] = torch.randn((nodelenght-1)*2)
            node_features = edge_pred_graph.ndata['feature'].to(self.device)
            # edge_label = edge_pred_graph.edata['label']
            # train_mask = edge_pred_graph.edata['train_mask']

            # -->得到的是边的特征，其实边也可以看成点
            pred = self.model(edge_pred_graph, node_features)
            # print("pred =", pred, pred.size())
            # print.show()
            if i == 0:
                # 初始值
                pred_new = pred
                # 增加一个维度
                pred_new = pred_new.unsqueeze(0)
                # print("pred_new =", pred_new, pred_new.shape)
            else:
                # 拼接特征
                # 增加一个维度
                pred = pred.unsqueeze(0)
                pred_new = torch.cat((pred_new, pred), 0)

        # dist_gap_mean = self.data_feature.get("dist_gap_mean", 0.274716042312)
        # dist_gap_std = self.data_feature.get("dist_gap_std", 0.127051674693)
        # current_dis = (batch["current_dis"] - dist_gap_mean) / dist_gap_std
        # calculate the dist for local paths
        # local_dist = get_local_seq(current_dis, 3, dist_gap_mean, dist_gap_std, self.device)
        # local_dist = torch.unsqueeze(local_dist, dim=2)

        # pred_new = torch.cat((pred_new, local_dist), dim=2)
        # print("pred_new =", pred_new, pred_new.shape)
        # print.show
        return pred_new


# 去除数组中用于填充补齐的0，并返回处理后的数组
# def filter_zero(lists):
#     new_lists = []
#     for list in lists:
#         for index in range(1, len(list)):
#             if list[index] == 0:
#                 new_lists.append(list[:index])
#                 break
#             elif index == (len(list) - 1):
#                 new_lists.append(list)
#     return new_lists

# 去除数组中用于填充补齐的0，并返回处理后的数组--fixed bug
def filter_zero(lists):
    res = []
    for list in lists:
        new_lists = []
        for index in range(1, len(list)):
            if list[index] != 0:
                new_lists = list[:index+1]
        res.append(new_lists)
    return res


# In[3]:


# 按步长和滑动窗口大小计算一条轨迹的速度
def calculate_speed(dis_list, time_list, step, win_size, speed_mean, speed_std, dist_gap_mean, dist_gap_std,
                    time_gap_mean,
                    time_gap_std):
    speed = []
    dis = []
    time = []
    start = 0
    # print(len(dis_list))
    while start < len(dis_list):
        if start + win_size < len(dis_list):
            # temp_time = 0
            # for index in range(start + 1, start + win_size):
            #     temp_time += time_list[index]
            temp_time = time_list[start + win_size - 1] - time_list[start]
            if temp_time <= 0:
                temp_time = 1
            temp_dis = dis_list[start + win_size - 1] - dis_list[start]
            temp_speed = temp_dis / temp_time * 3600
            time.append(temp_time)
            dis.append(temp_dis)
            speed.append(temp_speed)
        else:
            # temp_time = 0
            # for index in range(start + 1, len(dis_list)):
            #     temp_time += time_list[index]
            # 单独处理最后一个点
            temp_time = time_list[len(dis_list) - 1] - time_list[start]
            temp_dis = dis_list[len(dis_list) - 1] - dis_list[start]
            if start == len(dis_list) - 1:
                temp_time = time_list[start] - time_list[start - 1]
                temp_dis = dis_list[start] - dis_list[start - 1]
            if temp_time <= 0:
                temp_time = 1
            temp_speed = temp_dis / temp_time * 3600
            time.append(temp_time)
            dis.append(temp_dis)
            speed.append(temp_speed)
            break
        start += step
    # print(len(speed))
    speed = (np.array(speed) - speed_mean) / speed_std
    dis = (np.array(dis) - dist_gap_mean) / dist_gap_std
    time = (np.array(time) - time_gap_mean) / time_gap_std
    '''
    print("speed = ", speed, type(speed))
    print("dis = ", dis, type(dis))
    print("time = ", time, type(time))
    '''
    # numpy to list
    speed = speed.tolist()
    dis = dis.tolist()
    time = time.tolist()
    '''
    print("speed = ", speed, type(speed))
    print("dis = ", dis, type(dis))
    print("time = ", time, type(time))
    print(1/0)
    '''
    return speed, dis, time


# In[4]:


# 按最长长度用0补齐数据
def pad_list(list, max_len):
    for item in list:
        while (len(list[item]['speed']) < max_len):
            list[item]['speed'].append(0)
            list[item]['dis'].append(0)
            list[item]['time'].append(0)
    return list


# In[5]:

'''
# 按步长和滑动窗口计算一个batch的速度
def calculate_batch_speed(batch,step,win_size):
    current_dis = filter_zero(batch['current_dis'])
    current_tim = filter_zero(batch['current_tim'])
    uid = batch['uid']
    uid_list = uid.tolist()
    uids = uid.clone()
    uid_list = [i for item in uid_list for i in item]
    set_lst = set(uid_list)
    print("set_lst =", set_lst)
    print("uid_list =", uid_list)
    if len(set_lst) != len(uid_list):
        print("重复")
        print(show)
    # print("uid_list =", uid_list)
    # print(show)
    res = {}
    for uid in uids:
        res[str(uid[0])] = {}
    max_len = 0
    for index in range(0,len(uids)):
        temp_dis = current_dis[index]
        temp_tim = current_tim[index]
        res[str(uids[index][0])]['speed'],res[str(uids[index][0])]['dis'],res[str(uids[index][0])]['time'] = calculate_speed(temp_dis,temp_tim,step,win_size)
        length = len(res[str(uids[index][0])]['speed'])
        max_len = max(length,max_len)
    res = pad_list(res,max_len)
    return res
'''


# 按步长和滑动窗口计算一个batch的速度
def calculate_batch_speed(batch, step, win_size, speed_mean, speed_std, dist_gap_mean, dist_gap_std, time_gap_mean,
                          time_gap_std):
    current_dis = filter_zero(batch['current_dis'])
    current_tim = filter_zero(batch['current_tim'])
    uids = batch['uid']
    res = {}
    max_len = 0
    for index in range(0, len(uids)):
        res[str(index)] = {}
        temp_dis = current_dis[index]
        temp_tim = current_tim[index]
        # print(temp_dis)
        # print(temp_tim)
        # print(uids[index])
        res[str(index)]['speed'], res[str(index)]['dis'], res[str(index)]['time'] = calculate_speed(temp_dis, temp_tim,
                                                                                                    step, win_size,
                                                                                                    speed_mean,
                                                                                                    speed_std,
                                                                                                    dist_gap_mean,
                                                                                                    dist_gap_std,
                                                                                                    time_gap_mean,
                                                                                                    time_gap_std)
        length = len(res[str(index)]['speed'])
        max_len = max(length, max_len)
    res = pad_list(res, max_len)
    return res


class GeoConv(nn.Module):
    def __init__(self, kernel_size, num_filter, data_feature={}, device=torch.device('cpu')):
        super(GeoConv, self).__init__()

        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.data_feature = data_feature
        self.device = device

        self.state_em = nn.Embedding(2, 2)
        # self.process_coords = nn.Linear(4, 16)
        self.process_coords = nn.Linear(10, 16)
        self.conv = nn.Conv1d(16, self.num_filter, self.kernel_size)
        self.gcn = GNN_inGeo(in_features=10,
                             hidden_features=128,
                             # hidden_features=32,
                             out_features=10,
                             data_feature=data_feature,
                             device=device,
                             ).to(device)
        self.gat1 = GAT(in_features=4,
                        data_feature=data_feature,
                        device=device)

    def forward(self, batch):
        '''
        time_gap_mean = self.data_feature.get("time_gap_mean", 43.8756927994)
        time_gap_std = self.data_feature.get("time_gap_std", 51.4811932987)
        # print("timeid =", batch["timeid"], batch["timeid"].size())

        current_tim = ((batch["current_tim"]) - time_gap_mean) / time_gap_std
        print("current_tim =", current_tim)
        dist_gap_mean = self.data_feature.get("dist_gap_mean", 0.274716042312)
        dist_gap_std = self.data_feature.get("dist_gap_std", 0.127051674693)
        # print("batch =", batch["current_dis"])
        current_dis = (batch["current_dis"] - dist_gap_mean) / dist_gap_std
        print("current_dis =", current_dis, current_dis.size())
        print(show)
        '''
        longi_mean = self.data_feature.get("longi_mean", 104.05810954320589)
        longi_std = self.data_feature.get("longi_std", 0.04988770679679998)
        # 经度操作
        current_longi = (batch["current_longi"] - longi_mean) / longi_std
        # print("current_longi =", current_longi, current_longi.shape)
        lngs = torch.unsqueeze(current_longi, dim=2)
        # print("lngs =", lngs, lngs.shape)

        lati_mean = self.data_feature.get("lati_mean", 30.652312982784895)
        lati_std = self.data_feature.get("lati_std", 0.04988770679679998)
        # 纬度操作
        current_lati = (batch['current_lati'] - lati_mean) / lati_std
        # print("current_lati =", current_lati, current_lati.shape)
        lats = torch.unsqueeze(current_lati, dim=2)
        # print("lats =", lats, lats.shape)

        # 可用和不可用
        states = self.state_em(batch['current_state'].long())

        # GPS序列，可对其进行操作-->在途特征可在这进行赋值
        # (起点处几个点的特征3个-速度，距离，时间，最近一次几个点的特征3个-速度，距离，时间)
        locs = torch.cat((lngs, lats, states), dim=2)
        locs_origin = locs.clone()
        # print("locs =", locs, locs.size())
        # print("batch['current_state'] =", batch['current_state'], batch['current_state'].size())
        # print(show)

        speed_mean = self.data_feature.get("speed_mean", 10.975444864116037)
        speed_std = self.data_feature.get("speed_std", 9.675614171662321)
        # print(speed_mean)
        # print(speed_std)
        # print(1/0)
        # dist_gap_mean = self.data_feature.get("dist_gap_mean", 0.26620705556513263)
        # dist_gap_std = self.data_feature.get("dist_gap_std", 0.1317369578211975)
        dist_gap_mean = self.data_feature.get("dist_gap_mean", 0.274716042312)
        dist_gap_std = self.data_feature.get("dist_gap_std", 0.127051674693)
        # time_gap_mean = self.data_feature.get("time_gap_mean", 42.408652928294686)
        # time_gap_std = self.data_feature.get("time_gap_std", 50.802932353194855)
        time_gap_mean = self.data_feature.get("time_gap_mean", 43.8756927994)
        time_gap_std = self.data_feature.get("time_gap_std", 51.4811932987)
        step = 1
        window = 6

        res = calculate_batch_speed(batch, step, window, speed_mean, speed_std, dist_gap_mean, dist_gap_std,
                                    time_gap_mean, time_gap_std)
        # print("res =", res)
        # print(1 / 0)
        # print("type", type(res))
        # print(res.keys())
        # print(show)
        k_flag = True
        for k in res.keys():
            # print("len(res.keys()) =", res.keys(), len(res.keys()))
            # print("len(res.keys()) =", list(res.keys())[0])
            # print("k =", k)
            res_value = res.get(k)
            # print(res_value)
            # print(show)
            flag = True
            # 速度，距离，时间
            for m in res_value.keys():
                # print("m =", m, type(m))
                # if m != type(m)('time'):
                # print("Hello World")

                res_value_value = res_value.get(m)
                # print(res_value_value, '\n', len(res_value_value), type(res_value_value))
                new_value = torch.tensor(res_value_value)
                new_value = torch.unsqueeze(new_value, dim=1)
                # print(new_value, new_value.size())
                if flag == False:
                    x = torch.cat((x, new_value), dim=1)
                else:
                    # print("True")
                    x = new_value
                    flag = False
            # print(show)
            # print("x =", x, x.size())
            # new_x = x.expand(x.size()[0], x.size()[1]+3)
            # expand_attr_t = attr_t.expand(conv_locs.size()[:2] + (attr_t.size()[-1],))
            # print("x =", new_x, new_x.size())
            # print(show)
            x = torch.unsqueeze(x, dim=0)
            # print("x =", x, x.size())
            # print(show)
            if k_flag == False:
                y = torch.cat((y, x), dim=0)
            else:
                # print("True")
                y = x
                k_flag = False
        # print("y before =", y, y.size())
        # y = y.expand(y.size()[0], y.size()[1], y.size()[2])
        # print("y after =", y.size())
        # print.show
        # 每条轨迹
        # feat = torch.empty((locs.size()[0], locs.size()[1], y.size()[2]))
        # print("feat =", feat, feat.size())
        # print.show()
        # y 是在途特征
        # print("locs =", locs.size())
        # print("y =", y.size())
        for i in range(locs.size()[0]):
            # cat_flag = True
            ratio = 0.85
            for j in range(locs.size()[1]):
                # print("locs[i][j] =", locs[i][j])
                if j > window:
                    near_feat = (ratio * y[i][j - window]).to(self.device)
                    fast_feat = ((1 - ratio) * y[i][0]).to(self.device)
                    temp_loc = torch.cat((locs[i][j], near_feat, fast_feat), dim=0)
                    # print("temp_loc =", temp_loc)
                else:
                    near_feat = (ratio * y[i][0]).to(self.device)
                    fast_feat = ((1 - ratio) * y[i][0]).to(self.device)
                    temp_loc = torch.cat((locs[i][j], near_feat, fast_feat), dim=0)
                    # print("temp_loc =", temp_loc)
                temp_loc = temp_loc.unsqueeze(0)
                # print("temp_loc =", temp_loc, temp_loc.size())
                # print.show
                if j > 0:
                    new_loc = torch.cat((new_loc, temp_loc), dim=0)
                elif j == 0:
                    new_loc = temp_loc
            # print("new_loc =", new_loc, new_loc.size())
            new_loc = new_loc.unsqueeze(0)
            # print("new_loc =", new_loc, new_loc.size())
            if i == 0:
                new_loc_batch = new_loc
            elif i > 0:
                new_loc_batch = torch.cat((new_loc_batch, new_loc), dim=0)
        # print("new_loc_batch =", new_loc_batch, new_loc_batch.size())
        # print.show
        # 新处理的轨迹，具有在途的特征
        locs = new_loc_batch
        locs_copy = locs.clone()
        conv_locs = locs_copy

        # GAT
        # gat_locs = self.gat1(batch, locs_origin)
        # locs = gat_locs
        
        # GNN
        # locs_origin is origin, dimension is 4
        # locs is updated, dimension is 10

        # print("locs_origin =", locs_origin, locs_origin.size())
        # print("locs =", locs, locs.size())

        # gcn_locs = self.gcn(batch, locs)
        # print("gcn_locs =", gcn_locs, gcn_locs.size())
        # print.show()
        # GCN
        # locs = gcn_locs

        # print("locs =", locs, locs.shape)
        # print("locs.size() =", locs.size(), locs.size()[0], locs.size()[1], locs.size()[2])

        # print.show()

        # 以下这段为地理卷积最关键的部分，注释掉即表示不使用地理卷积，直接返回GPS点经纬度序列

        # {# geo_Cov

        # map the coords into 16-dim vector
        locs = torch.tanh(self.process_coords(locs))
        # print("locs =", locs, locs.size())
        locs = locs.permute(0, 2, 1)
        # print("locs2 =", locs, locs.size())


        conv_locs = F.elu(self.conv(locs)).permute(0, 2, 1)
        # print("conv_locs =", conv_locs, conv_locs.shape)
        # print.show()

        dist_gap_mean = self.data_feature.get("dist_gap_mean", 0.274716042312)
        dist_gap_std = self.data_feature.get("dist_gap_std", 0.127051674693)
        # print("batch =", batch["current_dis"])
        current_dis = (batch["current_dis"] - dist_gap_mean) / dist_gap_std
        # print("current_dis =", current_dis, current_dis.size())
        # print(show)
        # calculate the dist for local paths
        local_dist = get_local_seq(current_dis, self.kernel_size, dist_gap_mean, dist_gap_std, self.device)
        local_dist = torch.unsqueeze(local_dist, dim=2)
        # print("local_dist =", local_dist, local_dist.size())
        conv_locs = torch.cat((conv_locs, local_dist), dim=2)
        # print("conv_locs =", conv_locs, conv_locs.shape)
        # print,show()
        # return conv_locs
        # }


        # return gnn result
        # conv_locs = gcn_locs
        # conv_locs = torch.cat((gcn_locs, locs_copy), dim=2)

        # conv_locs = locs_origin
        return locs_origin, conv_locs


class MyAttention(nn.Module):
    def __init__(self, in_feature, out_feature, data_feature, device=torch.device('cpu')):
        super(MyAttention, self).__init__()
        # self.gat = GATConv(10, out_feature, num_heads=8)
        self.gat = GATConv((in_feature, out_feature), out_feature, num_heads=8)
        self.device = device
        self.data_feature = data_feature

    def forward(self, batch, gnn_locs, geo_conv_locs):
        # print("myattetion")
        nodelenght = gnn_locs.size()[1]  # 节点个数
        # print("nodelenght =", nodelenght)
        srcnodeid = np.arange(0, nodelenght)  # 节点ID
        dstnodeid = np.arange(0, nodelenght * 2)
        src = np.append(srcnodeid, srcnodeid)  # 源节点
        dst = dstnodeid  # 目标节点
        # print("src =", src)
        # print("dst =", dst)
        # gnn和geo_conv特征合并
        g = dgl.heterograph({('node_type', 'edge_type', 'node_type'): (src, dst)}).to(self.device)
        # 同时建立反向边-->有向图
        # edge_pred_graph = dgl.graph((np.concatenate([src, dst]), np.concatenate([dst, src])))
        # edge_pred_graph = dgl.graph((src, dst)).to(self.device)
        # edge_pred_graph = dgl.add_self_loop(edge_pred_graph)
        dist_gap_mean = self.data_feature.get("dist_gap_mean", 0.274716042312)
        dist_gap_std = self.data_feature.get("dist_gap_std", 0.127051674693)
        current_dis = (batch["current_dis"] - dist_gap_mean) / dist_gap_std
        # calculate the dist for local paths
        local_dist = get_local_seq(current_dis, 3, dist_gap_mean, dist_gap_std, self.device)
        local_dist = torch.unsqueeze(local_dist, dim=2)

        # print("gnn_locs =", gnn_locs, gnn_locs.size())
        # print("geo_conv_locs =", geo_conv_locs, geo_conv_locs.size())
        feat = local_dist.to(self.device)
        feat = torch.cat((feat, feat), dim=1)
        # print("feat =", feat, feat.size())
        # print("feat[0] =", feat[0], feat[0].size())
        # print.show()
        contfeature = torch.cat((gnn_locs, geo_conv_locs), dim=1)

        for i in range(0, gnn_locs.size()[0]):
            # 建立点和边特征，以及边的标签-->直接用出来的节点特征
            # 初始特征直接用距离表示吧，这样更合适，别随意初始化

            # 初始特征直接用距离表示吧，这样更合适，别随意初始化
            srcfeat = feat[i].to(self.device)
            dstfeat = contfeature[i].to(self.device)
            # print("srcfeat =", srcfeat, srcfeat.size())
            # print("contfeature =", contfeature[i], contfeature[i].size())

            res = self.gat(g, (srcfeat, dstfeat))
            # print("res =", res, res.size())
            res = res[0:nodelenght]
            # print("nodelenght =", nodelenght)
            # print("res =", res, res.size())
            res = res.mean(axis=1, keepdim=False)
            # print("res after=", res, res.size())
            # print(show)
            if i == 0:
                # 初始值
                pred_new = res
                # 增加一个维度
                pred_new = pred_new.unsqueeze(0)
                # print("pred_new =", pred_new, pred_new.shape)
            else:
                # 拼接特征
                # 增加一个维度
                pred = res.unsqueeze(0)
                pred_new = torch.cat((pred_new, pred), 0)
                # print("pred_new =", pred_new, pred_new.shape)
                # print.show
        dist_gap_mean = self.data_feature.get("dist_gap_mean", 0.274716042312)
        dist_gap_std = self.data_feature.get("dist_gap_std", 0.127051674693)
        current_dis = (batch["current_dis"] - dist_gap_mean) / dist_gap_std
        # calculate the dist for local paths
        local_dist = get_local_seq(current_dis, 3, dist_gap_mean, dist_gap_std, self.device)
        local_dist = torch.unsqueeze(local_dist, dim=2)
        # print("local_dist =", local_dist, local_dist.size())
        # print("pred_new =", pred_new, pred_new.size())
        # print.show()
        # 路网特征+路段距离
        # pred_new = torch.cat((pred_new, local_dist), dim=2)
        # print("pred_new =", pred_new, pred_new.size())
        # print(show)
        return pred_new


class SpatioTemporal(nn.Module):
    '''
    attr_size: the dimension of attr_net output
    pooling optitions: last, mean, attention
    '''

    def __init__(self, attr_size, kernel_size=3, num_filter=32, pooling_method='attention',
                 rnn_type='LSTM', rnn_num_layers=1, hidden_size=128,
                 data_feature={}, device=torch.device('cpu')):
        super(SpatioTemporal, self).__init__()

        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.pooling_method = pooling_method
        self.hidden_size = hidden_size

        self.data_feature = data_feature
        self.device = device

        self.geo_conv = GeoConv(
            kernel_size=kernel_size,
            num_filter=num_filter,
            data_feature=data_feature,
            device=device,
        )
        self.gnn = GNN(in_features=4,
                       hidden_features=128,
                       out_features=32,
                       data_feature=data_feature,
                       device=device,
                       ).to(device)
        self.gat = GAT(in_features=4,
                       data_feature=data_feature,
                       device=device)
        self.myattention = MyAttention(in_feature=1,
                                       out_feature=33,
                                       data_feature=data_feature,
                                       device=device).to(device)
        # num_filter: output size of each GeoConv + 1:distance of local path + attr_size: output size of attr component
        if rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(
                # 不用地理卷积，直接用经纬度序列-->只用时序特征
                # input_size=4 + attr_size,
                input_size=num_filter + 1 + attr_size,
                hidden_size=hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True,
            )
        elif rnn_type.upper() == 'RNN':
            self.rnn = nn.RNN(
                input_size=num_filter + 1 + attr_size,
                hidden_size=hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True
            )
        else:
            raise ValueError('invalid rnn_type, please select `RNN` or `LSTM`')
        if pooling_method == 'attention':
            self.attr2atten = nn.Linear(attr_size, hidden_size)

    def out_size(self):
        # return the output size of spatio-temporal component
        return self.hidden_size

    def mean_pooling(self, hiddens, lens):
        # note that in pad_packed_sequence, the hidden states are padded with all 0
        hiddens = torch.sum(hiddens, dim=1, keepdim=False)

        lens = torch.FloatTensor(lens).to(self.device)

        lens = Variable(torch.unsqueeze(lens, dim=1), requires_grad=False)

        hiddens = hiddens / lens

        return hiddens

    def atten_pooling(self, hiddens, attr_t):
        atten = torch.tanh(self.attr2atten(attr_t)).permute(0, 2, 1)

        # hidden b*s*f atten b*f*1 alpha b*s*1 (s is length of sequence)
        alpha = torch.bmm(hiddens, atten)
        alpha = torch.exp(-alpha)

        # The padded hidden is 0 (in pytorch), so we do not need to calculate the mask
        alpha = alpha / torch.sum(alpha, dim=1, keepdim=True)

        hiddens = hiddens.permute(0, 2, 1)
        hiddens = torch.bmm(hiddens, alpha)
        hiddens = torch.squeeze(hiddens)

        return hiddens

    def forward(self, batch, attr_t):
        # 地理卷积？咋搞？
        locs_cpoy, conv_locs = self.geo_conv(batch)
        # print("locs_copy =", locs_cpoy, locs_cpoy.size())
        # print(1/0)
        # conv_locs = locs_cpoy

        # GAT
        # gat_locs = self.gat(batch, locs_cpoy)
        # conv_locs = gat_locs
        # print("gat_locs =", gat_locs, gat_locs.size())
        # print.show

        # GNN left
        # gnn_locs = self.gnn(batch, locs_cpoy, attr_t)
        # conv_locs = gnn_locs
        # 操作gnn_locs和conv_locs
        # conv_locs = self.myattention(batch, gnn_locs, conv_locs)

        # print("gnn_locs =", gnn_locs, gnn_locs.size())
        # print("conv_locs =", conv_locs, conv_locs.size())

        # conv_locs = torch.cat((gnn_locs, conv_locs), dim=2)
        # print("conv_locs =", conv_locs, conv_locs.size())
        # print(show)
        # print("conv_locs =", conv_locs, conv_locs.size())
        # print("attr_t =", attr_t, attr_t.size())
        attr_t = torch.unsqueeze(attr_t, dim=1)
        # print("attr_t =", attr_t, attr_t.size())
        # print("attr_t =", attr_t, attr_t.shape)
        expand_attr_t = attr_t.expand(conv_locs.size()[:2] + (attr_t.size()[-1],))
        # print("expand_attr_t =", expand_attr_t, expand_attr_t.shape)
        # print.show()
        # concat the loc_conv and the attributes
        # -->一条GPS序列上的点的特征是一样的？
        conv_locs = torch.cat((conv_locs, expand_attr_t), dim=2)

        lens = [batch["current_dis"].shape[1]] * batch["current_dis"].shape[0]
        lens = list(map(lambda x: x - self.kernel_size + 1, lens))

        packed_inputs = nn.utils.rnn.pack_padded_sequence(conv_locs, lens, batch_first=True)

        packed_hiddens, _ = self.rnn(packed_inputs)
        hiddens, lens = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first=True)

        if self.pooling_method == 'mean':
            return packed_hiddens, lens, self.mean_pooling(hiddens, lens)
        else:
            # self.pooling_method == 'attention'
            # print("packed_hiddens =", packed_hiddens)
            # print("lens =", lens)
            # print("self.atten_pooling(hiddens, attr_t) =", self.atten_pooling(hiddens, attr_t), self.atten_pooling(hiddens, attr_t).size())
            # print(show)
            return packed_hiddens, lens, self.atten_pooling(hiddens, attr_t)


class EntireEstimator(nn.Module):
    def __init__(self, input_size, num_final_fcs, hidden_size=128):
        super(EntireEstimator, self).__init__()

        self.input2hid = nn.Linear(input_size, hidden_size)

        self.residuals = nn.ModuleList()
        for i in range(num_final_fcs):
            self.residuals.append(nn.Linear(hidden_size, hidden_size))

        self.hid2out = nn.Linear(hidden_size, 1)

    def forward(self, attr_t, sptm_t):
        inputs = torch.cat((attr_t, sptm_t), dim=1)

        hidden = F.leaky_relu(self.input2hid(inputs))

        for i in range(len(self.residuals)):
            residual = F.leaky_relu(self.residuals[i](hidden))
            hidden = hidden + residual

        out = self.hid2out(hidden)

        return out

    def eval_on_batch(self, pred, label, mean, std):
        label = label

        label = label * std + mean
        pred = pred * std + mean

        return loss.masked_mape_torch(pred, label)


class LocalEstimator(nn.Module):
    def __init__(self, input_size, eps=10):
        super(LocalEstimator, self).__init__()

        self.input2hid = nn.Linear(input_size, 64)
        self.hid2hid = nn.Linear(64, 32)
        self.hid2out = nn.Linear(32, 1)

        self.eps = eps

    def forward(self, sptm_s):
        hidden = F.leaky_relu(self.input2hid(sptm_s))

        hidden = F.leaky_relu(self.hid2hid(hidden))

        out = self.hid2out(hidden)

        return out

    def eval_on_batch(self, pred, lens, label, mean, std):
        label = nn.utils.rnn.pack_padded_sequence(label, lens, batch_first=True)[0]
        label = label

        label = label * std + mean
        pred = pred * std + mean

        return loss.masked_mape_torch(pred, label, eps=self.eps)


class DeepTTE(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super(DeepTTE, self).__init__(config, data_feature)
        self.config = config
        self.data_feature = data_feature
        self.device = config.get('device', torch.device('cpu'))
        # 尝试把用户ID屏蔽掉，看看结果如何
        uid_emb_size = config.get("uid_emb_size", 16)
        weekid_emb_size = config.get("weekid_emb_size", 3)
        timdid_emb_size = config.get("timdid_emb_size", 8)
        uid_size = data_feature.get("uid_size", 24000)
        # -->输出27维，其中用户id维度为16
        embed_dims = [
            # 如果不想使用user ID可直接将其注释掉，即注释下面一行
            ('uid', uid_size, uid_emb_size),
            ('weekid', 7, weekid_emb_size),
            ('timeid', 1440, timdid_emb_size),
        ]

        # parameter of attribute / spatio-temporal component
        self.kernel_size = config.get('kernel_size', 3)
        num_filter = config.get('num_filter', 32)
        pooling_method = config.get("pooling_method", "attention")

        # parameter of multi-task learning component
        num_final_fcs = config.get('num_final_fcs', 3)
        final_fc_size = config.get('final_fc_size', 128)
        self.alpha = config.get('alpha', 0.3)

        rnn_type = config.get('rnn_type', 'LSTM')
        rnn_num_layers = config.get('rnn_num_layers', 1)
        hidden_size = config.get('hidden_size', 128)

        self.eps = config.get('eps', 10)

        # attribute component-->输出28维，多出的1个维度是距离的，其中用户id的维度为
        self.attr_net = Attr(embed_dims, data_feature)

        # spatio-temporal component
        self.spatio_temporal = SpatioTemporal(
            attr_size=self.attr_net.out_size(),
            kernel_size=self.kernel_size,
            num_filter=num_filter,
            pooling_method=pooling_method,
            rnn_type=rnn_type,
            rnn_num_layers=rnn_num_layers,
            hidden_size=hidden_size,
            data_feature=data_feature,
            device=self.device,
        )

        self.entire_estimate = EntireEstimator(
            input_size=self.spatio_temporal.out_size() + self.attr_net.out_size(),
            num_final_fcs=num_final_fcs,
            hidden_size=final_fc_size,
        )

        self.local_estimate = LocalEstimator(
            input_size=self.spatio_temporal.out_size(),
            eps=self.eps,
        )

        self._init_weight()

    def _init_weight(self):
        for name, param in self.named_parameters():
            if name.find('.bias') != -1:
                param.data.fill_(0)
            elif name.find('.weight') != -1:
                nn.init.xavier_uniform_(param.data)

    def forward(self, batch):
        attr_t = self.attr_net(batch)

        # sptm_s: hidden sequence (B * T * F); sptm_l: lens (list of int);
        # sptm_t: merged tensor after attention/mean pooling
        sptm_s, sptm_l, sptm_t = self.spatio_temporal(batch, attr_t)
        # print("sptm_s =", sptm_s)
        # print("sptm_l =", sptm_l)
        # print("sptm_t =", sptm_t)
        # print,show()
        entire_out = self.entire_estimate(attr_t, sptm_t)

        # sptm_s is a packed sequence (see pytorch doc for details), only used during the training
        if self.training:
            local_out = self.local_estimate(sptm_s[0])
            return entire_out, (local_out, sptm_l)
        else:
            return entire_out

    def calculate_loss(self, batch):
        if self.training:
            entire_out, (local_out, local_length) = self.predict(batch)
        else:
            entire_out = self.predict(batch)

        time_mean = self.data_feature.get("time_mean", 1555.75269436)
        time_std = self.data_feature.get("time_std", 646.373021152)
        entire_out = (entire_out - time_mean) / time_std
        time = (batch["time"] - time_mean) / time_std
        entire_loss = self.entire_estimate.eval_on_batch(entire_out, time, time_mean, time_std)

        if self.training:
            # get the mean/std of each local path
            time_gap_mean = self.data_feature.get("time_gap_mean", 43.8756927994)
            time_gap_std = self.data_feature.get("time_gap_std", 51.4811932987)
            mean, std = (self.kernel_size - 1) * time_gap_mean, (self.kernel_size - 1) * time_gap_std
            current_tim = (batch["current_tim"] - time_gap_mean) / time_gap_std

            # get ground truth of each local path
            local_label = get_local_seq(current_tim, self.kernel_size, mean, std, self.device)
            local_loss = self.local_estimate.eval_on_batch(local_out, local_length, local_label, mean, std)

            return (1 - self.alpha) * entire_loss + self.alpha * local_loss
        else:
            return entire_loss

    def predict(self, batch):
        time_mean = self.data_feature.get("time_mean", 1555.75269436)
        time_std = self.data_feature.get("time_std", 646.373021152)
        if self.training:
            entire_out, (local_out, local_length) = self.forward(batch)
            entire_out = entire_out * time_std + time_mean
            return entire_out, (local_out, local_length)
        else:
            entire_out = self.forward(batch)
            entire_out = entire_out * time_std + time_mean
            return entire_out
