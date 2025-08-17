from torch_geometric.io.planetoid import edge_index_from_dict
from torch_geometric.nn import ChebConv,TransformerConv
from dataload import dataloader
from opt import *
import torch.nn.functional as F
from torch import nn
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import subgraph
from torch_geometric.nn.dense.diff_pool import dense_diff_pool
from torch_geometric.nn import SAGPooling
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime



opt = OptInit().initialize()


class Transhemispheric_Fusion_Graph(torch.nn.Module):

    def __init__(self):
        super(Transhemispheric_Fusion_Graph, self).__init__()
        self._setup()
        self.score_saver = ScoreSaver()
    def _setup(self):
        self.graph_convolution_l_1 = GCNConv(111,64)
        self.graph_convolution_r_1 = GCNConv(111,64)

        self.graph_convolution_l_2 = GCNConv(64,20)
        self.graph_convolution_r_2 = GCNConv(64,20)

        self.graph_convolution_g_1 = GCNConv(20,20)

        self.pooling_1 = SAGPooling(20, opt.k1)
        self.socre_gcn = ChebConv(20, int(opt.k2*112), K=3, normalization='sym')
        self.pooling_2= dense_diff_pool

        self.weight = nn.Parameter(torch.FloatTensor(64, 20)).to(opt.device)
        self.bns=nn.BatchNorm1d(20).to(opt.device)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(20)
        nn.init.xavier_normal_(self.weight)

    def forward(self, data,sample_id=None):
        edges, features = data.edge_index, data.x
        edges, features = edges.to(opt.device), features.to(opt.device)

        edge_attr = data.edge_attr
        edge_attr = edge_attr.to(opt.device).to(torch.float32)

        adj=data.adj
        adj=torch.tensor(adj)
        adj=adj.float()
        adj = adj.to(opt.device)

        # Left and right hemisphere index of fmri data in the ABIDE dataset based on the HO Brain Atlas.
        leftBrain = torch.tensor([  6.,   5.,  55.,   1.,  98.,  71.,  73.,  77.,  63.,  96.,  79.,  15.,
        104.,   4.,  25.,  23.,  41.,  43.,  45.,  17.,  61.,  65.,  59.,  57.,
         86.,  21.,  35.,  37.,  39.,  94., 110.,   3.,  69.,  81.,  84., 100.,
        102., 106.,  47.,  27.,  75.,   2.,  67.,  19.,  49.,  31.,  33., 108.,
         51.,  53.,  88.,  90.,  92.,  29.,   0.],device=opt.device).long()
        rightBrain = torch.tensor([ 13.,  12.,  54.,   8.,  97.,  70.,  72.,  76.,  62.,  95.,  78.,  14.,
        103.,  11.,  24.,  22.,  40.,  42.,  44.,  16.,  60.,  64.,  58.,  56.,
         85.,  20.,  34.,  36.,  38.,  93., 109.,  10.,  68.,  80.,  83.,  99.,
        101., 105.,  46.,  26.,  74.,   9.,  66.,  18.,  48.,  30.,  32., 107.,
         50.,  52.,  87.,  89.,  91.,  28.,   7.],device=opt.device).long()

        # Get a subgraph of the left and right hemispheres of the brain.
        new_left_edges, new_left_edge_attr = subgraph(subset=leftBrain.type(torch.long), edge_index=edges,
                                                      edge_attr=edge_attr, num_nodes=111)
        new_right_dges, new_right_edge_attr = subgraph(subset=rightBrain.type(torch.long), edge_index=edges,
                                                       edge_attr=edge_attr, num_nodes=111)

        # The intrahemispheric convolution.
        features = F.dropout(features, p=opt.dropout, training=self.training)
        node_features_left = torch.nn.functional.leaky_relu(self.graph_convolution_l_1(features, new_left_edges, new_left_edge_attr))


        node_features_right = torch.nn.functional.leaky_relu(self.graph_convolution_r_1(features, new_right_dges, new_right_edge_attr))


        node_features_1 = torch.zeros(111,64).to(opt.device)
        node_features_1[leftBrain.long(),:] = node_features_left[leftBrain.long(),:]
        node_features_1[rightBrain.long(), :] = node_features_right[rightBrain.long(), :]

        node_features_1 = F.dropout(node_features_1, p=opt.dropout, training=self.training)
        node_features_left = torch.nn.functional.leaky_relu(self.graph_convolution_l_2(node_features_1, new_left_edges, new_left_edge_attr))


        node_features_right = torch.nn.functional.leaky_relu(self.graph_convolution_r_2(node_features_1, new_right_dges, new_right_edge_attr))


        node_features_2 = torch.zeros(111,20).to(opt.device)
        node_features_2[leftBrain.long(),:] = node_features_left[leftBrain.long(),:]
        node_features_2[rightBrain.long(), :] = node_features_right[rightBrain.long(), :]

        # The interhemispheric convolution.
        node_features_2 = torch.nn.functional.leaky_relu(self.graph_convolution_g_1(node_features_2, edges, edge_attr))


        # --The  pooling  of the THSP-GCN.
        # The channel 1,Self-Attention Graph Pooling
        pooling_features, edges, edge_attr,batch, perm, score = self.pooling_1(node_features_2, edges,edge_attr)
        if opt.train==0:
            if sample_id is None:
               self.score_saver.save_single_sample(perm, score, sample_id)
        # The channel 2,Cluster compensation
        ass_matrix=F.softmax(self.socre_gcn(node_features_2,edges),dim=-1)
        H_coarse,assign_matrix, link_loss, ent_loss = self.pooling_2(node_features_2, adj,ass_matrix)

        # The cross-channel convolution.
        inter_channel_adj = features.new_zeros(100,56)
        assign_matrix = torch.squeeze(ass_matrix)
        j = 0
        for i in range(0,110):
            if i in perm:
                inter_channel_adj[j, :]=assign_matrix[i, :]
                j = j+1
        H_coarse = torch.squeeze(H_coarse)
        H1=torch.matmul(inter_channel_adj, H_coarse)
        H2=pooling_features + H1
        graph_embedding = H2.view(1, -1)

        return graph_embedding

class Graph_Embedding_Graph(nn.Module):
    def __init__(self):
        super(Graph_Embedding_Graph, self).__init__()
        self.num_layers = 4

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()


        self.convs.append(TransformerConv(in_channels=2000, out_channels=20, heads=1))
        self.bns.append(nn.BatchNorm1d(20))

        self.convs.append(TransformerConv(in_channels=20, out_channels=20, heads=1))
        self.bns.append(nn.BatchNorm1d(20))

        self.convs.append(TransformerConv(in_channels=20, out_channels=20, heads=1))
        self.bns.append(nn.BatchNorm1d(20))


        self.convs.append(TransformerConv(in_channels=20, out_channels=20, heads=1))
        self.bns.append(nn.BatchNorm1d(20))

        self.out_fc = nn.Linear(80, 2)


        self.a = torch.nn.Parameter(torch.Tensor(20, 1))

    def reset_parameters(self):

        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

        self.out_fc.reset_parameters()
        self.a.reset_parameters()
        torch.nn.init.normal_(self.weights)

    def forward(self, features, edge_index):
        x = features
        # Graph transformer and information aggregation layers.
        x = F.dropout(x, p=opt.dropout, training=self.training)
        x3 = self.convs[0](x,edge_index)

        x = self.bns[0](x3)
        x = F.leaky_relu(x, inplace=True)
        fc = x

        x = F.dropout(x, p=opt.dropout, training=self.training)
        x3 = self.convs[1](x, edge_index)
        x = self.bns[1](x3)
        x = F.leaky_relu(x, inplace=True)
        fc = torch.cat((fc, x), dim=-1)

        x = F.dropout(x, p=opt.dropout, training=self.training)
        x3 = self.convs[2](x, edge_index)

        x = self.bns[2](x3)
        x = F.leaky_relu(x, inplace=True)
        fc = torch.cat((fc, x), dim=-1)

        x = F.dropout(x, p=opt.dropout, training=self.training)

        x3 = self.convs[3](x, edge_index)

        x = self.bns[3](x3)
        x = F.leaky_relu(x, inplace=True)
        fc = torch.cat((fc, x), dim=-1)
        x = self.out_fc(fc)

        return x

class dfab_gcn(torch.nn.Module):

    def __init__(self):
        super(dfab_gcn, self).__init__()
        self._setup()

    def _setup(self):
        self.individual_graph_model = Transhemispheric_Fusion_Graph()
        self.population_graph_model = Graph_Embedding_Graph()

    def forward(self, graphs):
        dl = dataloader()
        embeddings = []

        # Brain connectomic graph
        for graph in graphs:
            embedding= self.individual_graph_model(graph)
            embeddings.append(embedding)
        embeddings = torch.cat(tuple(embeddings))
        # Similarity population graph
        edge_index= dl.get_inputs(embeddings)
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)
        #edge_index = edge_index.detach().to(opt.device)
        predictions = self.population_graph_model(embeddings,edge_index)

        return predictions

class ScoreSaver:
    def __init__(self, output_dir="./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def save_single_sample(self, perm, score, sample_id):
        df = pd.DataFrame({
            "Node": perm.detach().cpu().numpy(),
            "Score": score.detach().cpu().numpy()
        })
        df.to_excel(self.output_dir / f"{sample_id}_scores.xlsx", index=False)
