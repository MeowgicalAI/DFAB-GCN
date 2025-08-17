import numpy as np
from sklearn.model_selection import StratifiedKFold
from graph import get_node_feature
import csv
import pandas as pd
from torch import nn
import sys
from opt import *
import torch.nn.functional as F

opt = OptInit().initialize()


def standardization_intensity_normalization(dataset, dtype):
    mean = dataset.mean()
    std = dataset.std()

    return ((dataset - mean) / std).astype(dtype)


def intensityNormalisationFeatureScaling(dataset, dtype):
    max = dataset.max()
    min = dataset.min()

    return ((dataset - min) / (max - min)).astype(dtype)


class dataloader():
    def __init__(self):
        self.pd_dict = {}
        self.num_classes = opt.num_classes

    def load_data(self):

        subject_IDs = get_ids()
        # Read data and labels.
        # It is recommended to adjust the group in the phe file to 0 and 1.
        labels = get_subject_labels(subject_IDs, score='Group')
        num_nodes = len(subject_IDs)

        y_onehot = np.zeros([num_nodes, self.num_classes])
        y = np.zeros([num_nodes])

        for i in range(num_nodes):
            y_onehot[i, int(labels[subject_IDs[i]]) - 1] = 1
            y[i] = int(labels[subject_IDs[i]])

        # Get the labels and features of the subjects.
        self.y = y
        self.raw_features = get_node_feature()

        return self.raw_features, self.y

    def data_split(self, n_folds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=666)
        cv_splits = list(skf.split(self.raw_features, self.y))
        return cv_splits

    def get_inputs(self, embeddings):
        #Calculate the cosine similarity
        n = embeddings.shape[0]
        device = embeddings.device
        chunk_size = 100
        adj_matrix = torch.zeros(n, n, device=device)
        for i in range(0, n, chunk_size):
            chunk = embeddings[i:i + chunk_size]
            sim_chunk = F.cosine_similarity(
                chunk.unsqueeze(1),
                embeddings.unsqueeze(0),
                dim=2
            )

            # Fill it into the adjacency matrix
            adj_matrix[i:i + chunk_size] = (sim_chunk + 1) / 2
        self.node_ftr = np.array(embeddings.detach().cpu().numpy())
        n = self.node_ftr.shape[0]
        num_edge = n * (1 + n) // 2 - n
        edge_index = np.zeros([2, num_edge], dtype=np.int64)
        aff_score = np.zeros(num_edge, dtype=np.float32)
        aff_adj = adj_matrix
        flatten_ind = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[:, flatten_ind] = [i, j]
                aff_score[flatten_ind] = aff_adj[i][j]
                flatten_ind += 1
        assert flatten_ind == num_edge, "Error in computing edge input"
        # Set the threshold, which is beta in the paper.
        keep_ind = np.where(aff_score > opt.beta)[0]
        edge_index = edge_index[:, keep_ind]
        return edge_index

def get_subject_labels(subject_list, score):
    scores_dict = {}
    labels = opt.labels_path
    with open(labels) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['SUB_ID'] in subject_list:
                scores_dict[row['SUB_ID']] = row[score]
    return scores_dict


def get_ids(num_subjects=None):
    subject_IDs = np.genfromtxt(opt.subject_IDs_path, dtype=str)
    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]
    return subject_IDs


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass