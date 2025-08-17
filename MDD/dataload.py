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
        # Read data, including phenotypic data and labels.
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
        torch.cuda.empty_cache()  # 强制清理显存

        device = embeddings.device
        n = embeddings.shape[0]
        threshold = opt.beta
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        normalized = embeddings / (norms + 1e-8)
        edge_list = []
        chunk_size = 64

        for i in range(0, n, chunk_size):
            chunk = normalized[i:i + chunk_size]
            sim_chunk = torch.mm(chunk, normalized.T)
            for k in range(chunk_size):
                global_i = i + k
                if global_i >= n:
                    break
                mask = (sim_chunk[k] > threshold) & (torch.arange(n, device=device) > global_i)
                cols = torch.where(mask)[0]
                if cols.numel() > 0:
                    rows = torch.full_like(cols, global_i)
                    edge_list.append(torch.stack([rows, cols]))
            del chunk, sim_chunk, mask
            torch.cuda.empty_cache()
        if edge_list:
            edge_index = torch.cat(edge_list, dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
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
        self.log = open(filename, "w", encoding="utf-8")  # 防止编码错误

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass