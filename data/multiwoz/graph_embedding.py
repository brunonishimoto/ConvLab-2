from torch import nn, optim
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import json
import copy
import math
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'multiwoz')


class NodeEmbeddings(nn.Module):

    def __init__(self, num_embedding, embedding_dim):
        super(NodeEmbeddings, self).__init__()

        self.embeddings = nn.Embedding(num_embedding, embedding_dim)

    def forward(self, idxs):
        return self.embeddings(idxs)

    def get_embeddings(self):
        return self.embeddings.weight.detach().numpy()

class GraphEmbedding():

    def __init__(self, embedding_dim):
        self.adj_matrix = pd.read_csv(os.path.join(DATA_DIR, 'adj_matrix.csv'), index_col=0)
        self.idx2node = json.load(open(os.path.join(DATA_DIR, 'idx2node.json'), 'r'))

        self.num_embedding = self.adj_matrix.shape[0]
        self.node_embeddings = NodeEmbeddings(self.num_embedding, embedding_dim)

        self.optim = optim.SGD(self.node_embeddings.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss(reduction='sum')

    def load_embedding(self, filepath):
        if os.path.exists(filepath):
            self.node_embeddings.load_state_dict(torch.load(filepath))

    def save_embedding(self, filepath):
        torch.save(self.node_embeddings.state_dict(), filepath)

    def update(self, epoch):
        inputs = torch.Tensor([])
        targets = torch.Tensor([])
        for i in range(self.num_embedding):
            for j in range(self.num_embedding):
                targets = torch.cat([targets, torch.Tensor([self.adj_matrix[self.idx2node[str(i)]][self.idx2node[str(j)]]])])

                emb1 = self.node_embeddings(torch.LongTensor([i])).view(-1)
                emb2 = self.node_embeddings(torch.LongTensor([j])).view(-1)

                dot_product = torch.dot(emb1, emb2).view(1)
                inputs = torch.cat([inputs, dot_product])

        loss = self.loss_fn(inputs, targets)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        print(f"Epoch {epoch}: Loss - {loss.item()}")
        return loss

    def _distance_matrix(self):
        distance_matrix = np.zeros((self.num_embedding, self.num_embedding))
        embeddings = self.node_embeddings.get_embeddings()

        for i in range(self.num_embedding):
            for j in range(self.num_embedding):
                if i == j:
                    distance_matrix[i][j] = 0.
                else:
                    distance_matrix[i][j] = 1 - embeddings[i].dot(embeddings[j])

        df = pd.DataFrame(data=distance_matrix, index=list(self.idx2node.values()), columns=list(self.idx2node.values()))
        return distance_matrix


    def plot_embeddings(self):
        "Creates and TSNE model and plots it"
        distance_matrix = self._distance_matrix()
        labels = list(self.idx2node.values())

        tsne_model = TSNE(perplexity=5, n_components=2, n_iter=10000, random_state=23, metric='precomputed')
        new_values = tsne_model.fit_transform(distance_matrix)

        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])

        plt.figure(figsize=(16, 16))
        for i in range(len(x)):
            plt.scatter(x[i],y[i])
            plt.annotate(labels[i],
                        xy=(x[i], y[i]),
                        xytext=(5, 2),
                        textcoords='offset points',
                        ha='right',
                        va='bottom')
        plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--load_file", type=str, default="embeddings.mdl", help="file of model to load")
    parser.add_argument("--save_file", type=str, default="", help="file of model to save")
    parser.add_argument("--epoch", type=int, default=0, help="number of epochs to train")
    parser.add_argument("--emb_dim", type=int, default=50, help="dimension of embedding")
    parser.add_argument("--plot", type=bool, default=True, help="flag to plot or not the embedding")
    args = parser.parse_args()

    graph_embedding = GraphEmbedding(args.emb_dim)

    if args.load_file:
        graph_embedding.load_embedding(os.path.join(DATA_DIR, args.load_file))

    for i in range(args.epoch):
       graph_embedding.update(i)

    if args.save_file:
        graph_embedding.save_embedding(os.path.join(DATA_DIR, args.save_file))

    if args.plot:
        graph_embedding.plot_embeddings()
