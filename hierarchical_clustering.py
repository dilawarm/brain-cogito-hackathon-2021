from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from matplotlib import pyplot as plt
import pandas as pd

class HierarchicalClustering:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def cluster_data(self):
        if self.df is None: 
            print("No dataset")
            return

        Z = linkage(self.df, 'ward')
        self.Z = Z
        return self

    def extract_cell_name_from_clusters(self, n_clusters=5):
        Z = self.Z

        labels = list(self.df.columns)

        clusters = cut_tree(Z, n_clusters=n_clusters)
        clusters.reshape(clusters.shape[0])

        grouped = {}
        for i, cluster in enumerate(clusters):
            if str(cluster[0]) not in grouped:
                grouped[str(cluster[0])] = []

            grouped[str(cluster[0])].append(labels[i]) 

        grouped_inv = {}
        for (k, v) in grouped.items():
            for cell_name in v:
                grouped_inv[cell_name] = k

        df = pd.DataFrame(data=map(lambda cell_cluster: [cell_cluster[0], cell_cluster[1]], grouped_inv.items()))
        df.columns = ['cell_name', 'cluster']
        return df

    def show_dendrogram(self):
        self.cluster_data()
        Z = self.Z

        if Z is None:
            print("No Z!")
            return

        labels = list(self.df.columns)

        plt.figure()
        dn = dendrogram(Z, orientation='right', labels=labels)
        plt.show()

        return dn


if __name__ == '__main__':
    df = pd.read_csv('./data/relative_distance.csv')
    df.set_index(keys="cell_name", inplace=True)
    HierarchicalClustering(df).show_dendrogram()
    





