import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# constants
pca_components = 8  # I used 8 for a dataset with 14 features


# PREPROCESS YOUR DATA BEFORE RUNNING THE MODEL
# Add your preprocessing
# Remember to set an index (timestep or other indicator)
df = pd.read_csv("hackathon_kpis_anonymised/hackathon_kpis_anonymised.csv")
df = df[df['cell_name'] == '00_11Z']
df = df.fillna(df.median(), axis='index')
df = df.drop(columns='cell_name')
df = df.sort_values('timestamp')
df = df.set_index('timestamp')
print(df.head())
print(df.info())

### End of preprocessing, don't change code below


def add_aggregated_time_information(data, window_size=5):
    """
    data: ndarray, r rows (examples) with c features (columns)
    window_size: int, how many previous examples to aggregate over
    """
    time = np.zeros(shape=data.shape)
    for i in range(window_size, time.shape[0]):
        time[i] = np.mean(data[i-window_size:i], axis=0)

    return np.concatenate((data, time), axis=1)

pca = PCA(n_components=pca_components)
data = pca.fit_transform(df)
data = add_aggregated_time_information(data)
model =  IsolationForest(contamination = 0.1, random_state=42)
model.fit(data)
score = model.score_samples(data)

# Plot
plt.scatter(df.index, score)
plt.show()
