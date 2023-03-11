import pandas as pd

from utils import *

import os
import sys
from knowledge_graph import KnowledgeGraph
from tqdm import tqdm
import pickle
import random
import torch
from datetime import datetime
from sklearn.cluster import MeanShift
import numpy as np
# X=np.array([[1, 1, 4], [2, 1, 7], [1, 0, 6],[4, 7, 8], [3, 5,9], [3, 6,5]])
embeds=load_embed(BEAUTY)
X=embeds[PRODUCT]
print('ok')
clustering = MeanShift(bandwidth=5.4).fit(X)
print(clustering)
print(clustering.labels_)
print(clustering.cluster_centers_)
cluster=clustering.labels_
cluster=pd.DataFrame(cluster)
cluster.to_csv('./cluster.csv')
cluster_centers=clustering.cluster_centers_
cluster_centers=pd.DataFrame(cluster_centers)
cluster_centers.to_csv('./cluster_center.csv')


# print(embeds[PRODUCT])
