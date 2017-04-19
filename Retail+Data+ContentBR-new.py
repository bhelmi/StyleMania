# Content-based Filtering

import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
import scipy

data = pd.read_csv('Online Retail2.csv', header = 0, encoding = "ISO-8859-1", dtype={'StockCode': str})

ds = data.dropna(subset=['Description'])
desc_uniq = ds['Description'].unique()

tf = TfidfVectorizer(analyzer='word',
                         ngram_range=(1, 1),
                         min_df=0,
                         stop_words='english')

tfidf_matrix = tf.fit_transform(desc_uniq)

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

plt.figure(figsize=(12,8))
plt.imshow(cosine_similarities, interpolation='nearest', cmap=plt.cm.hot,
    extent=(0.5,np.shape(cosine_similarities)[0]+0.5,0.5,np.shape(cosine_similarities)[1]+0.5))
plt.colorbar()
plt.show()

similar_indices = cosine_similarities[1].argsort()[:-100:-1]

plt.clf()
D = cosine_similarities
fig = plt.figure(figsize=(15,15))
Y = sch.linkage(D, method='centroid')
Z1 = sch.dendrogram(Y, orientation='right', no_plot=True)
Z2 = sch.dendrogram(Y, no_plot=True)

axmatrix = fig.add_axes([0.01,0.01,0.85,0.85])
idx1 = Z1['leaves']
idx2 = Z2['leaves']
D = D[idx1,:]
D = D[:,idx2]
im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=plt.cm.hot)
axmatrix.set_xticks([])
axmatrix.set_yticks([])
axcolor = fig.add_axes([0.91,0.01,0.02,0.85])
plt.colorbar(im, cax=axcolor)
plt.show()

plt.clf()
axmatrix = fig.add_axes([0.01,0.01,0.85,0.85])
idx1 = Z1['leaves']
idx2 = Z2['leaves']
D = D[idx1,:]
D = D[:,idx2]
im = axmatrix.matshow(D[0:1000,0:1000], aspect='auto', origin='lower', cmap=plt.cm.hot)
axmatrix.set_xticks([])
axmatrix.set_yticks([])
axcolor = fig.add_axes([0.91,0.01,0.02,0.85])
plt.colorbar(im, cax=axcolor)
plt.show()
plt.savefig('similarityMatrix1000')



