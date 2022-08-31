import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import k_means as km
from matplotlib.animation import FuncAnimation

sns.set_style('darkgrid')

np.random.seed(6)
# Fit Model 

data_2 = pd.read_csv('data_2.csv')

X = data_2[['x0', 'x1']]
X['x0'] = X['x0']/np.max(X['x0'])
X['x1'] = X["x1"]/np.max(X['x1'])

model_2 = km.KMeans(k=10, init="kmeans++", max_iter=10, alpha=0.1)
model_2.fit(X)

# Compute Silhouette Score 
z = model_2.predict(X)
print(f'Distortion: {km.euclidean_distortion(X, z) :.3f}')
print(f'Silhouette Score: {km.euclidean_silhouette(X, z) :.3f}')

# Plot cluster assignments
_, ax = plt.subplots(figsize=(5, 5), dpi=100)

def predict_history(X, centroids):
    X = X.to_numpy()
    distances = km.cross_euclidean_distance(centroids, X)
    prediction = np.argmin(distances, axis=0)  
    return prediction  

def animate(i):
    ax.clear()

    C = model_2.get_centroid_history()
    z = predict_history(X, C[i])
    K = len(C[i])
    sns.scatterplot(x='x0', y='x1', hue=z, hue_order=range(K), palette='tab10', data=X, ax=ax);
    sns.scatterplot(x=C[i,:,0], y=C[i,:,1], hue=range(K), palette='tab10', marker='*', s=250, edgecolor='black', ax=ax)
    ax.legend().remove();

anim = FuncAnimation(_, animate, frames=model_2.max_iter + 1, interval=1000, repeat=True)
anim.save('animation.gif', writer='pillow', fps=1)
plt.show()