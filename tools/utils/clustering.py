from sklearn import cluster
from scipy.spatial import distance
import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np


def compute_bic(kmeans,X):
    """
    
    FROM: https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)

def silhouette_score(X, rng = (2,30)):
    '''https://datascience.stackexchange.com/questions/6508/k-means-incoherent-behaviour-choosing-k-with-elbow-method-bic-variance-explain

    '''
    from sklearn.metrics import silhouette_score

    s = []
    for n_clusters in range(rng[0],rng[1]):
        kmeans = cluster.KMeans(n_clusters=n_clusters)
        kmeans.fit(X)
    
        labels = kmeans.labels_
        #centroids = kmeans.cluster_centers_
    
        s.append(silhouette_score(X, labels, metric='euclidean'))
    return s


if __name__ == '__main__':
    
    # IRIS DATA
    iris = sklearn.datasets.load_iris()
    X = iris.data[:, :4]  # extract only the features
    #Xs = StandardScaler().fit_transform(X)
    Y = iris.target
    
    ks = range(1,25)
    
    # run 9 times kmeans and save each result in the KMeans object
    KMeans = [cluster.KMeans(n_clusters = i, init="k-means++").fit(X) for i in ks]
    
    # now run for each cluster the BIC computation
    BIC = [compute_bic(kmeansi,X) for kmeansi in KMeans]
    
    print(BIC)
    
    plt.plot(BIC)


