import sys
import numpy as np
import pandas as pd
# from sklearn import preprocessing # Though standard, not necessary here.

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot') 

# List colors for clustering
c = ['red', 'blue', 'green', 'orange', 'yellow', 'brown']
  
def do_kmeans(data, clusters=0):
    """ Passes # of clusters & returns a tuple of cluster centers & labels."""
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters= clusters)
    kmeans.fit(data)
    KMeans(copy_x=True, 
           init='k-means++', 
           max_iter=300, 
           n_clusters=clusters, 
           n_init=10, 
           n_jobs=1, 
           precompute_distances='auto', 
           random_state=None, 
           tol=0.0001,
           verbose=0)
    model = kmeans
    return model.cluster_centers_, model.labels_


def main():
    """Constructs KMeans sample groups & centroids."""
    fig = plt.figure()
    ax = fig.add_subplot(111)

    sample_colors = [ c[labels[i]] for i in range(len(T)) ]
    ax.scatter(T[:, 0], T[:, 1], c=sample_colors, marker='o', alpha=0.2)

    # Plots centroids as X's with labels
    ax.scatter(CC[:, 0], CC[:, 1], marker='x', s=169, linewidths=3, zorder=1000, c=c)
    for i in range(len(centroids)): ax.text(CC[i, 0], 
                                            CC[i, 1], str(i), 
                                            zorder=500010, 
                                            fontsize=18, 
                                            color=c[i])

    # Add the cluster label back into the dataframe and display it:
    df['label'] = pd.Series(labels, index=df.index)

    plt.show()
    
if __name__ == "__main__":
    """Performs a KMeans Clustering for key funnel stages for UCPMS."""
    # Loads Data
    filename = sys.argv[1]
    ucpms_df = pd.read_csv(filename)
    # Re-index and moves confidential data
    del ucpms_df['Last Login']
    ucpms_df = ucpms_df.reset_index()
    del ucpms_df['ID']
    # Adds key performance indicators to identify stages of deposit process
    ucpms_df['pending+claimed+declined'] = ucpms_df['Pending Publications'] + \
                            ucpms_df['Claimed Publications'] + ucpms_df['Declined Publications']
    ucpms_df['claim %'] = ucpms_df['Claimed Publications'] / \
                        ucpms_df['pending+claimed+declined']
    ucpms_df['deposit %'] = ucpms_df['Completed Deposits'] / ucpms_df['Claimed Publications']
    ucpms_filtered = ucpms_df.dropna(axis=0)
    ucpms_filtered = ucpms_filtered[['claim %','deposit %']]
    df = ucpms_filtered

    # Performs KMeans
    n_clusters = 2
    centroids, labels = do_kmeans(ucpms_filtered, n_clusters)
    print "The centroids are:\n", centroids

    T = ucpms_filtered.as_matrix()
    CC = centroids
    
    main()
