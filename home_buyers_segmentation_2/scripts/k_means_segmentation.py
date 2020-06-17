from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def get_segments(df_original, df_normalized):
    # hyperparameter tuning
    num_clusters = _define_number_of_segments(df_normalized)

    # run model - get clusters
    cluster_labels = _get_clusters(df_normalized, num_clusters)

    # Create a cluster label column in original dataset
    df_new = df_original.assign(Cluster=cluster_labels)

    return df_new


def _define_number_of_segments(df_normalized):
    # get number of segments - elbow method

    # n_clusters = range(1, 10)
    # inertia = {}
    # inertia_values = []
    #
    # for n in n_clusters:
    #     model = KMeans(
    #         n_clusters=n,
    #         init='k-means++',
    #         max_iter=500,
    #         random_state=42)
    #     model.fit(df_normalized)
    #     inertia[n]=model.inertia_
    #     inertia_values.append(model.inertia_)
    #
    # for key, val in inertia.items():
    #     print(str(key) + ' : ' + str(val))
    #
    # plt.plot(n_clusters, inertia_values, 'bx-')
    # plt.xlabel('Values of K')
    # plt.ylabel('Inertia')
    # plt.title('The Elbow Method using Inertia')
    # plt.show()

    optimal_segments_number = 6
    return optimal_segments_number


def _get_clusters(df_normalized, num_clusters):
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=1)
    kmeans_model.fit(df_normalized)

    # Extract cluster labels
    cluster_labels = kmeans_model.labels_

    return cluster_labels
