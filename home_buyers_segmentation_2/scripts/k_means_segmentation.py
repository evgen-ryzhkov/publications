from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def get_segments(df_original, df_normalized, cluster_column_name, num_clusters):
    # run model - get clusters
    cluster_labels = _get_clusters(df_normalized, num_clusters)

    # Create a cluster label column in original dataset
    df_original[cluster_column_name] = cluster_labels

    return df_original


def get_number_of_segments(df_normalized):
    # get number of segments - elbow method
    n_clusters = range(1, 10)
    inertia = {}
    inertia_values = []

    for n in n_clusters:
        model = KMeans(
            n_clusters=n,
            init='k-means++',
            max_iter=500,
            random_state=42)
        model.fit(df_normalized)
        inertia[n]=model.inertia_
        inertia_values.append(model.inertia_)

    # for key, val in inertia.items():
    #     print(str(key) + ' : ' + str(val))

    plt.plot(n_clusters, inertia_values, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method using Inertia')
    plt.show()

    print('[INFO] Set number of cluster variable according to the elbow plot')
    exit()


def _get_clusters(df_normalized, num_clusters):
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=1)
    kmeans_model.fit(df_normalized)

    # Extract cluster labels
    cluster_labels = kmeans_model.labels_

    return cluster_labels


def validate_cluster_sizes(df_segmented, cluster_col_name):
    print('[INFO] Segment sizes validation...')
    df_stat = round(((df_segmented.groupby(cluster_col_name).size())/len(df_segmented))*100).astype(int)
    # print('Cluster distribution, %\n', df_stat)

    f_validation = True
    for i in range(len(df_stat)):
        # each segment has to be in 5-30%
        if (df_stat[i] < 4) | (df_stat[i] > 35):
            print('[Validation Error: There are segments with bad distribution.]')
            f_validation = False
            exit()

    # visualisation of segments distribution
    if f_validation:
        # df_stat.plot.bar()
        # plt.show()

        print('[OK] Segment sizes are OK!')

    return f_validation, df_stat
