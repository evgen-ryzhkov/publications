import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def analyse_segments(df_original, df_normalized, df_segmented, cluster_col_name):

    f_validation, df_stat = _validate_cluster_sizes(df_segmented, cluster_col_name)

    _profile_clusters(df_segmented, df_stat, cluster_col_name)

    # _show_snake_plot(df_original, df_normalized, df_segmented)


def _validate_cluster_sizes(df_segmented, cluster_col_name):
    print('[INFO] Segment sizes validation...')
    df_stat = round(((df_segmented.groupby(cluster_col_name).size())/len(df_segmented))*100).astype(int)
    print('Cluster distribution, %\n', df_stat)

    f_validation = True
    for i in range(len(df_stat)):
        if (df_stat[i] < 5) | (df_stat[i] > 40):
            print('[Validation Error: There are segments with bad distribution.]')
            f_validation = False

    # visualisation of segments distribution
    if f_validation:
        # df_stat.plot.bar()
        # plt.show()

        print('[OK] Segment sizes are OK!')

    return f_validation, df_stat


def _profile_clusters(df_segmented, df_stat, cluster_col_name):

    # explore median values
    # median_columns = ['prop_size', 'prop_complectation']

    # df_profile = round(df_segmented.groupby('Cluster')[median_columns].median()).astype(int)
    # df_merged = pd.concat([df_stat, df_profile], axis=1)

    # print(df_merged.sort_values(by=[0], ascending=False))

    # detail explore features
    # pd.set_option('display.float_format', lambda x: '%.0f' % x)

    # fast / validation profiling
    print(df_segmented.groupby(cluster_col_name).describe())

    # detail profiling

    # create profiling table
    property_types = ['Apartment', 'Townhouse', 'Semi_detached house', 'Detached_House']
    size_types = ['S', 'M', 'L']
    complectation_types = ['Poor', 'Normal', 'Good', 'Excellent']

    df_profiling_cols = property_types + size_types + complectation_types

    # initiation of df_profiling
    n_clusters = 5
    n_columns = len(df_profiling_cols)
    init_array = np.zeros((n_clusters, n_columns))
    df_profiling = pd.DataFrame(data=init_array, columns=df_profiling_cols)

    # fill df_profiling with real values
    # value - percents each type of features in cluster
    for cluster in range(n_clusters):
        df_cluster = df_segmented.loc[df_segmented[cluster_col_name] == cluster]
        df_cluster_len = len(df_cluster)

        for col in df_profiling_cols:

            # there are different columns in original df
            # for different profiling columns
            # for example: columnn prop_size in df_original
            # transforms into three columns S, M, L in df_profiling
            df_col_name = ''
            if col in property_types:
                df_col_name = 'prop_type'
            elif col in size_types:
                df_col_name = 'prop_size'
            elif col in complectation_types:
                df_col_name = 'prop_complectation'

            percent_val = round((len(df_cluster.loc[df_cluster[df_col_name] == col]) / df_cluster_len) * 100)
            df_profiling.loc[cluster, col] = percent_val

    # add distribution values for each cluster
    df_profiling = pd.concat([df_stat, df_profiling], axis=1, sort=False)
    print(df_profiling)

    sns.heatmap(df_profiling, annot=True)
    plt.show()


def _show_snake_plot(df_original, df_normalized, df_segmented):

    # Transform df_normal as df and add cluster column
    df_normalized = pd.DataFrame(df_normalized,
                                 index=df_original.index,
                                 columns=df_original.columns)
    df_normalized['Cluster'] = df_segmented['Cluster']

    # Melt data into long format
    df_melt = pd.melt(df_normalized.reset_index(),
                      id_vars=['Cluster'],
                      value_vars=['Apartment', 'Detached_House', 'Semi_detached house', 'Townhouse', 'prop_size', 'prop_complectation'],
                      var_name='Metric',
                      value_name='Value')

    # fig, ax = plt.subplots()

    plt.figure(figsize=(20, 6))
    # fig.set_size_inches(30, 6)
    plt.xlabel('Metric')
    plt.ylabel('Value')
    sns.pointplot(data=df_melt, x='Metric', y='Value', hue='Cluster')
    plt.show()