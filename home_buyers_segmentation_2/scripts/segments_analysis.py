import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def analyse_segments(df_original, df_normalized, df_segmented):

    f_validation, df_stat = _validate_cluster_sizes(df_segmented)

    _profile_clusters(df_segmented, df_stat)

    # _show_snake_plot(df_original, df_normalized, df_segmented)


def _validate_cluster_sizes(df_segmented):
    print('[INFO] Segment sizes validation...')
    df_stat = round(((df_segmented.groupby('Cluster').size())/len(df_segmented))*100).astype(int)
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


def _profile_clusters(df_segmented, df_stat):

    # explore median values
    # median_columns = ['prop_size', 'prop_complectation']

    # df_profile = round(df_segmented.groupby('Cluster')[median_columns].median()).astype(int)
    # df_merged = pd.concat([df_stat, df_profile], axis=1)

    # print(df_merged.sort_values(by=[0], ascending=False))

    # detail explore features
    # pd.set_option('display.float_format', lambda x: '%.0f' % x)

    # fast / validation profiling
    print(df_segmented.groupby('Cluster').describe())

    # detail profiling

    # create profiling table
    n_clusters = 5
    df_profiling_cols = [
        'Apartment', 'Townhouse', 'Semi_detached house', 'Detached_House',
        'S', 'M', 'L',
        'Poor', 'Normal', 'Good', 'Excellent'
        ]

    apartment_percents_arr = []
    townhouse_percents_arr = []
    semi_detached_percents_arr = []
    detached_percents_arr = []
    size_s_percents_arr = []
    size_m_percents_arr = []
    size_l_percents_arr = []
    compl_poor_percents_arr = []
    compl_norm_percents_arr = []
    compl_good_percents_arr = []
    compl_exc_percents_arr = []
    for cluster in range(n_clusters):

        df_cluster = df_segmented.loc[df_segmented['Cluster'] == cluster]
        df_cluster_len = len(df_cluster)

        percent_val = round((len(df_cluster.loc[df_cluster['prop_type'] == 'Apartment']) / df_cluster_len) * 100)
        apartment_percents_arr.append(percent_val)

        percent_val = round((len(df_cluster.loc[df_cluster['prop_type'] == 'Townhouse']) / df_cluster_len) * 100)
        townhouse_percents_arr.append(percent_val)

        percent_val = round((len(df_cluster.loc[df_cluster['prop_type'] == 'Semi_detached house']) / df_cluster_len) * 100)
        semi_detached_percents_arr.append(percent_val)

        percent_val = round((len(df_cluster.loc[df_cluster['prop_type'] == 'Detached_House']) / df_cluster_len) * 100)
        detached_percents_arr.append(percent_val)

        percent_val = round((len(df_cluster.loc[df_cluster['prop_size'] == 'S']) / df_cluster_len) * 100)
        size_s_percents_arr.append(percent_val)

        percent_val = round((len(df_cluster.loc[df_cluster['prop_size'] == 'M']) / df_cluster_len) * 100)
        size_m_percents_arr.append(percent_val)

        percent_val = round((len(df_cluster.loc[df_cluster['prop_size'] == 'L']) / df_cluster_len) * 100)
        size_l_percents_arr.append(percent_val)

        percent_val = round((len(df_cluster.loc[df_cluster['prop_complectation'] == 'Poor']) / df_cluster_len) * 100)
        compl_poor_percents_arr.append(percent_val)

        percent_val = round((len(df_cluster.loc[df_cluster['prop_complectation'] == 'Normal']) / df_cluster_len) * 100)
        compl_norm_percents_arr.append(percent_val)

        percent_val = round((len(df_cluster.loc[df_cluster['prop_complectation'] == 'Good']) / df_cluster_len) * 100)
        compl_good_percents_arr.append(percent_val)

        percent_val = round((len(df_cluster.loc[df_cluster['prop_complectation'] == 'Excellent']) / df_cluster_len) * 100)
        compl_exc_percents_arr.append(percent_val)

    # for col in df_profiling_cols:
    #
    #     for cluster in range(n_clusters):
    #
    #
    #         df_cluster_len = len(df_cluster)
    #
    #         df_cluster_col_len = df_cluster.loc[df_cluster[]]
    #         col_percent =
    #
    #         df_profiling[cluster][col] = 1
    profile_dic = {
        'Flat': apartment_percents_arr,
        'Townhouse': townhouse_percents_arr,
        'S detached house': semi_detached_percents_arr,
        'Detached_House': detached_percents_arr,
        'S': size_s_percents_arr,
        'M': size_m_percents_arr,
        'L': size_l_percents_arr,
        'Poor': compl_poor_percents_arr,
        'Normal': compl_norm_percents_arr,
        'Good': compl_good_percents_arr,
        'Excellent': compl_exc_percents_arr

    }

    df_profiling = pd.DataFrame(data=profile_dic)
    df_profiling = pd.concat([df_stat, df_profiling], axis=1, sort=False)
    print(df_profiling)

    heat_map = sns.heatmap(df_profiling, annot=True)
    plt.show()

    # print(df_segmented.groupby('Cluster'))
    # print(df_cluster_profile.astype('object').describe().transpose())





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