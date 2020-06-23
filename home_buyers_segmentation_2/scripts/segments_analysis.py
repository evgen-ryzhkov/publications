import matplotlib.pyplot as plt
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
    # print(df_segmented.groupby('Cluster')['prop_type'].describe())

    # create profiling table
    print('Cluster 4 ----------------')

    print(df_segmented.groupby('Cluster'))
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