import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def analyse_segments(df_original, df_normalized, df_segmented):
    # show clasters stats
    # print('[INFO] Clusters stat ------------')
    # print(df_segmented.info())
    # df_stat_count = df_segmented.groupby('Cluster').size()
    # print(df_stat_count)
    # df_stat_count.plot.bar()
    # plt.show()

    # snake plot approach
    # Transform df_normal as df and add cluster column
    df_normalized = pd.DataFrame(df_normalized,
                                 index=df_original.index,
                                 columns=df_original.columns)
    df_normalized['Cluster'] = df_segmented['Cluster']

    # Melt data into long format
    df_melt = pd.melt(df_normalized.reset_index(),
                      id_vars=['Cluster'],
                      value_vars=['ob-type', 'ob-beds', 'ob-bath', 'ob-sqft',
                                  'lot-size',
                                  'distance-downtown', 'commute-rate', 'crime-rate', 'dog-friendly-rate', 'quiet-rate',
                                  'elem-schools-rate',
                                  'fin-price'],
                      var_name='Metric',
                      value_name='Value')

    # fig, ax = plt.subplots()

    plt.figure(figsize=(30, 6))
    # fig.set_size_inches(30, 6)
    plt.xlabel('Metric')
    plt.ylabel('Value')
    sns.pointplot(data=df_melt, x='Metric', y='Value', hue='Cluster')
    plt.show()
