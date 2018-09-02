from sklearn.cluster import KMeans
# UMAP
import umap
from scipy.sparse.csgraph import connected_components

#  TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
import pandas as pd
import time, datetime
import sys, re

sys.path.append('../../../github/module/')
from preprocessing import factorize_categoricals
from make_file import make_npy, make_feature_set
from convinience_function import get_categorical_features, get_numeric_features
from logger import logger_func
from categorical_encoding import select_category_value_agg


start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
#  logger = logger_func()

unique_id = 'SK_ID_CURR'
target = 'TARGET'

ignore_features = [unique_id, target, 'valid_no', 'is_train', 'is_test']


def kmeans(df, cluster=10):

    print(f'\nKMEANS INPUT DATA:\n{df.head()}')
    seed = 1208

    params = {'n_clusters':cluster,
              'n_init' : 10,
              'max_iter' : 300,
              'tol' : 1e-4,
              'precompute_distances' : 'auto',
              'verbose' : 0,
              'random_state' : seed,
              'copy_x' : True,
              'n_jobs' : -1,
              'algorithm' : 'auto'
              }

    kmeans = KMeans(**params).fit(df)

    df['cluster'] = kmeans.labels_

    df.to_csv(f'../output/embedding/umap_kmeans_3D_{cluster}cluster.csv')

    return df


def t_SNE(data, D):

    params = {'n_jobs':-1, 'n_components':D}
    start_time = time.time()
    logger.info(f't_SNE train start: {start_time}')
    # t-SNE
    tsne_model = TSNE(**params)
    embedding = tsne_model.fit_transform(data)

    print(f't_SNE train end: {time.time() - start_time}')

    return embedding


def UMAP(data, D):

    params = {'n_components':D}
    start_time = time.time()
    logger.info(f'UMAP train starttime: {start_time}')
    # UMAP
    embedding = umap.UMAP(**params).fit_transform(data)

    logger.info(f'UMAP train end. caliculation time: {time.time() - start_time}')

    return embedding


def main():

    #  df = pd.read_csv(f'../output/embedding/umap_kmeans_3D_10cluster.csv', index_col=unique_id)
    #  print(df.drop_duplicates())
    #  sys.exit()

    base = pd.read_csv('../data/base.csv')

    ' 学習に使うfeature_setをmerge '
    prefix = 'AREA_'
    #  path = '../features/embedding/*.npy'
    path = '../features/3_winner/*.npy'
    data = make_feature_set(base, path)
    data.set_index(unique_id, inplace=True)
    data.drop(['is_train', 'is_test', 'valid_no', target], axis=1, inplace=True)

    logger.info(f'\nconcat end\ndata shape: {data.shape}')

    categorical = get_categorical_features(data, [])
    categorical.remove('a_ORGANIZATION_TYPE')
    data = factorize_categoricals(data, categorical)
    data.fillna(-1, inplace=True)
    data = data.replace(np.inf, np.nan)
    data = data.replace(-1*np.inf, np.nan)
    for col in data.columns:
        if col=='a_ORGANIZATION_TYPE':continue
        data[col].fillna(data[col].mean(), inplace=True)

    data_list = [data.copy()]

    for i, df in enumerate(data_list):

        D = 3
        cluster = 15

        mean = df.groupby('a_ORGANIZATION_TYPE').mean()
        std = df.groupby('a_ORGANIZATION_TYPE').std()
        mean.columns = [col+'_mean' for col in mean.columns]
        std.columns = [col+'_std' for col in std.columns]
        df = mean.join(std)

        result = UMAP(df, D)
        logger.info(f'UMAP result shape: {result.shape}')

        #  data = pd.DataFrame(data, columns=['x', 'y'])
        df_emb = pd.DataFrame(result, columns=['x', 'y', 'z'], index=df.index)
        df_emb['a_ORGANIZATION_TYPE'] = df.index
        df_emb = data.reset_index()[[unique_id, 'a_ORGANIZATION_TYPE']].merge(df_emb, on='a_ORGANIZATION_TYPE', how='inner').drop('a_ORGANIZATION_TYPE', axis=1).set_index(unique_id)

        result = kmeans(df_emb, cluster)

        result.rename(columns={'cluster':f'{prefix}{D}D_embd_{cluster}cluster@'}, inplace=True)

        make_npy(result, ignore_list = ignore_features)
        sys.exit()


if __name__ == "__main__":
    main()
