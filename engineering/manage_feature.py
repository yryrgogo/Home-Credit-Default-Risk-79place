import gc
import numpy as np
import pandas as pd
import datetime
from datetime import date, timedelta
import glob
import sys
import re
import shutil
from itertools import combinations
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm
from feature_name_list import application_cat, previous_cat, previous_num
from dimensionality_reduction import UMAP, t_SNE

sys.path.append('../../../github/module/')
from preprocessing import set_validation, split_dataset, factorize_categoricals, get_dummies, data_regulize, max_min_regularize, inf_replace
from load_data import pararell_load_data, x_y_split
from convinience_function import get_categorical_features, get_numeric_features
from logger import logger_func
from make_file import make_feature_set, make_npy
from statistics_info import correlation

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
logger = logger_func()

unique_id = 'SK_ID_CURR'
target = 'TARGET'
ignore_features = [unique_id, target, 'valid_no', 'valid_no_2',
                   'valid_no_3', 'valid_no_4', 'is_train', 'is_test']

pd.set_option('max_columns', 100)


def make_data_columns_table():

    path_list = glob.glob('../data/tmp/*.csv')
    result = pd.DataFrame([])
    for path in path_list:
        if path.count('app') and path.count('summary'):
            dname = 'app'
        elif path.count('bureau') and path.count('summary'):
            dname = 'bureau'
        elif path.count('prev') and path.count('summary'):
            dname = 'prev'
        elif path.count('train'):
            dname = 'app'
        elif path.count('bureau'):
            dname = 'bureau'
        elif path.count('prev'):
            dname = 'prev'
        elif path.count('POS'):
            dname = 'pos'
        elif path.count('inst'):
            dname = 'is'
        elif path.count('credit'):
            dname = 'ccb'

        data = pd.read_csv(path)
        if dname == '':
            dname = re.search(r'/([^/.]*).csv', path).group(1)

        columns = list(data.columns)
        col_length = []
        for col in columns:
            col_length.append(len(col))

        df = pd.Series(columns, name='column').to_frame()
        df['dname'] = dname
        df['length'] = col_length

        ' application_train以外にTARGETがあったら除く '
        if dname != 'app':
            features = df['column'].values
            for f in features:
                if f == target:
                    df = df[df['column'] != target]

        if len(result) == 0:
            result = df.copy()
        else:
            result = pd.concat([result, df])
        logger.info(result.shape)
        logger.info(result.tail())
    result = result.drop_duplicates()
    result.sort_values(by=['dname', 'length'], ascending=False, inplace=True)
    result.to_csv('../data/data_columns_table.csv', index=False)
    sys.exit()


def make_feature_manage_table():

    app_cat_list = application_cat()
    prev_cat_list = previous_cat()
    prev_num_list = previous_num()
    prev_num = pd.Series(prev_num_list, name='prev_num').to_frame()

    table = pd.DataFrame([])
    for app_cat in app_cat_list:
        tmp_table = pd.DataFrame([])
        for prev_cat in prev_cat_list:
            prev_num['prev_cat'] = prev_cat
            if len(tmp_table) == 0:
                tmp_table = prev_num.copy()
            else:
                tmp_table = pd.concat([tmp_table, prev_num], axis=0)

        tmp_table['app_cat'] = app_cat
        if len(table) == 0:
            table = tmp_table
        else:
            table = pd.concat([table, tmp_table], axis=0)

    logger.info(f'table shape: {table.shape}')
    table['make_flg'] = 0
    base = pd.read_csv('../data/base.csv')

    ' 特徴量セットの確認 '
    path = '../features/f_previous_feature/*.npy'
    path_list = glob.glob(path)
    dataset = make_feature_set(base, path)
    dataset = dataset.set_index(unique_id)

    key_list = ['prev_cat', 'app_cat', 'prev_num']
    key_dict = {}
    key_dict = check_loop(key_dict)

    for col in dataset.columns:
        for app_cat in app_cat_list:
            for prev_cat in prev_cat_list:
                for prev_num in prev_num_list:

                    if col.count(app_cat) and col.count(prev_cat) and col.count(prev_num):
                        key = f'{app_cat}_{prev_cat}_{prev_num}'
                        if key_dict[key] == 1:
                            continue
                        elif key_dict[key] == 0:
                            tmp = table.query(f"app_cat=='{app_cat}'").query(
                                f"prev_cat=='{prev_cat}'").query(f"prev_num=='{prev_num}'")
                            tmp['make_flg'] = 1
                            tmp_2 = table[key_list].merge(
                                tmp, on=key_list, how='left')
                            tmp_2.fillna(0, inplace=True)
                            table['make_flg'] += tmp_2['make_flg'].values
                            logger.info(col)
                            logger.info(table.columns)

    table.to_csv(
        f'../output/{start_time[:12]}prev_cat_num_table.csv', index=False)


def make_individual_feature_set():
    ' 集計条件のかぶらないfeature同士を集める '
    key_dict = {}
    bins_list = ['bin20', 'bin15', 'bin10']
    #  bins_list = [15]
    method = 'mean'
    method = 'std'
    path_list = glob.glob('../features/1_second_valid/*.npy')

    key_dict = check_loop(key_dict)
    for bins in bins_list:
        for path in path_list:
            key_dict = check_loop(key_dict, path, 1, method, bins)
    #  print(key_dict)
    sys.exit()


def check_loop(df, path='', code=0, method='', bins='bin99'):
    app_cat_list = application_cat()
    prev_cat_list = previous_cat()
    prev_num_list = previous_num()
    for app_cat in app_cat_list:
        for prev_cat in prev_cat_list:
            for prev_num in prev_num_list:

                key = f'{app_cat}_{prev_cat}_{prev_num}'
                if code == 0:
                    df[key] = 0
                elif code == 1 and path.count(app_cat) and path.count(prev_cat) and path.count(prev_num) and path.count(bins) and path.count(method):
                    if df[key] == 1:
                        continue
                    elif df[key] == 0:
                        print(path)
                        print(prev_num)
                        shutil.move(path, '../features/1_first_valid/')
                        df[key] = 1

    return df


def check_feature_elems(dcols, path):

    prefix_list = ['a_', 'b_', 'ccb_', 'p_', 'is_', 'pos_', 'abp_', 'ap']
    data_list = ['app', 'bureau', 'prev', 'ccb', 'pos', 'is']
    base = pd.read_csv('../data/base.csv')
    df = make_feature_set(base[unique_id].to_frame(),
                          path).set_index(unique_id)
    feature_arr = df.columns

    dcols.sort_values(by=['dname', 'length'], ascending=False, inplace=True)

    ' 各カラムの使用数をカウントする為,辞書でもつ'
    col_dict = {}
    for dname in dcols['dname'].drop_duplicates():
        col_dict[dname] = dname
        tmp = {}
        for col in dcols.query(f"dname=='{dname}'")['column']:
            tmp[col] = 0
            col_dict[dname] = tmp

    for f in feature_arr:
        if f[:2] == 'a_' or f[:4] == 'abp_' or f[:3] == 'ap_':
            dcolumns = dcols.query("dname=='app'")['column'].values
            dname = 'app'
        elif f[:2] == 'b_':
            dcolumns = dcols.query("dname=='bureau'")['column'].values
            dname = 'bureau'
        elif f[:2] == 'p_':
            dcolumns = dcols.query("dname=='prev'")['column'].values
            dname = 'prev'
        elif f[:4] == 'ccb_':
            dcolumns = dcols.query("dname=='ccb'")['column'].values
            dname = 'ccb'
        elif f[:4] == 'pos_':
            dcolumns = dcols.query("dname=='pos'")['column'].values
            dname = 'pos'
        elif f[:3] == 'is_':
            dcolumns = dcols.query("dname=='is'")['column'].values
            dname = 'is'

        cnt_col_list = []
        ' まずは元データのカラムをチェック '
        for col in dcolumns:
            if f.count(col):
                logger.info(f'f:{f} dname:{dname} col:{col}')
                col_dict[dname][col] += 1
                cnt_col_list.append(col)

        ' 残りのデータのカラムをチェック '
        tmp_data_list = data_list.copy()
        tmp_data_list.remove(dname)
        for dname in tmp_data_list:
            for col in dcols.query(f"dname=='{dname}'")['column'].values:
                ' 同じカラム名は二度カウントさせない '
                if col in cnt_col_list:
                    continue
                if f.count(col):
                    logger.info(f'f:{f} dname:{dname} col:{col}')
                    col_dict[dname][col] += 1
                    cnt_col_list.append(col)

    result = pd.DataFrame(col_dict).T.stack().reset_index().rename(
        columns={'level_0': 'dname', 'level_1': 'feature', 0: 'cnt'})

    result.to_csv(
        f'../eda/{start_time[:11]}_feature_set_elems.csv', index=False)


def check_feature_detail(path):

    base = pd.read_csv('../data/base.csv')
    df = make_feature_set(base[unique_id].to_frame(), path)

    for col in df.columns:
        if col in ignore_features:
            continue
        print(df[col].drop_duplicates().sort_values())


def corr_check(df=[]):

    corr = df.corr(method='pearson')
    print(corr)
    sys.exit()
    #  corr = corr.sort_index(axis=1)
    #  corr = corr.unstack().reset_index().rename(columns={'level_0': 'feature', 'level_1':'feature_2', 0:'corr'})

    importance = pd.read_csv(
        '../output/cv_feature1099_importances_auc_0.8072030486159842.csv')[['feature', 'rank']]
    df = importance.query("rank<=200")

    #  df = corr.merge(importance, on='feature', how='inner')
    #  df = df.query("rank<=200")
    #  importance.rename(columns={'feature':'feature_2', 'rank':'rank_2'}, inplace=True)
    #  df = df.merge(importance, on='feature_2', how='inner')
    #  df['corr'] = np.abs(df['corr'])

    feature_list = df['feature'].drop_duplicates().values

    base = pd.read_csv('../data/base.csv')
    path_list = glob.glob('../features/3_winner/*.npy')

    for i in range(20):
        start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
        #  tmp_feature_list = df.query(f'''feature=="{feat}"''')['feature_2'].values
        seed = np.random.randint(0, 100000)+605
        np.random.seed(seed=seed)
        emb_list = np.random.choice(feature_list, 10, replace=False)

        use_paths = []
        for elem in emb_list:
            for path in path_list:
                if path.count(elem):
                    use_paths.append(path)
                    break

        logger.info(f'SELECT PATH: {len(use_paths)}')

        data = make_feature_set(base[unique_id].to_frame(
        ), path='', use_feature=use_paths).set_index(unique_id)

        for col in data.columns:
            data[col] = data[col].replace(np.inf, np.nan)
            data[col] = data[col].replace(-1*np.inf, np.nan)
            data[col] = data[col].fillna(data[col].median())

        #  df_emb = UMAP(data=data, D=2)
        df_emb = t_SNE(data=data, D=2)
        df_emb = pd.DataFrame(data=df_emb, columns=['x', 'y'])
        df_emb[unique_id] = base[unique_id].values

        df_emb.to_csv(
            f'../output/{start_time}_umap_seed{seed}.csv', index=False)


def corr_selection():
    ' featureの相関を見る '
    #  keras_1 = pd.read_csv('../output/20180821_065448_559features_auc0.78203_keras_prediction.csv')
    #  keras_2 = pd.read_csv('../output/20180821_065448_830features_auc0.77992_keras_prediction.csv')
    #  keras_1.fillna(0, inplace=True)
    #  keras_2.fillna(0, inplace=True)
    #  t_value_1 = keras_1[target].values
    #  p_value_1 = keras_1['prediction'].values
    #  t_value_2 = keras_2[target].values
    #  p_value_2 = keras_2['prediction'].values
    #  keras_1['prediction'] = t_value_1 + p_value_1
    #  keras_2['prediction'] = t_value_2 + p_value_2
    #  keras_1['prediction_2'] = keras_2['prediction']
    #  result = keras_1

    #  for col in second.columns:
    #      second.rename(columns = {col:f'1_{col}'}, inplace=True)
    #  result = second.join(first)

    feature_1 = df['feature'].values
    feature_2 = df['feature_2'].values

    ' 先頭行を除く '
    emb_list = []
    all_list = []

    for f1 in feature_1:
        for f2 in feature_2:
            corr_list = [f1, f2]
            elem_cnt = 0
            for tmp_f1 in df.query(f'''feature=="{f1}"''')['feature_2'].values:
                for tmp_f2 in df.query(f'''feature_2=="{f2}"''')['feature'].values:
                    tmp_c1 = df.query(f'''feature=="{f1}"''').query(
                        f'''feature_2=="{tmp_f1}"''')['corr'].values[0]
                    tmp_c2 = df.query(f'''feature_2=="{f2}"''').query(
                        f'''feature=="{tmp_f2}"''')['corr'].values[0]

                    if tmp_c1 >= tmp_c2 and tmp_f2 not in corr_list:
                        corr_list.append(tmp_f2)
                    elif tmp_c1 < tmp_c2 and tmp_f1 not in corr_list:
                        corr_list.append(tmp_f1)
                    else:
                        pass
                        #  logger.info('This feature already exist.')

                    if len(corr_list) >= 5:
                        break
                if len(corr_list) >= 5:
                    break

            for elem in corr_list:
                if elem in all_list:
                    elem_cnt += 1
                if elem_cnt == 3:
                    elem_cnt = 999
                    break

            if elem_cnt == 999:
                logger.info('THIS IS SIMILER SET. CONTINUE')
                continue

            logger.info(f'Add: {corr_list}')
            emb_list.append(corr_list)

            all_list += corr_list
            all_list = list(set(all_list))

            if len(emb_list) > 10:
                break
        if len(emb_list) > 10:
            break

    print(len(emb_list))
    for i in emb_list:
        print(len(i))
    #  corr.to_csv(f'../output/{start_time[:11]}p_TARGET_corr.csv')
    sys.exit()


def main():

    #  data = pd.read_csv('../data/FULL_OLD_BURO_MMM.csv')
    #  path = '../features/3_winner/*.npy'
    path = '../features/1_third_valid/*.npy'
    path = '../features/history/*.npy'
    base = pd.read_csv('../data/base.csv')
    data = make_feature_set(base[unique_id].to_frame(), path)
    #  data = make_feature_set(base['is_train'].to_frame(), path)
    #  data = make_feature_set(base[[unique_id, target]], path)
    data = make_feature_set(base[[unique_id, target, 'is_train', 'is_test', 'valid_no_4']], path)
    logger.info(data.shape)

    #  for col in data.columns:
    #      logger.info(f'\n{col}: {len(data[col][data[col]==np.inf])}')
    ' 特徴量セットの正規化verを作成する（NN / LR / EXT向け） '
    data = data_regulize(df=data, na_flg=1, inf_flg=1, mm_flg=1, float16_flg=1, ignore_feature_list=ignore_features, logger=logger)
    data.to_csv('../data/regular_no_app_2.csv', index=False)
    logger.info(data.shape)
    #  logger.info(data.head())

    for col in data.columns:
        logger.info(data[col].drop_duplicates().sort_values)
    data.to_csv('../data/nn_history.csv', index=False)
    sys.exit()

    ' 正規化 '

    ' infのreplace '

    ' NaN埋め '

    #  logger.info(f'\n{col}: {len(data[col][data[col]==np.inf])}')

    #  check_feature_detail(path)
    #  sys.exit()

    ' 各データ名とそのカラム名をテーブルにする '
    #  make_data_columns_table()
    #  sys.exit()

    ' 特徴量セットの構成を検証する→作ったテーブルに使用回数をカウント '
    #  dcols = pd.read_csv('../data/data_columns_table.csv')
    #  check_feature_elems(dcols, path)
    #  sys.exit()

    #  make_feature_manage_table()
    #  make_individual_feature_set()
    pred_1 = pd.read_csv('../submit/20180825_204_submit_lgb_rate0.02_1099features_CV0.8082070133827914_LB0.806_early150_iter20000_regular_dima_params.csv').set_index(unique_id).rename(columns={target:f'{target}_cv8082'})
    pred_2 = pd.read_csv('../submit/20180827_072_submit_lgb_rate0.02_1099features_CV0.80606353200866_LB_early150_iter20000_dart.csv').set_index(unique_id).rename(columns={target:f'{target}_cv8060_dart'})
    pred_3 = pd.read_csv('../submit/20180825_224_submit_lgb_rate0.02_1099features_CV0.8072030486159842_LB0.808_early150_iter20000_no_regular_dima_params.csv').set_index(unique_id).rename(columns={target:f'{target}_cv8072'})
    pred = pred_1.join(pred_2).join(pred_3)
    corr_check(df=pred)


if __name__ == '__main__':
    main()
