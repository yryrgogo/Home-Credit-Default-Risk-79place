import os
import re
import gc
import sys
import glob
import shutil
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
#========================================================================
# original library 
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
import utils, ml_utils
#========================================================================

key = 'MachineIdentifier'
target = 'HasDetections'
seed = 1208
ignore_list = [key, target, 'country_os_group', 'f000_AvSigVersion']
remove_cols = ['PuaMode', 'Census_ProcessorClass', 'Census_IsWIMBootEnabled', 'IsBeta', 'Census_IsFlightsDisabled', 'Census_IsFlightingInternal', 'AutoSampleOptIn', 'Census_ThresholdOptIn', 'SMode', 'Census_IsPortableOperatingSystem', 'Census_DeviceFamily', 'UacLuaenable', 'Census_IsVirtualDevice', 'Platform', 'Census_OSSkuName', 'Census_OSInstallLanguageIdentifier', 'Processor']


def move_same_feature(move_path='../features/4_winner/*.gz', rm_name='', ok_name=''):
    move_path_list = glob.glob(move_path)
    tmp_list = []

    for path in move_path_list:
        filename = re.search(r'/([^/.]*).gz', path).group(1)[5:]
        if filename not in tmp_list:
            tmp_list.append(filename)
        else:
            path = path.replace(ok_name, rm_name)
            print(path)
            shutil.move(path, '../features/no_use/')


def move_no_use_feature(move_path='../features/4_winner/*.gz'):
    move_path_list = glob.glob(move_path)

    for path in move_path_list:
        for col in remove_cols:
            if path.count(col):
                shutil.move(path, '../features/no_use/')
try:
    if sys.argv[1]=='1':
        move_no_use_feature()
    if sys.argv[1]=='2':
        move_same_feature(rm_name=sys.argv[2], ok_name=sys.argv[3])
except IndexError:
    pass


def get_basic_var():
    # 使用しないfeatureを追加する
    ignore_path_list = glob.glob('../features/no_use/*.gz')
    for path in ignore_path_list:
        filename = re.search(r'/([^/.]*).gz', path).group(1)
        ignore_list.append(filename)

    return key, target, ignore_list


def get_dataset(is_debug=False, is_cat_encode=True, feat_path='../features/4_winner/*.gz', base=[]):
    feat_path_list = glob.glob(feat_path)
    #  feat_path_list += glob.glob('../features/5_tmp/*.gz')

    train, test = ml_utils.get_train_test(feat_path_list=feat_path_list, target=target, base=base)
    print(train.shape, test.shape)

    #  if is_debug:
    #      train = train.head(10000)
    #      test = test.head(500)

    if is_cat_encode:
        #========================================================================
        # Categorical Encode
        cat_cols = utils.get_categorical_features(df=train, ignore_list=ignore_list)
        print(f"Categorical: {cat_cols}")

        #Fit LabelEncoder
        for col in cat_cols:
            # 最も頻度の多いカテゴリでimpute
            max_freq = list(train[col].value_counts().index)[0]
            train[col].fillna(max_freq, inplace=True)
            test[col].fillna(max_freq, inplace=True)
            le = LabelEncoder().fit(pd.concat([train[col], test[col]], axis=0).value_counts().index.tolist())
            train[col] = le.transform(train[col])
            test[col]  = le.transform(test[col])
        #========================================================================

    print(train.shape, test.shape)

    return train, test


def get_kfold(is_group=True, base=[], fold_n=5, fold_seed=1208):

    train = base[~base[target].isnull()]
    del base
    gc.collect()
    if is_group:
        group_list = train['country_group'].unique()
        valid_list = []
        for group in group_list:
            tmp = train[train['country_group']==group][[key, target]]
            valid_list.append(tmp)

        kfold = utils.get_kfold(fold_seed=fold_seed, valid_list=valid_list, key=key, target=target)

    else:
        kfold = ml_utils.get_kfold(fold_n=fold_n, fold_type='stratified', seed=seed, train=train, Y=train[target])

    return kfold


def save_feature(prefix, df_feat, dir_path='../features/1_first_valid', feat_check=False):

    ignore_features = ['ID_code', 'target', 'index']
    length = len(df_feat)
    if feat_check:
        for col in df_feat.columns:
            if col in ignore_features:
                continue
            null_len = df_feat[col].dropna().shape[0]
            if length - null_len>0:
                print(f"{col}  | null shape: {length - null_len}")

            max_val = df_feat[col].max()
            min_val = df_feat[col].min()
            if max_val==np.inf or min_val==-np.inf:
                print(f"{col} | max: {max_val} | min: {min_val}")
        sys.exit()

    for col in df_feat.columns:
        if col in ignore_features: continue

        feature = df_feat[col].values.astype('float32')

        if prefix[0]=='4':

            inf_max = np.max(feature)
            inf_min = np.min(feature)

            if inf_max == np.inf:
                v_max = np.max(np.where(feature==inf_max, np.mean(feature), feature))
                feature = np.where(feature==inf_max, v_max, feature)
            if inf_min == -np.inf:
                v_min = np.min(np.where(feature==inf_min, np.mean(feature), feature))
                feature = np.where(feature==inf_min, v_min, feature)

            feature = pd.Series(feature)
            length = len(feature)
            null_len = feature.dropna().shape[0]
            if length - null_len==0:
                pass

            else:
                if col.count('month_lag'):
                    val_min = np.min(feature)
                    feature = np.where(feature!=feature, val_min-1, feature)
                elif col.count('month_diff'):
                    val_max = np.max(feature)
                    feature = np.where(feature!=feature, val_max-1, feature)
                else:
                    for val_min in np.sort(feature):
                        if not(val_min==val_min):
                            continue
                        else:
                            break
                    feature = np.where(feature!=feature, val_min-1, feature)

        #  if feature.shape[0] != 400000:
        #      print(col)
        #      sys.exit()
        col = col.replace('.', '_')
        feat_path = f'{dir_path}/{prefix}_{col}@'
        if os.path.exists(feat_path): continue
        elif os.path.exists( f'../features/2_second_valid/{prefix}_{col}@'): continue
        elif os.path.exists( f'../features/3_third_valid/{prefix}_{col}@'): continue
        elif os.path.exists( f'../features/4_winner/{prefix}_{col}@'): continue
        elif os.path.exists( f'../features/5_tmp/{prefix}_{col}@'): continue
        elif os.path.exists( f'../features/6_subset/{prefix}_{col}@'): continue
        elif os.path.exists( f'../features/7_escape/{prefix}_{col}@'): continue
        elif os.path.exists( f'../features/8_ensemble/{prefix}_{col}@'): continue
        elif os.path.exists( f'../features/9_gdrive/{prefix}_{col}@'): continue
        elif os.path.exists( f'../features/all_features/{prefix}_{col}@'): continue
        else:
            utils.to_pkl_gzip(path=feat_path, obj=feature)


def get_feature_set(feat_path='../features/all_features/*.gz', feat_key='', is_debug=False, is_cat_encode=True):
    feat_path_list = glob.glob(feat_path)

    path_list = []
    for path in feat_path_list:
        filename = re.search(r'/([^/.]*).gz', path).group(1)
        if path.count(feat_key) and feat_key[:4]==filename[:4]:
            path_list.append(path)

    train, test = ml_utils.get_train_test(feat_path_list=path_list, target=target)
    print(train.shape, test.shape)

    if is_debug:
        train = train.head(10000)
        test = test.head(500)

    if is_cat_encode:
        #========================================================================
        # Categorical Encode
        cat_cols = utils.get_categorical_features(df=train, ignore_list=ignore_list)
        print(f"Categorical: {cat_cols}")

        #Fit LabelEncoder
        for col in cat_cols:
            # 最も頻度の多いカテゴリでimpute
            max_freq = list(train[col].value_counts().index)[0]
            train[col].fillna(max_freq, inplace=True)
            test[col].fillna(max_freq, inplace=True)
            le = LabelEncoder().fit(pd.concat([train[col], test[col]], axis=0).value_counts().index.tolist())
            train[col] = le.transform(train[col])
            test[col]  = le.transform(test[col])
        #========================================================================

    return train, test
