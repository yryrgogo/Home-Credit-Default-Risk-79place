code=0
#  code=1
#  code=2
import pandas as pd
import os
import shutil
import sys
import glob
import re

unique_id = 'SK_ID_CURR'
p_id = 'SK_ID_CURR'
target = 'TARGET'
ignore_features = [unique_id, target, 'valid_no', 'valid_no_4', 'is_train', 'is_test', 'SK_ID_PREV']


def move_to_second_valid(best_select=[], rank=0, key_list=[]):

    if len(best_select)==0:
        #  best_select = pd.read_csv('../output/use_feature/feature869_importance_auc0.806809193200456.csv')
        best_select = pd.read_csv('../output/cv_feature1912_importances_auc_0.7940528815911253.csv')
        #  best_select = pd.read_csv('../output/cv_feature1330_importances_auc_0.8066523340763816.csv')
        #  best_select = pd.read_csv('../output/cv_feature1099_importances_auc_0.8072030486159842.csv')
        #  best_feature = best_select['feature'].values
        #  best_feature = best_select.query("rank>=750")['feature'].values
        #  best_feature = best_select.query("rank>=100")['feature'].values
        #  best_feature = best_select.query("rank>=2000")['feature'].values
        best_feature = best_select.query("rank>=1000")['feature'].values
        #  best_feature = best_select.query("rank>=1047")['feature'].values
        best_feature = [col for col in best_feature if col.count('max') or col.count('min') or col.count('sum')]
        #  best_feature = [col for col in best_feature if col.count('impute')]

        if len(best_feature)==0:
            sys.exit()
        for feature in best_feature:
            if feature not in ignore_features:
                try:
                    shutil.move(f'../features/4_winner/{feature}.fp.gz', '../features/2_second_valid/')
                    #  shutil.move(f'../features/go_dima/{feature}.npy', '../features/1_second_valid/')
                except FileNotFoundError:
                    print(f'FileNotFound. : {feature}.fp')
                    pass
        print(f'move to third_valid:{len(best_feature)}')

    else:
        tmp = best_select.query(f"rank>={rank}")['feature'].values
        for key in key_list:
            best_feature = [col for col in tmp if col.count(key)]

            if len(best_feature)==0:
                sys.exit()
            for feature in best_feature:
                if feature not in ignore_features:
                    shutil.move(f'../features/3_winner/{feature}.npy', '../features/1_third_valid')
            print(f'move to third_valid:{len(best_feature)}')


def move_to_use():

    #  best_select = pd.read_csv('../output/cv_feature1476_importances_auc_0.8091815613330919.csv')
    best_select = pd.read_csv('../output/cv_feature2715_importances_auc_0.7939371168035297.csv')
    #  best_select = pd.read_csv('../output/cv_feature1194_importances_auc_0.809452251037472.csv')
    best_feature = best_select['feature'].values
    #  best_feature = best_select.query('flg_2==0')['feature'].values
    #  best_feature = best_select.query('flg==0')['feature'].values

    #  path_list_imp = glob.glob('../features/3_winner/*.npy')
    #  impute_list = []
    #  for path in path_list_imp:
    #      imp_name = re.search(r'/([^/.]*).npy', path).group(1)[:-7]
    #      impute_list.append(imp_name)
    #  best_feature = dima_list

    path_list = glob.glob('../features/2_second_valid/*.fp.gz')
    #  path_list = glob.glob('../features/3_third_valid/*.fp')

    for path in path_list:
        filename = re.search(r'/([^/.]*).fp.gz', path).group(1)
        if filename in best_feature:
            shutil.move(path, '../features/4_winner/')


def move_feature(feature_name, move_path='../features/9_delete'):

    try:
        shutil.move(f'../features/4_winner/{feature_name}.fp.gz', move_path)
    except FileNotFoundError:
        print(f'FileNotFound. : {feature_name}.fp.gz')
        pass



def main():
    if code==0:
        move_to_second_valid()
    elif code==1:
        move_to_use()
    elif code==2:
        move_file()


if __name__ == '__main__':

    main()
