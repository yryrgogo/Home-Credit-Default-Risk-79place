#========================================================================
# argv[1]: code
# argv[2]: valid_path
# argv[3]: rank
# argv[4]: session / user
#========================================================================
import sys
try:
    code = int(sys.argv[1])
except IndexError:
    code=0
except ValueError:
    pass
try:
    if sys.argv[4]=='one':
        win_path = f'../features/4_winner/'
        tmp_win_path = f'../features/5_tmp/'
        test_path = f'../features/6_test_feat/'
        tmp_test_path = f'../features/7_tmp_test/'
        delete_train_path = f'../features/9_delete/'
        delete_test_path = '../features/8_delete_test/'
        second_path = '../features/2_second_valid/'
    elif sys.argv[4]=='two':
        win_path = f'../features/12_user_train/'
        tmp_win_path = f'../features/10_user_tmp_train/'
        test_path = f'../features/13_user_test/'
        tmp_test_path = f'../features/11_user_tmp_test/'
        delete_train_path = f'../features/14_user_train_delete/'
        delete_test_path = f'../features/15_user_test_delete/'
        second_path = '../features/10_user_tmp_train/'
    else:
        win_path = f'../features/4_winner/'
        tmp_win_path = f'../features/5_tmp/'
        test_path = f'../features/6_test_feat/'
        tmp_test_path = f'../features/7_tmp_test/'
        delete_train_path = f'../features/9_delete/'
        delete_test_path = '../features/8_delete_test/'
        second_path = '../features/2_second_valid/'
except IndexError:
    win_path = f'../features/4_winner/'
    tmp_win_path = f'../features/5_tmp/'
    test_path = f'../features/6_test_feat/'
    tmp_test_path = f'../features/7_tmp_test/'
    delete_train_path = f'../features/9_delete/'
    delete_test_path = '../features/8_delete_test/'
    second_path = '../features/2_second_valid/'
#  code=1
#  code=2
import numpy as np
import pandas as pd
import os
import shutil
import glob
import re
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
import utils
from utils import logger_func


#========================================================================
# Global Variable
from info_home_credit import hcdr_key_cols
key, target, ignore_list = hcdr_key_cols()
#========================================================================


def to_win_dir_Nfeatures(path='../features/1_first_valid/*.gz', N=100):
    path_list = glob.glob(path)
    np.random.seed(1208)
    np.random.shuffle(path_list)
    path_list = path_list[:N]
    for path in path_list:
        try:
            shutil.move('train_'+path, win_path)
            shutil.move('test_'+path, win_path)
        except shutil.Error:
            shutil.move('train_'+path, '../features/9_delete')
            shutil.move('test_'+path, '../features/9_delete')

def move_to_second_valid(best_select=[], path='', rank=0, key_list=[]):
    logger = logger_func()
    if len(best_select)==0:
        try:
            if path=='':
                path = sys.argv[2]
        except IndexError:
            pass
        best_select = pd.read_csv(path)
        try:
            if rank==0:
                rank = int(sys.argv[3])
        except IndexError:
            pass
        best_feature = best_select.query(f"rank>={rank}")['feature'].values
        try:
            best_feature = [col for col in best_feature if col.count(sys.argv[5])]
        except IndexError:
            best_feature = [col for col in best_feature if col.count('')]
        #  best_select = pd.read_csv('../output/use_feature/feature869_importance_auc0.806809193200456.csv')
        #  best_feature = best_select['feature'].values
        #  best_feature = best_select.query("rank>=750")['feature'].values
        #  best_feature = best_select.query("rank>=1047")['feature'].values
        #  best_feature = [col for col in best_feature if col.count('max') or col.count('min') or col.count('sum')]
        #  best_feature = [col for col in best_feature if col.count('impute')]

        if len(best_feature)==0:
            sys.exit()
        for feature in best_feature:
            if feature not in ignore_list:
                try:
                    shutil.move(f"{win_path}train_{feature}.gz", second_path)
                    shutil.move(f"{win_path}test_{feature}.gz", second_path)
                    #  shutil.move(f'../features/go_dima/{feature}.npy', '../features/1_second_valid/')
                except FileNotFoundError:
                    print(f'FileNotFound. : {feature}.gz')
                    pass
                except shutil.Error:
                    logger.info(f'Shutil Error: {feature}')
        print(f'move to third_valid:{len(best_feature)}')

def move_to_use():

    try:
        path = sys.argv[2]
    except IndexError:
        path = ''
    best_select = pd.read_csv(path)
    tmp_best_feature = best_select['feature'].values
    best_feature = []
    for feat in tmp_best_feature:
        best_feature.append('train_'+feat)
        best_feature.append('test_'+feat)

    win_list = glob.glob(win_path + '*.gz')
    for path in win_list:
        filename = re.search(r'/([^/.]*).gz', path).group(1)
        if filename  in ignore_list: continue
        try:
            shutil.move(path, tmp_win_path)
        except shutil.Error:
            shutil.move(path, delete_train_path)

    if sys.argv[4]=='one':
        first_list = glob.glob('../features/1_first_valid/*.gz')
        second_list = glob.glob('../features/2_second_valid/*.gz')
        third_list = glob.glob('../features/3_third_valid/*.gz')
        tmp_list = glob.glob('../features/5_tmp/*.gz')
        path_list = first_list + second_list + third_list + tmp_list
    elif sys.argv[4]=='two':
        path_list = glob.glob(tmp_win_path + '*.gz')
    else :
        first_list = glob.glob('../features/1_first_valid/*.gz')
        second_list = glob.glob('../features/2_second_valid/*.gz')
        third_list = glob.glob('../features/3_third_valid/*.gz')
        tmp_list = glob.glob('../features/5_tmp/*.gz')
        path_list = first_list + second_list + third_list + tmp_list

    done_list = []
    for path in path_list:

        try:
            filename = re.search(r'/([^/.]*).gz', path).group(1)
        except AttributeError:
            print(f"AttributeError: \nPathName: {path}")
            sys.exit()
        #  if filename[5:] in best_feature or filename[6:] in best_feature:
        if filename in best_feature:
            try:
                shutil.move(path, win_path)
                done_list.append(filename)
            except shutil.Error:
                shutil.move(path, delete_train_path)

    logger = logger_func()
    loss_list = set(list(best_feature)) - set(done_list)
    logger.info(f"Loss List:")
    for loss in loss_list:
        logger.info(f"{loss}")


def move_feature(feature_name, move_path='../features/9_delete'):

    try:
        shutil.move(f'../features/4_winner/{feature_name}.gz', move_path)
    except FileNotFoundError:
        print(f'FileNotFound. : {feature_name}.gz')
        pass


def main():
    if code==0:
        move_to_second_valid()
    elif code==1:
        move_to_use()
    elif code==2:
        move_file()
    elif code==4:
        to_win_dir_Nfeatures(N=int(sys.argv[2]))


if __name__ == '__main__':

    main()
