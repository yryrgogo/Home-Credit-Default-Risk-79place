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
win_path = f'../features/4_winner/'
second_path = '../features/2_second_valid/'
gdrive_path = '../features/9_gdrive/'
ignore_list = []

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

# path rename
#  win_path = '../features/4_winner/*.gz'
#  path_list = glob.glob(win_path)
#  for path in path_list:
#      tmp = utils.read_pkl_gzip(path)
#      if path.count('107_his_train_'):
#          utils.to_pkl_gzip(path=path.replace(r'107_his_train_', '107_his_train_his_'), obj=tmp)
#      elif path.count('107_his_test_'):
#          utils.to_pkl_gzip(path=path.replace(r'107_his_test_', '107_his_test_his_'), obj=tmp)
#  sys.exit()


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


def move_to_second_valid(best_select=[], path='', rank=0, gain=0, key_list=[]):
    logger = logger_func()
    if len(best_select)==0:
        try:
            if path=='':
                path = sys.argv[2]
        except IndexError:
            pass
        best_select = pd.read_csv(path)
        try:
            select_list = sys.argv[3].split('_')
            select_type = select_list[0]
            select_num = np.int(select_list[1])
        except IndexError:
            pass

        if select_type=='rank':
            best_feature = best_select.query(f"rank<={select_num}")['feature'].values
        elif select_type=='gain':
            best_feature = best_select.query(f"importance>={select_num}")['feature'].values
        try:
            best_feature = [col for col in best_feature if col.count(sys.argv[4])]
        except IndexError:
            best_feature = [col for col in best_feature if col.count('')]

        if len(best_feature)==0:
            sys.exit()

        path_list = glob.glob('../features/*.gz')

        select_path = []
        for feature in best_feature:
            for path in path_list:
                filename = re.search(r'/([^/.]*).gz', path).group(1)
                if filename[:3]=='tra':
                    filename = filename[6:]
                if filename[:3]=='tes':
                    filename = filename[5:]
                #  if path.count(feature) and feature not in ignore_list:
                if feature==filename:
                    select_path.append(path)

        move_path = list(set(path_list) - set(select_path))
        for move in move_path:
            try:
                shutil.move(move, '../features/no_use/')
            except FileNotFoundError:
                logger.info(f'FileNotFoundError: {feature}')
            except shutil.Error:
                logger.info(f'Shutil Error: {feature}')
        print(f'move to third_valid:{len(best_feature)}')


def move_to_use():

    try:
        path = sys.argv[2]
    except IndexError:
        path = ''
    best_select = pd.read_csv(path)
    best_feature = best_select['feature'].values

    win_list = glob.glob(win_path + '*')
    first_list = glob.glob('../features/1_first_valid/*')
    second_list = glob.glob('../features/2_second_valid/*')
    third_list = glob.glob('../features/3_third_valid/*')
    tmp_list = glob.glob('../features/5_tmp/*')
    path_list = third_list
    #  path_list = third_list + tmp_list + win_list
    #  path_list = first_list + second_list + third_list + tmp_list + win_list

    done_list = []
    for feature in best_feature:
        for path in path_list:
            try:
                filename = re.search(r'/([^/.]*).gz', path).group(1)
            except AttributeError:
                continue
            #  if path.count(feature):
            #  if filename==feature:
            if filename.replace('stan_', '')==feature:
                try:
                    shutil.move(path, win_path)
                    #  filename = re.search(r'/([^/.]*).gz', path).group(1)
                    done_list.append(filename)
                except shutil.Error:
                    pass
                    #  shutil.move(path, gdrive_path)
                except FileNotFoundError:
                    pass
                    #  shutil.move(path, gdrive_path)

    logger = logger_func()
    best_feature = [f for f in best_feature]

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
