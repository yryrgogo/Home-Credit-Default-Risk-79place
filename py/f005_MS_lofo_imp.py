import os
import sys
import glob
import pandas as pd
import numpy as np
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
sys.path.append(f"../py/")
import MS_utils
import utils, ml_utils, kaggle_utils
from utils import logger_func
from sklearn.model_selection import KFold
from lofo import LOFOImportance, plot_importance
import seaborn as sns
from matplotlib import pyplot as plt
import datetime
%matplotlib inline

pd.set_option('max_rows', 200)

# Columns
key, target, ignore_list = MS_utils.get_basic_var()

# Basic Args
n_jobs = 36
seed = 1208
fold_n = 3
sampling = np.int(sys.argv[1])
lofo_list = []


# Train Test Load
base = utils.read_df_pkl('../input/base_Av*')
train, test = MS_utils.get_dataset(base=base, feat_path='../features/4_winner/*.gz')
vi_col = 'f000_AvSigVersion'
train.sort_values(vi_col, inplace=True)
test.sort_values(vi_col, inplace=True)
train.drop(vi_col, axis=1, inplace=True)
test.drop(vi_col, axis=1, inplace=True)

# extract a sample of the data
group_list = sorted(train['country_group'].unique().tolist())


features = [col for col in train.columns if col.count('f0')]
for group in group_list:
    start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

    if sampling:
        sample_train = train[train['country_group']==group].sample(n=sampling)
    else:
        sample_train = train[train['country_group']==group]
    print(f"Group {group} | Train Shape: {sample_train.shape}")

    # define the validation scheme
    cv = KFold(n_splits=fold_n, shuffle=False, random_state=seed)

    # define the binary target and the features

    # define the validation scheme and scorer. The default model is LightGBM
    lofo_imp = LOFOImportance(sample_train, features, target, cv=cv, scoring="roc_auc", n_jobs=n_jobs)

    # get the mean and standard deviation of the importances in pandas format
    importance_df = lofo_imp.get_importance()
    # plot the means and standard deviations of the importances
    fig.patch.set_facecolor('white')
    sns.set_style("whitegrid")
    logger.info(importance_df)
    plot_importance(importance_df, figsize=(10, 20))
    plt.savefig(f'../lofo_png/{start_time[4:12]}_lofo_row{len(sample_train)}_feat{len(sample_train.columns)}_gr{group}.png')
    plt.show()
    
    importance_df.columns = [f"gr{group}_{col}" if col.count('imp') else col  for col in importance_df.columns]
    lofo_list.append(importance_df)
    
print(len(lofo_list))

#========================================================================
# Result
for num, df in enumerate(lofo_list):
    
    df.set_index('feature', inplace=True)
    if num==0:
        base = df.copy()
    else:
        base = base.join(df)
        
mean_cols = [col for col in base.columns if col.count('_mean')]
base_mean = base[mean_cols]
for col in mean_cols:
    base_mean[col] = base_mean[col].map(lambda x: np.round(x, 4))
base_mean['cnt_minus'] = (base[mean_cols]<0).sum(axis=1)
base_mean.to_csv(f"../output/0306_lofo_group_fold{fold_n}_feat{len(features)}_sample{sampling}.gz")
#========================================================================