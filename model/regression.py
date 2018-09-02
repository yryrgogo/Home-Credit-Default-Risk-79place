import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import r2_score
import datetime
from tqdm import tqdm
import sys
from params_lgbm import train_params_0729
sys.path.append('../../../github/module')
from load_data import load_data, x_y_split
from preprocessing import set_validation, split_dataset

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())


' データセットからそのまま使用する特徴量 '
unique_id = 'SK_ID_CURR'
target = 'TARGET'
ignore_features = [unique_id, target, 'valid_no', 'valid_no_2', 'valid_no_3', 'valid_no_4', 'is_train', 'is_test']
val_col = 'valid_no_4'

"""Model Parameter"""
metric = 'l2'

early_stopping_rounds = 150
num_iterations = 7000

fix_params = train_params_0729()
fix_params['objective'] = 'regression'
fix_params['metric'] = 'l2'
fix_params['learning_rate'] = 0.04


def sc_metrics(test, pred):
    if metric == 'l2':
        return r2_score(test, pred)


def cross_validation(logger, train, test, target, categorical_feature):
    '''
    Explain:
        交差検証を行う.
        必要な場合はグリッドサーチでパラメータを探索する.
    Args:
    Return:
    '''

    list_score = []
    list_best_iterations = []
    best_params = None

    if metric == 'l2':
        best_score = 0
    elif metric == 'logloss':
        best_score = 100

    x_train, y_train = x_y_split(train, target)
    x_test, y_test = x_y_split(test, target)

    use_cols = list(x_train.columns)

    now = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

    for params in tqdm(list(ParameterGrid(all_params))):
        logger.info('params: {}'.format(params))

        lgb_train = lgb.Dataset(data=x_train, label=y_train)
        lgb_eval = lgb.Dataset(data=x_val, label=y_val)

        ' 学習 '
        reg = lgb.train(fix_params,
                        lgb_train,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=150,
                        verbose_eval=200
                        )

        y_pred = reg.predict(x_test)
        sc_score = sc_metrics(y_test, y_pred)

        list_score.append(sc_score)
        list_best_iterations.append(reg.best_iteration_)
        logger.info('{}: {}'.format(metric, sc_score))

        params['n_estimators'] = int(np.mean(list_best_iterations))
        sc_score = np.mean(list_score)
        if metric == 'logloss':
            if best_score > sc_score:
                best_score = sc_score
                best_params = params
        elif metric == 'l2':
            if best_score < sc_score:
                best_score = sc_score
                best_params = params

        logger.info('current {}: {}  best params: {}'.format(
            metric, best_score, best_params))

    logger.info('CV best score : {}'.format(best_score))
    logger.info('CV best params: {}'.format(best_params))

    # params output file
    df_params = pd.DataFrame(best_params, index=['params'])
    df_params.to_csv(f'../output/{start_time[:11]}_best_params_{metric}_{best_score}.csv', index=False)

    lgb_train = lgb.Dataset(data=x_train, label=y_train)
    lgb_eval = lgb.Dataset(data=x_val, label=y_val)

    ' 学習 '
    reg = lgb.train(fix_params,
                    lgb_train,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=150,
                    verbose_eval=200
                    )

    ' 予測 '
    y_pred = reg.predict(x_val[use_cols])

    # feature importance output file
    feim_result = pd.Series(reg.feature_importance, name='importance')
    feature_name = pd.Series(use_cols, name='feature')
    features = pd.concat([feature_name, feim_result], axis=1)
    features.sort_values(by='importance', ascending=False, inplace=True)

    sc_score = sc_metrics(y_test, y_pred)
    list_score.append(sc_score)
    features.to_csv('../output/{}_feature_importances_{}_{}.csv'.format(
        start_time[:11], metric, sc_score), index=False)

    mean_score = np.mean(list_score)
    logger.info('CV & TEST mean {}: {}  best_params: {}'.format(
        metric, mean_score, best_params))


def validation(logger, dataset, val_no, target, categorical, viz_flg=0, pts_score='0', best_score=0):
    '''
    Explain:
        trainセットとvalidationセットを読み込み、モデルを検証する。
        必要な場合は予測結果を探索する為、入力データに予測値を結合したデータを返す。
    Args:
    Return:
    '''
    # 学習用、テスト用データセットを作成
    dataset = dataset.set_index('SK_ID_CURR')
    train, valid = split_dataset(dataset, val_no)
    x_train, y_train = x_y_split(train, target)
    x_val, y_val = x_y_split(valid, target)

    y_train = np.log1p(y_train)
    y_val = np.log1p(y_val)

    ' 全サンプルに対する予測を出力する '
    x_all, y_all = x_y_split(dataset, target)

    use_cols = list(x_train.columns.values)

    lgb_train = lgb.Dataset(data=x_train, label=y_train)
    lgb_eval = lgb.Dataset(data=x_val, label=y_val)

    ' 学習 '
    reg = lgb.train(fix_params,
                    lgb_train,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=10000,
                    verbose_eval=200
                    )

    ' 予測 '
    y_pred = reg.predict(x_val[use_cols])

    y_pred = np.expm1(y_pred)
    y_val = np.expm1(y_val)

    ' スコア '
    sc_score = sc_metrics(y_val, y_pred)

    ' 全体予測 '
    all_pred = reg.predict(x_all[use_cols])
    all_pred = np.expm1(all_pred)

    logger.info('prediction score: {}'.format(sc_score))

    ' feature importance output '
    ftim = pd.DataFrame({'feature': use_cols, 'importance': reg.feature_importance})
    ftim['score'] = sc_score

    ' best_scoreを更新したら、全サンプルに予測値をつけて出力 '
    if sc_score > best_score:
        ftim.to_csv(f"../prediction/use_features/{start_time[:11]}_valid{val_no}_use_{len(use_cols)}col_auc_{str(sc_score).replace('.', '_')}.csv", index=False)
        y_pred = reg.predict(x_all[use_cols])
        x_all = x_all.reset_index()[[unique_id, 'valid_no']]
        x_all['prediction'] = y_pred
        x_all['valid_no'] = x_all['valid_no'].map(lambda x:val_no if x==val_no else 0)
        x_all[f'score_cv{val_no}'] = sc_score
        x_all.to_csv(f"../prediction/{start_time[:11]}_valid{val_no}_{len(use_cols)}col_auc_{str(sc_score).replace('.', '_')}.csv", index=False)


    return ftim, all_pred


def prediction(logger, train, test, target, categorical):
    '''
    Explain:
    Args:
    Return:
    '''

    ' explain/target split '
    x_train, y_train = x_y_split(train, target)
    use_cols = x_train.columns.values

    ' 値の差の重みを揃える為、対数変換しておく '
    y_train = np.log1p(y_train)

    lgb_train = lgb.Dataset(data=x_train, label=y_train)

    ' 学習 '
    reg = lgb.train(fix_params,
                    lgb_train
                    )

    ' スコアを出力 '
    y_pred = reg.predict(x_train)
    ' 対数変換を元に戻す '
    y_train = np.expm1(y_train)
    y_pred = np.expm1(y_pred)
    sc_score = sc_metrics(y_train, y_pred)

    logger.info(f'prediction {metric}: {sc_score}')

    ' 予測 '
    y_pred = reg.predict(test[use_cols])
    y_pred = np.expm1(y_pred)

    return y_pred


def cross_prediction(logger, train, test, target, categorical_feature=[], val_col='valid_no'):
    '''
    Explain:
        交差検証を行い予測を行う.
    Args:
    Return:
    '''

    list_score = []
    list_pred = []
    list_best_iterations = []
    best_params = None

    if metric == 'auc':
        best_score = 0
    elif metric == 'logloss':
        best_score = 100

    # 学習用、テスト用データセットを作成

    valid_list = list(train[val_col].drop_duplicates().values)
    try:
        valid_list.remove(-1)
    except ValueError:
        pass
    prediction = np.array([])

    for val_no in valid_list:

        trn_set, valid = split_dataset(train, val_no, val_col=val_col)
        x_train, y_train = x_y_split(trn_set, target)
        x_val, y_val = x_y_split(valid, target)

        y_train = np.log1p(y_train)
        y_val = np.log1p(y_val)

        use_cols = list(x_train.columns)

        lgb_train = lgb.Dataset(data=x_train, label=y_train)
        lgb_eval = lgb.Dataset(data=x_val, label=y_val)
        fix_params['num_iterations'] = num_iterations


        ' 学習 '
        try:
            clf = lgb.train(fix_params,
                            lgb_train,
                            valid_sets=lgb_eval,
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=200,
                            #  categorical_feature = categorical_feature
                            )
        except TypeError:
            return np.zeros(len(test)), 0

        y_pred = clf.predict(x_val)
        sc_score = sc_metrics(y_val, y_pred)

        y_train = np.expm1(y_train)
        y_val = np.expm1(y_val)
        y_pred = np.expm1(y_pred)

        list_score.append(sc_score)
        list_best_iterations.append(clf.current_iteration())
        logger.info(f'validation: {val_no} / {metric}: {sc_score}')

        test_pred = clf.predict(test[use_cols])

        if len(prediction)==0:
            prediction = test_pred
        else:
            prediction += test_pred


    cv_score = np.mean(list_score)
    logger.info(f'\nCross Validation End\nCV score : {cv_score}')

    prediction = prediction / len(valid_list)

    return prediction, cv_score
