import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
import datetime
from tqdm import tqdm
import sys
sys.path.append('../../../github/module')
from load_data import load_data, x_y_split
from preprocessing import set_validation, split_dataset
from params_lgbm import train_params, train_params_0729, train_params_0815, xgb_params_0814, extra_params, lgr_params
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, ridge
from sklearn.preprocessing import StandardScaler


start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
sc = StandardScaler()

' データセットからそのまま使用する特徴量 '
unique_id = 'SK_ID_CURR'
target = 'TARGET'
ignore_features = [unique_id, target, 'valid_no', 'is_train', 'is_test', 'valid_no_2', 'valid_no_3', 'valid_no_4']



def sc_metrics(test, pred, metric='auc'):
    if metric == 'logloss':
        return log_loss(test, pred)
    elif metric == 'auc':
        return roc_auc_score(test, pred)
    else:
        print('score caliculation error!')


def cross_validation(logger, dataset, target, val_col='valid_no', params=train_params, metric='auc', categorical_feature=[], feim_log=[1], truncate_flg=0, num_iterations=3500, learning_rate=0.04, early_stopping_rounds=150, model_type='lgb'):
    '''
    Explain:
        CVで評価を行う.
    Args:
    Return:
    '''

    list_score = []
    list_best_iterations = []
    best_params = None

    if metric == 'auc':
        best_score = 0
    elif metric == 'logloss':
        best_score = 100

    cv_feim = pd.DataFrame([])
    valid_list = dataset[val_col].drop_duplicates().values
    ' カラム名をソートし、カラム順による影響をなくす '
    dataset.sort_index(axis=1, inplace=True)

    for i, val_no in enumerate(valid_list):

        train, valid = split_dataset(dataset, val_no, val_col=val_col)
        x_train, y_train = x_y_split(train, target)
        x_val, y_val = x_y_split(valid, target)

        use_cols = list(x_train.columns)
        if i==0:
            logger.info(f'\nTrainset Col Number: {len(use_cols)}')

        clf, y_pred = classifier_model(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, params=params, categorical_feature=categorical_feature, num_iterations=num_iterations, early_stopping_rounds=early_stopping_rounds, learning_rate=learning_rate, model_type=model_type)

        sc_score = sc_metrics(y_val, y_pred, metric)
        if val_no==2:
            val_2_score = sc_score

        list_score.append(sc_score)
        #  list_best_iterations.append(clf.current_iteration())
        logger.info(f'validation: {val_no} / {metric}: {sc_score}')

        if truncate_flg==1:
            sum_score = sum(list_score)
            ' 一定基準を満たさなかったら打ち止め '
            #  if val_no==2 and sum_score<0.807:
            #  if val_no==2 and sum_score<0.8110:
            if val_no==2 and sum_score<0.813:
            #  if val_no==2 and sum_score<0.8038:
                return [val_2_score], 0
            #  elif val_no==1 and sum_score<1.6110:
            elif val_no==1 and sum_score<1.6155:
            #  elif val_no==1 and sum_score<1.5967:
                return [val_2_score], 0
            #  elif val_no==5 and (sum_score<2.406 or sc_score<0.8065):
            elif val_no==5 and sum_score<2.3:
            #  elif val_no==5 and (sum_score<2.395 or sc_score<0.798):
                return [val_2_score], 0
            #  elif val_no==3 and sum_score<3.209:
            elif val_no==3 and sum_score<3.1:
            #  elif val_no==3 and sum_score<3.193:
                return [val_2_score], 0

        ' Feature Importance '
        if model_type=='lgb':
            tmp_feim = pd.Series(clf.feature_importance(), name=f'{val_no}_importance')
            feature_name = pd.Series(use_cols, name='feature')
            feim = pd.concat([feature_name, tmp_feim], axis=1)
        elif model_type=='xgb':
            tmp_feim = clf.get_fscore()
            feim = pd.Series(tmp_feim,  name=f'{val_no}_importance').to_frame().reset_index().rename(columns={'index':'feature'})
        elif model_type=='ext':
            tmp_feim = clf.feature_importance_()
            feim = pd.Series(tmp_feim,  name=f'{val_no}_importance').to_frame().reset_index().rename(columns={'index':'feature'})

        if len(cv_feim)==0:
            cv_feim = feim.copy()
        else:
            cv_feim = cv_feim.merge(feim, on='feature', how='inner')

    cv_score = np.mean(list_score)
    logger.info(f'train shape: {x_train.shape}')
    logger.info(f'\nCross Validation End\nCV score : {cv_score}')

    cv_feim['cv_score'] = cv_score

    importance = []
    for val_no in valid_list:
        if len(importance)==0:
            importance = cv_feim[f'{val_no}_importance'].values
        else:
            importance += cv_feim[f'{val_no}_importance'].values

    cv_feim['avg_importance'] = importance / len(valid_list)
    cv_feim.sort_values(by=f'avg_importance', ascending=False, inplace=True)
    cv_feim['rank'] = np.arange(len(cv_feim))+1

    #  cv_feim.to_csv(f'../output/cv_feature_importances_{metric}_{cv_score}.csv')

    return cv_feim, len(use_cols)


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

    ' 全サンプルに対する予測を出力する '
    x_all, y_all = x_y_split(dataset, target)

    use_cols = list(x_train.columns.values)

    lgb_train = lgb.Dataset(data=x_train, label=y_train)
    lgb_eval = lgb.Dataset(data=x_val, label=y_val)

    ' 学習 '
    clf = lgb.train(train_params,
                    lgb_train,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=150,
                    verbose_eval=200
                    )

    y_pred = clf.predict(x_val)

    ' スコア '
    sc_score = sc_metrics(y_val, y_pred)

    logger.info('prediction score: {}'.format(sc_score))

    ' feature importance output '
    ftim = pd.DataFrame({'feature': use_cols, 'importance': clf.feature_importance()})
    ftim['score'] = sc_score

    ' best_scoreを更新したら、全サンプルに予測値をつけて出力（可視化用） '
    if sc_score > best_score:
        ftim.sort_values(by='importance', ascending=False, inplace=True)
        ftim.to_csv(f"../prediction/use_features/{start_time[:11]}_valid{val_no}_use_{len(use_cols)}col_auc_{str(sc_score).replace('.', '_')}.csv", index=False)
        y_pred = clf.predict(x_all[use_cols])
        x_all = x_all.reset_index()[[unique_id, 'valid_no']]
        x_all['prediction'] = y_pred
        x_all['valid_no'] = x_all['valid_no'].map(lambda x:val_no if x==val_no else 0)
        x_all[f'score_cv{val_no}'] = sc_score
        x_all.to_csv(f"../prediction/{start_time[:11]}_valid{val_no}_{len(use_cols)}col_auc_{str(sc_score).replace('.', '_')}.csv", index=False)

    return ftim


def prediction(logger, train, test, target, categorical_feature=[], metric='auc', params={}, num_iterations=20000, learning_rate=0.02, model_type='lgb'):

    x_train, y_train = x_y_split(train, target)
    use_cols = x_train.columns.values
    use_cols = [col for col in use_cols if col not in ignore_features and not(col.count('valid_no'))]
    test = test[use_cols]

    ' 学習 '
    clf, y_pred = classifier_model(x_train=x_train[use_cols], y_train=y_train, params=params, test=test, categorical_feature=categorical_feature, num_iterations=num_iterations, learning_rate=learning_rate, model_type=model_type)

    return y_pred, len(use_cols)


def cross_prediction(logger, train, test, target, categorical_feature=[], val_col='valid_no', metric='auc', params={}, num_iterations=20000, learning_rate=0.02, early_stopping_rounds=150, model_type='lgb'):
    '''
    Explain:
        交差検証を行う.
        必要な場合はグリッドサーチでパラメータを探索する.
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


    valid_list = train[val_col].drop_duplicates().values
    prediction = np.array([])
    cv_feim = pd.DataFrame([])

    ' カラム名をソートし、カラム順による精度への影響をなくす '
    train.sort_index(axis=1, inplace=True)
    test.sort_index(axis=1, inplace=True)

    for i, val_no in enumerate(valid_list):

        trn_set, valid = split_dataset(train, val_no, val_col=val_col)
        x_train, y_train = x_y_split(trn_set, target)
        x_val, y_val = x_y_split(valid, target)

        use_cols = list(x_train.columns)
        if i==0:
            logger.info(f'\nTrainset Col Number: {len(use_cols)}')

        if model_type=='xgb':
            " XGBは'[]'と','と'<>'がNGなのでreplace "
            if i==0:
                test = test[use_cols]
            use_cols = []
            for col in x_train.columns:
                use_cols.append(col.replace("[", "-q-").replace("]", "-p-").replace(",", "-o-"))
            x_train.columns = use_cols
            x_val.columns = use_cols
            test.columns = use_cols

        clf, y_pred = classifier_model(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, params=params, categorical_feature=categorical_feature, num_iterations=num_iterations, early_stopping_rounds=early_stopping_rounds, learning_rate=learning_rate, model_type=model_type)

        sc_score = sc_metrics(y_val, y_pred)

        list_score.append(sc_score)
        #  list_best_iterations.append(clf.current_iteration())
        logger.info(f'validation: {val_no} / {metric}: {sc_score}')

        ' validationの予測値をとる '
        #  val_pred = clf.predict(x_val[use_cols])
        val_pred = y_pred
        if i==0:
            val_stack = x_val.reset_index()[unique_id].to_frame()
            val_stack[target] = val_pred
        else:
            tmp = x_val.reset_index()[unique_id].to_frame()
            tmp[target] = val_pred
            val_stack = pd.concat([val_stack, tmp], axis=0)
        logger.info(f'valid_no: {val_no} | valid_stack shape: {val_stack.shape} | cnt_id: {len(val_stack[unique_id].drop_duplicates())}')

        if model_type != 'xgb':
            test_pred = clf.predict(test[use_cols])
        elif model_type == 'xgb':
            #  columns = []
            #  for col in test.columns:
            #      if col not in use_cols:continue
            #      columns.append(col.replace("[", "-").replace("]", "--").replace(",", "---"))
            #  tmp_test = test[use_cols].copy()
            #  tmp_test.columns = columns
            test_pred = clf.predict(xgb.DMatrix(test))

        if len(prediction)==0:
            #  prediction = test_pred
            prediction = test_pred / len(valid_list)
        else:
            #  prediction += test_pred
            prediction += test_pred / len(valid_list)

        ' Feature Importance '
        if model_type=='lgb':
            tmp_feim = pd.Series(clf.feature_importance(), name=f'{val_no}_importance')
            feature_name = pd.Series(use_cols, name='feature')
            feim = pd.concat([feature_name, tmp_feim], axis=1)

        elif model_type=='xgb':
            tmp_feim = clf.get_fscore()
            feim = pd.Series(tmp_feim,  name=f'{val_no}_importance', index=use_cols).to_frame().reset_index().rename(columns={'index':'feature'})

        if len(cv_feim)==0:
            cv_feim = feim
        else:
            cv_feim = cv_feim.merge(feim, on='feature', how='inner')

    cv_score = np.mean(list_score)
    logger.info(f'use_features: {use_cols}')
    logger.info(f'params: {params}')
    logger.info(f'\nCross Validation End\nCV score : {cv_score}')

    ' fold数で平均をとる '
    prediction = prediction / len(valid_list)

    pred_stack = test.reset_index()[unique_id].to_frame()
    pred_stack[target] = prediction
    result_stack = pd.concat([val_stack, pred_stack], axis=0)
    logger.info(f'result_stack shape: {result_stack.shape} | cnt_id: {len(result_stack[unique_id].drop_duplicates())}')

    importance = []
    for val_no in valid_list:
        if len(importance)==0:
            importance = cv_feim[f'{val_no}_importance'].values
        else:
            importance += cv_feim[f'{val_no}_importance'].values

    cv_feim['avg_importance'] = importance / len(valid_list)
    cv_feim.sort_values(by=f'avg_importance', ascending=False, inplace=True)
    cv_feim['rank'] = np.arange(len(cv_feim))+1

    cv_feim.to_csv(f'../output/cv_feature{len(cv_feim)}_importances_{metric}_{cv_score}.csv', index=False)

    return prediction, cv_score, len(use_cols), result_stack


def classifier_model(x_train, y_train, x_val=[], y_val=[], test=[], params={}, categorical_feature=[], num_iterations=3500, learning_rate=0.1, early_stopping_rounds=150, model_type='lgb'):

    ' LightGBM / XGBoost / ExtraTrees / LogisticRegression'
    if model_type=='lgb':
        lgb_train = lgb.Dataset(data=x_train, label=y_train)
        if len(x_val)>0:
            lgb_eval = lgb.Dataset(data=x_val, label=y_val)
            clf = lgb.train(params=params,
                            train_set=lgb_train,
                            valid_sets=lgb_eval,
                            num_boost_round=num_iterations,
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=200,
                            categorical_feature = categorical_feature
                            )
            y_pred = clf.predict(x_val)
        else:
            clf = lgb.train(params,
                            train_set=lgb_train,
                            categorical_feature = categorical_feature
                            )
            y_pred = clf.predict(test)

    elif model_type=='xgb':

        d_train = xgb.DMatrix(x_train, label=y_train)
        if len(x_val)>0:
            d_valid = xgb.DMatrix(x_val, label=y_val)
            watch_list = [(d_train, 'train'), (d_valid, 'eval')]

            clf = xgb.train(params,
                            dtrain=d_train,
                            evals=watch_list,
                            num_boost_round=num_iterations,
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=200
                            )
            y_pred = clf.predict(d_valid)
        else:
            d_test = xgb.DMatrix(test)
            clf = xgb.train(params,
                            dtrain=d_train,
                            num_boost_round=num_iterations
                            )
            y_pred = clf.predict(d_test)

    elif model_type=='ext':

        clf = ExtraTreesClassifier(
            **params,
            n_estimators = num_iterations
        )

        clf.fit(x_train, y_train)
        y_pred = clf.predict_proba(x_val)[:,1]

    elif model_type=='lgr':

        clf = LogisticRegression(
            **params,
            n_estimators = num_iterations
        )

        clf.fit(x_train, y_train)
        y_pred = clf.predict_proba(x_val)[:,1]

    else:
        logger.info(f'{model_type} is not supported.')

    return clf, y_pred
