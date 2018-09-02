import numpy as np
import pandas as pd
import gc
import os
import re
import sys
import datetime
from tqdm import tqdm
from lgbm_clf import validation
from sklearn.preprocessing import LabelEncoder

sys.path.append('../../../github/module/')
from preprocessing import set_validation, split_dataset

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

# PATH*************************************


# 学習結果のfeature importanceを軸に、featureと予測結果の関係をまとめる
def feature_summarize(ftim, dataset, corr_list, target, number, val_no):
    """
    予測結果を検証する為に、feature_importanceや特徴量同士の相関などを
    まとめる。

    Args:
        ftim    (df)    : feature_importanceをDFにし、cv_scoreなどを加えた
        dataset (df)    : 元のデータセット
        corr_list(list) : 学習に使用した特徴量リスト
        target(str)     : 目的変数
        number  (int)   : 全体の学習における通し番号
        val_no  (int)   : CVにおいて何番目のvalidationか

    Returns:result
    通し番号、CV番号、特徴量数、相関係数、特徴量リストファイル名を追加したDF
    """

    ftim['no'] = number
    ftim['val_no'] = val_no

    # feature_importanceを合計1としてスケーリング
    tmp = ftim['importance'].sum()
    ftim['importance'] = ftim['importance']/tmp
    ftim.sort_values(by='importance', ascending=False, inplace=True)

    # 基準とするfeatureを設定し、各featureの重要度の相対値をとる
    # （新しいfeatureと既存のgood featureを比較）
    #  comp = ftim[ftim['feature'] == 'SK_ID_CURR']['importance'].values[0]
    #  ftim['comp_air_id'] = ftim['importance']/comp

    corr = top_corr(dataset, corr_list, 5)
    corr_0 = top_corr(dataset[dataset[target] == 0], corr_list, 5, '_0')
    corr_1 = top_corr(dataset[dataset[target] == 1], corr_list, 5, '_1')

    result = ftim.merge(corr, on='feature', how='left')
    result = result.merge(corr_0, on='feature', how='left')
    result = result.merge(corr_1, on='feature', how='left')

    return result


def exploratory_train(logger, dataset, target, categorical_feature, max_val_no=1, number=0, pts_score='0', viz_flg=0, eda_flg=0, base_score={}):
    """
    時系列モデルの検証、学習、予測結果のEDAを行う。
    Partision、Validationの番号をデータセットにカラムとして持たせ、その番号
    毎に学習と予測を行い、今の特徴量セットの分析を進める。

    Args:
        dataset(df)              : 元のデータセット
        target                   : 目的変数
        categorical_feature(list): lightgbmにカテゴリ変数として認識させるリスト
        max_val_no(int)          : 複数のValidation_noがある場合、その最大値を入れる（番号は1からの連番が前提）
        number  (int)            : 全体の学習における通し番号。EDA用に出力するファイルを見やすくする為。

    Returns:
        result_val(DF) : 入力されたデータセットにおける指定Validation_noの予測結果をそれぞれ分析する為のデータフレーム。
    """

    """visualize用 （できたらメモリを圧迫しない様にしたい）"""
    key = np.arange(len(dataset))
    dataset['number'] = key
    if viz_flg == 1:
        visualize = dataset.copy()
        visualize.drop(['valid_no'], axis=1, inplace=True)
    else:
        visualize = []


    lbl = LabelEncoder()
    for cat in categorical_feature:
        dataset[cat] = lbl.fit_transform(dataset[cat].astype('str'))
    """ **************** """

    list_score = []
    list_diff = []
    # validation毎の結果をまとめるdf
    result_val = pd.DataFrame([])
    # validation毎に結果をまとめて結合していく
    for val_no in tqdm(range(1, max_val_no+1, 1)):

        # 学習用、テスト用データセットを作成
        train, test = split_dataset(dataset, val_no)
        # 学習
        ftim, use_cols = validation(logger, train, test, target,
                          categorical_feature, pts_score, visualize)

        # feature_importanceとまとめてスコアが重複してるので、一番目のみ取り出す
        score_cv = ftim['score_cv'].values[0]
        list_score.append(score_cv)

        if eda_flg == 1:

            if len(base_score) > 0:
                score_diff = score_cv - base_score[val_no-1]
            else:
                score_diff = 0
            list_diff.append(score_diff)
            ftim['score_diff'] = score_diff

            #  use_cols = list(train.columns)
            ftim['use_feature'] = f'{start_time[:11]}_{len(use_cols)}features.csv'

            # ここで相関を計算するfeatureのみリストに残す
            ftim['num_of_ft'] = len(use_cols)
            corr_list = use_cols.copy()
            for cat in categorical_feature:
                corr_list.remove(cat)

            "EDA用のデータセットを作成する"
            feature_summary = feature_summarize(
                ftim, dataset, corr_list, target, number, val_no)

            # validation毎にサマリーを格納し結合していく
            if len(result_val) == 0:
                result_val = feature_summary
            else:
                result_val = pd.concat([result_val, feature_summary], axis=0)

        del train, test
        gc.collect()

    if eda_flg==1:
        # 全Validationの平均スコアを格納する
        result_val['score_mean'] = np.mean(list_score)
        result_val['diff_mean'] = np.mean(list_diff)
        # 学習に使用した特徴量リストを確認する為のファイルを出力する
        if os.path.isdir(f'../output/{start_time}'):
            pd.Series(use_cols, name='features').to_csv(f'../output/{start_time}/{start_time[:11]}_No{number}_use_{len(use_cols)}features.csv', index=False)
        else:
            os.mkdir(f'../output/{start_time}')
            pd.Series(use_cols, name='features').to_csv(f'../output/{start_time}/{start_time[:11]}_No{number}_use_{len(use_cols)}features.csv', index=False)

        return result_val
    else:
        ftim['score_mean'] = np.mean(list_score)
        ftim['no'] = number
        return ftim


def top_corr(data, corr_list, num, suffix=''):
    """
    Explain:
        対象featureと相関の高い上位featureとその係数を取得する
        また、各パーティションにより相関が変わるかなども確認する
        (パーティション毎の相関などを求めたい際は、class毎にDFを切った上でわたす)
        変数名に対する横持ちでデータフレームが返ってくるので、featureカラムを
        keyとして結合する

    Args:
        data (DF)       : 相関を計算したいカラムをまとめたデータフレーム
        corr_list (list): 相関係数を取り出したい変数名のリスト
        num             : 上位何番目まで相関係数とその変数を取り出すか
        suffix          : 複数の相関を一つのDFにまとめたい時、区別用の文字を末尾につける

    Returns:
        result_corr(DF):
        上位相関係数とその変数をTableau可視化用フォーマットにしたDF。
        変数名に対して横持ちでvalueが入っているので、
        mergeする時はカラム名:featureをkeyとする

    """

    # 相関を計算
    df = data[corr_list].corr(method='pearson')
    df = df[df < 1]  # 相関が1となる行は表示しない

    # 結果を格納するデータフレーム
    result_corr = pd.DataFrame([])

    for ft in corr_list:

        # 相関係数上位を取り出しDFに変換、インデックスを振りなおす
        # reset_indexにより、変数名がカラム名:indexとしてDFに入る
        top = df[ft].sort_values(ascending=False)[:num].to_frame().reset_index()

        # インデックスにするリストを作る:要素はカラム名。相関の順位とカテゴリか係数かをわかるように
        idx_cat = []
        idx_cor = []
        for i in range(1, num+1, 1):
            idx_cat.append('corr_{}_cat'.format(i)+suffix)
            idx_cor.append('corr_{}'.format(i)+suffix)
        idx = idx_cat + idx_cor

        # 相関上位変数名とその相関係数を一つのリストにまとめる
        val = list(top['index']) + list(top[ft])
        top_cor = pd.Series(val, index=idx)
        top_cor['feature'] = ft  # どの変数について相関を求めたか
        top_cor = top_cor.to_frame().T  # 横持ちにする

        # 各変数について、横持ちを積み上げていく
        if len(result_corr) == 0:
            result_corr = top_cor
        else:
            result_corr = pd.concat([result_corr, top_cor], axis=0)

    return result_corr


def incremental_train(data, target, categorical_feature, valid_list, max_val_no):
    """ Summary line
    用意した特徴量の最適な組み合わせを発見する為、インクリメンタルな学習を行う。
    戻り値はなく、全学習＆予測結果のまとめをCSVで出力して終わり。

    Args:
        data(DF)        : 候補となる全特徴量を含んだデータフレーム。
        valid_list(list): 学習を試したい特徴量の組み合わせリスト。この特徴量
                           セットを順に流し、学習結果とその分析結果をまとめる。
    """

    number = 0
    result = pd.DataFrame([])

    for feature_path in tqdm(valid_list):
        dataset= data.copy()
        feature_set = pd.DataFrame([])
        for path in feature_path:
            feature = pd.read_csv(path)

            level = [col for col in feature.columns if not(col.count('@'))][0]

            dataset = dataset.merge(feature, on=level, how='left')
        print(dataset.shape)

        number += 1
        tmp_result = exploratory_train(
            dataset = valid_set,
            target = target,
            categorical_feature =  categorical_feature,
            max_val_no = 1,
            number = 0,
            pts_score = partision,
            viz_flg = 1,
            eda_flg = 0,
            base_score = {}
        )

        if len(result) == 0:
            result = tmp_result
        else:
            result = pd.concat([result, tmp_result], axis=0)

    result.to_csv('../output/{}_ftim.csv'.format(start_time[:11]), index=False)


