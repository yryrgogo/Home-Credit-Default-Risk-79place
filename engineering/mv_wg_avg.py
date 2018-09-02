import numpy as np
import pandas as pd
import sys
sys.path.append('../../../github/module/')
from date_function import date_diff


def moving_agg(method, data, level, index, value, window, periods, sort_col=None):

    '''
    Explain:
        移動平均を求める。リーケージに注意
    Args:
        method(sum|avg):
        data(DF)        : 入力データ
        level(list)     : 集計を行う粒度。最終的に欠損値補完を行う粒度が1カラム
                          でなく、複数カラム必要な時はリストで渡す。
                          0からindexの数分を集計粒度とする(順番注意)
        index(int)      : 集計する粒度のカラム数
        value(float)    : 集計する値
        window(int)     : 移動平均して集計する行の数(上の行に遡る数)
        periods(int)    : 集計時に最小いくつの値を必要とするか
        sort_col(column): 移動平均時にソートが必要な場合のカラム。
                          これを入力した場合、戻り値Seriesのindexになる
    Returns:
        result(DF)  : 移動平均による集計結果。groupby as_index=False
    '''

    ' 集計時に日付は邪魔なのでindexに入れてからソートする '
    if not(sort_col is None):
        data = data.set_index(sort_col)
        data = data.sort_index()

    level = level[:index]

    if method=='avg':
        result = data.groupby(level)[value].rolling(
            window=window, min_periods=periods).mean().reset_index()
    elif method=='sum':
        result = data.groupby(level)[value].rolling(
            window=window, min_periods=periods).sum().reset_index()

    result.rename(columns={value: f'{value}_mv_{method}_w{window}_p{periods}@{level}'}, inplace=True)

    return result


def exp_weight_avg(data, level, value, weight, label):

    '''
    Explain:
        重み付き平均。
    Args:
        data(DF)    : 入力データ
        level       : 集計する粒度
        value(float): 集計する値
        weight(int) : 重み付き平均の減衰率
        label       : 重みを計算するラベル。通常は日付など
    Return:
        result(Series): 重み付き平均の集計結果。返す行は各粒度の最下行になる
    '''

    N = len(data)
    max_label = data[label].max()

    ' 指数重み付き平均なので、各行の重みを何乗にするか '
    ' labelが日付の場合はdate_diff。そうでない場合はそのとき作る '
    if str(type(data[label])).count('date'):
        data['diff'] = abs(date_diff(max_label, data[label]))
        data['weight'] = data['diff'].map(lambda x: weight ** x.days)
    else:
        data['diff'] = abs(data[label] - max_label)
        data['weight'] = data['diff'].map(lambda x: weight ** x)

    ' 各行について、値へ重みをかける '
    data['tmp'] = data['weight'] * data[value]

    '''
    valueがNullになっている行の重みは0にする。（分母の合計に入ってしまう為）
    重みをかけた行の値はNullになっているが、重みの値はNullになっていない
    '''
    no_null = data[data[value].notnull()]
    null = data[data[value].isnull()]
    if len(null)>0:
        null['weight'] = 0
    if len(null)>0 and len(no_null)>0:
        data = pd.concat([null, no_null], axis=0)
    elif len(null)==0 and len(no_null)>0:
        data = no_null
    elif len(null)>0 and len(no_null)==0:
        data = null

    ' 分子、分母それぞれ合計をとって割る '
    tmp_result = data.groupby(level)['tmp', 'weight'].sum()
    result = tmp_result['tmp']/tmp_result['weight']

    ' feature name: 元のカラム名＋減衰率＋粒度 '
    result.name = f'{value}_wg{weight}_avg@{level}'

    return result


def let_wg_agg(data, level, number, value, method_list, weight_list, index, lag, segment=None, segment_list=['nothing']):
    """
    Explain:
        移動平均の前処理が必要な場合の処理をまとめた関数。

    Args:
        data(DF)          :
        level(list)       : 移動平均結果のvalueカラムとは別に残すカラムリスト.
                            頭(0)から移動平均の集計粒度となり、次のindex変数に
                            集計粒度カラム数が格納される。
                            例：idの粒度で集計し、id * dateの粒度で結果DFを作る
                                場合は、リストに[id, date]、index=1となる
                            ※リストの要素順は注意
        index(int)        : 集計に使うlevel内の要素の数。頭(0)から使う
        value(float)      : 集計する値
        method(sum|avg)   :
        weight_list(list) : 重み付き平均をの減衰率
        index             : 時系列などで重みの調整をする場合、そのインデックスとなるカラム名
        lag(int)          : リーク防止などで集計する値をshiftによってずらす行の数
                           （正：前lag行のデータ、 負：後lag行のデータ）
        segment(column)   : セグメントの入ったカラム名
        segment_list(list): セグメントに分けて集計を実施する場合、その
                            セグメントに切り分ける要素を格納したリスト
                            セグメントがない場合は引数なしで'nothing'が入る
    Return:
        なし。csvをoutput

    """

    ' 連休、非連休の祝日データセット作成 '
    consective, discrete = consective_holiday('air_cal_next')
    consective.sort_values(by=level, inplace=True)
    discrete.sort_values(by=level, inplace=True)

    """ 重み付き平均 """
    for method in method_list:
        for weight in weight_list:
            seg_result = pd.DataFrame([])
            ' recruit competisionの場合は、曜日毎の特徴量を作りたいので、segment=dow '
            for seg in segment_list:
                if seg != 'nothing':
                    ' 対象セグメントに絞る '
                    df_seg = data[data[segment] == seg]

                    '''
                    祝日はその曜日の重み付き平均に含めないが、週カウントは行う。
                    その為、祝日は値をNullにして行のみ残す
                    NULLにしたい値をもつカラムをdropしてtmpDFに格納（その行は
                    集計しないだけでカウントしたいので、インデックスは残す為）
                    '''
                    ' Nullにしたいレコードを落とす '
                    tmp = df_seg.drop(value, axis=1)
                    key_columns = list(tmp.columns.values)
                    df_seg = df_seg[df_seg['day_of_week'] != 'Special']
                    '''
                    全てのインデックスを持ったDFに、NULLにしたいレコードを落とした
                    DFをLeft Joinすることで、対象レコードのvalueのみNullにする
                    '''
                    dataset = tmp.merge(
                        df_seg, on=key_columns, how='left', copy=False)

                else:
                    dataset = data

                ' 目的変数を特徴量化する為、リークしない時系列でもたせる '
                dataset = lag_feature(
                    data=dataset, value=value, lag=lag, level=level)

                ''''''''''''''''''''''''''''''''''''''
                '''''  コア部分：重み付き平均    '''''
                ''''''''''''''''''''''''''''''''''''''

                if seg == 'nothing':
                    seg_wg = exp_weight_avg(
                        dataset, level[0], value, weight, index)
                    col_name = [
                        col for col in seg_wg.columns if col.count('@')][0]
                    tmp_result.to_csv(
                        f'../feature/valid_feature/{col_name}.csv', header=True, index=False)
                    return
                    ' セグメント別に集計しない場合、ここで完了 '

                index_list= dataset[index].drop_duplicates().values

                idx_stack = pd.DataFrame([])
                for idx in index_list:
                    seg_wg = exp_weight_avg(dataset, level[0], value, weight, index)
                    seg_wg = seg_wg.reset_index()
                    seg_wg[index] = idx

                    if len(idx_stack)==0:
                        idx_stack = seg_wg
                    else:
                        idx_stack = pd.concat([idx_stack, seg_wg], axis=0)
                ' 特徴量は@を持ったカラムとしているので、これで取得できる '
                col_name = [col for col in idx_stack.columns if col.count('@')][0]

                '''
                集計後はNULLでカウントさせてたセグメントのレコードは必要ないので
                inner joinで除外する
                '''
                tmp_result = idx_stack.merge(df_seg[level], on=level, how='inner')

                ' 欠損値を平均で補完 '
                tmp_result = impute_avg(tmp_result, level, 1, col_name)

                if len(seg_result) == 0:
                    seg_result = tmp_result
                else:
                    seg_result = pd.concat([seg_result, tmp_result], axis=0)

            print(seg_result.shape)
            '''
            ********************************
            **連休セグをもっと綺麗に分け、**
            **上記のループに含めることはで**
            **きないか？
            ********************************
            '''
            consective = lag_feature( data=consective, value=value, lag=lag, level=level)
            discrete = lag_feature(data=discrete, value=value, lag=lag, level=level)

            index_list= consective[index].drop_duplicates().values
            idx_stack = pd.DataFrame([])

            for idx in index_list:
                seg_wg = exp_weight_avg(consective, level[0], value, weight, index)
                seg_wg = seg_wg.reset_index()
                seg_wg[index] = idx

                if len(idx_stack)==0:
                    idx_stack = seg_wg
                else:
                    idx_stack = pd.concat([idx_stack, seg_wg], axis=0)

            ''' 連休の特徴量 '''
            consective_mv = moving_agg(
                method, consective, level, 1, value, window, 1, 'visit_date')
            ' feature nameを取得 '
            col_name = [
                col for col in consective_mv.columns if col.count('@')][0]
            ' NULL埋め '
            result_cont = impute_avg(consective_mv, level, 1, col_name)

            ''' 非連休の特徴量 '''
            discrete_mv = moving_agg(
                method, discrete, level, 1, value, window, 1, 'visit_date')
            ' feature nameを取得 '
            col_name = [
                col for col in discrete_mv.columns if col.count('@')][0]
            ' NULL埋め '
            result_disc = impute_avg(discrete_mv, level, 1, col_name)

            ''' 結合 '''
            result = pd.concat([seg_result, result_cont, result_disc], axis=0)

            ' 日程を元のデータセットと同様にする '
            result = air_cal[level].merge(result, on=level, how='inner')

            result.sort_values(by=level, inplace=True)
            #  result = result[col_name]
            result.to_csv(f'../feature/{col_name}.csv',
                          index=False, header=True)
            print(result.shape)
            print(result.head())
