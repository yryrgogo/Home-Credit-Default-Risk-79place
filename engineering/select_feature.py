code=0
code=1
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
        best_select = pd.read_csv('../output/cv_feature1350_importances_auc_0.809525405635011.csv')
        #  best_select = pd.read_csv('../output/cv_feature1330_importances_auc_0.8066523340763816.csv')
        #  best_select = pd.read_csv('../output/cv_feature1099_importances_auc_0.8072030486159842.csv')
        best_feature = best_select['feature'].values
        #  best_feature = best_select.query("flg==1")['feature'].values
        #  best_feature = best_select.query("rank<=50")['feature'].values
        #  best_feature = best_select.query("rank>=750")['feature'].values
        #  best_feature = best_select.query("rank>=100")['feature'].values
        #  best_feature = best_select.query("rank>=1000")['feature'].values
        #  best_feature = best_select.query("rank>=1300")['feature'].values
        best_feature = best_select.query("rank>=1200")['feature'].values
        #  best_feature = [col for col in best_feature if (col.count('a_') and col.count('AMT')) or col.count('p_Ap') or col.count('is_rm')]
        #  best_feature = [col for col in best_feature if col.count('impute')]
        #  best_feature = [col for col in best_feature if col.count('ker_')]
        best_feature = [col for col in best_feature if col.count('dima_') or col.count('gp_')]
        #  best_feature = [col for col in best_feature if col.count('new_len')]
        #  best_feature = [col for col in best_feature if col.count('dima_')]
        #  best_feature = [col for col in best_feature if col.count('gp_')]
        #  best_feature = [col for col in best_feature if col.count('new_len')]

        if len(best_feature)==0:
            sys.exit()
        for feature in best_feature:
            if feature not in ignore_features:
                try:
                    shutil.move(f'../features/3_winner/{feature}.npy', '../features/1_third_valid/')
                    #  shutil.move(f'../features/3_winner/{feature}.npy', '../features/1_second_valid/')
                    #  shutil.move(f'../features/feat_high_cv_overfit/{feature}.npy', '../features/1_third_valid/')
                except FileNotFoundError:
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
    best_select = pd.read_csv('../output/cv_feature1350_importances_auc_0.809525405635011.csv')
    #  best_select = pd.read_csv('../output/cv_feature1234_importances_auc_0.8091839448990605.csv')
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

    #  path_list = glob.glob('../features/1_second_valid/*.npy')
    path_list = glob.glob('../features/1_third_valid/*.npy')
    #  path_list = glob.glob('../features/win_tmp/*.npy')
    #  path_list = glob.glob('../features/dima/*.npy')

    for path in path_list:
        filename = re.search(r'/([^/.]*).npy', path).group(1)
        #  if filename in impute_list:
        #      print(f'continue: {filename}')
        #      continue
        #  if filename.count('NAME') or filename.count('TYPE') or filename.count('GENDER'):
        if filename in best_feature:
            #  shutil.move(path, '../features/1_third_valid/')
            shutil.move(path, '../features/3_winner/')
            #  shutil.move(path, '../features/CV08028/')
            #  shutil.move(path, '../features/feat_high_cv_overfit')
        #  for dima in dima_list:
        #      if filename.count(dima):
        #          shutil.move(path, '../features/dima_tmp/')



def main():
    if code==0:
        move_to_second_valid()
    elif code==1:
        move_to_use()
    elif code==2:
        move_file()


if __name__ == '__main__':

    dima_list = [
    'TARGET',
    'SK_ID_CURR',
    'EXT_SOURCE_1',
    'FLAG_DOCUMENT_3',
    'FLAG_DOCUMENT_6',
    'NAME_CONTRACT_TYPE',
    'FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_18',
    'FLAG_DOCUMENT_13',
    'FLAG_DOCUMENT_14',
    'CODE_GENDER',
    'FLAG_OWN_REALTY',
    'CNT_CHILDREN',
    'CNT_FAM_MEMBERS',
    'NAME_FAMILY_STATUS',
    'DAYS_BIRTH',
    'AMT_INCOME_TOTAL',
    'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE',
    'DAYS_EMPLOYED',
    'DAYS_REGISTRATION',
    'OWN_CAR_AGE',
    'OCCUPATION_TYPE',
    'ORGANIZATION_TYPE',
    'AMT_CREDIT',
    'AMT_ANNUITY',
    'AMT_GOODS_PRICE',
    'AMT_REQ_CREDIT_BUREAU_HOUR',
    'AMT_REQ_CREDIT_BUREAU_DAY',
    'AMT_REQ_CREDIT_BUREAU_WEEK',
    'AMT_REQ_CREDIT_BUREAU_MON',
    'AMT_REQ_CREDIT_BUREAU_QRT',
    'AMT_REQ_CREDIT_BUREAU_YEAR',
    'DEF_30_CNT_SOCIAL_CIRCLE',
    'DEF_60_CNT_SOCIAL_CIRCLE',
    'REGION_RATING_CLIENT_W_CITY',
    'FLAG_WORK_PHONE',
    'DAYS_LAST_PHONE_CHANGE',
    'FONDKAPREMONT_MODE',
    'HOUSETYPE_MODE',
    'TOTALAREA_MODE',
    'WALLSMATERIAL_MODE',
    'EMERGENCYSTATE_MODE',
    'null3',
    'nullsum',
    'mean',
    'es2',
    'es3',
    'FLAG_DOCUMENT_16_cash',
    'FLAG_DOCUMENT_18_cash',
    'FLAG_DOCUMENT_13_cash',
    'FLAG_DOCUMENT_14_cash',
    'FLAG_DOCUMENT_6_cash',
    'FLAG_DOCUMENT_16_cashFOC',
    'zzzzz',
    'singlewithkids',
    'rrrrr',
    'mmm',
    'bbbbb',
    'wwwww',
    'oooooooo',
    'sasasas',
    'sasauuuusas',
    'vvnn',
    'NEW_INC_BY_ORG',
    'AMT_CREDIT1',
    'AMT_ANNUITY1',
    'length',
    '12',
    '18',
    '24',
    '36',
    'rateCASH',
    'rateREVOLVING',
    'CONSUMER_GOODS_RATIO',
    'kek',
    'diff',
    'sum_avg',
    'sum_mode',
    'sum_medi',
    'PREV_cash_active_COUNT',
    'PREV_cash_not_active_COUNT',
    'PREV_consumer_active_COUNT',
    'PREV_consumer_2_COUNT',
    'PREV_consumer_3_COUNT',
    'PREV_consumer_4_COUNT',
    'PREV_revolving_active_COUNT',
    'PREV_revolving_2_COUNT',
    'PREV_revolving_3_COUNT',
    'PREV_cash_annui_MEDIAN',
    'PREV_cash_annui_MIN',
    'PREV_cash_annui_MAX',
    'PREV_consumer_annui_MEDIAN',
    'PREV_consumer_annui_MIN',
    'PREV_consumer_annui_MAX',
    'PREV_revolving_annui_MEDIAN',
    'PREV_left2pay_sum_SUM',
    'PREV_prev_length_MEDIAN',
    'PREV_prev_length_MIN',
    'PREV_prev_length_MAX',
    'PREV_AMT_CREDIT_1_MAX',
    'PREV_AMT_CREDIT_2_MAX',
    'PREV_AMT_CREDIT_3_MAX',
    'PREV_oblomis_MEAN',
    'PREV_CRR_MEAN',
    'PREV_DPD_MAX',
    'PREV_high_MIN',
    'PREV_middle_MIN',
    'PREV_low_action_MIN',
    'PREV_XNA_MIN',
    'PREV_lebngth_by_yield_MIN',
    'PREV_INST_NUM_INSTALMENT_VERSION_VAR_MAX',
    'PREV_inst_diff_MIN',
    'PREV_inst_diff_MAX',
    'PREV_inst_diff_1_MIN',
    'PREV_inst_diff_2_MIN',
    'PREV_inst_diff_3_MIN',
    'PREV_INST_NUM_INSTALMENT_NUMBER_MAX_MAX',
    'PREV_INST_DPD_MAX_MAX',
    'PREV_INST_DPD_SUM_SUM',
    'PREV_INST_DPD_MAX_1_MEAN',
    'PREV_INST_DPD_MAX_2_MEAN',
    'PREV_INST_DPD_MAX_3_MEAN',
    'PREV_CARD_USE_MEAN',
    'PREV_POS_CNT_INSTALMENT_VAR_MEAN',
    'PREV_POS_SK_DPD_3m_MEAN_MEAN',
    'PREV_CC_AMT_BALANCE_MEAN_MEAN',
    'PREV_CC_AMT_BALANCE_MAX_MAX',
    'PREV_CC_AMT_BALANCE_MAX_MEAN',
    'PREV_CC_AMT_BALANCE_1_MAX_SUM',
    'PREV_CC_USE_MEAN_MEAN',
    'PREV_CC_AMT_DRAWINGS_CURRENT_MEAN_MEAN',
    'PREV_CC_CNT_DRAWINGS_ATM_CURRENT_MEAN_MEAN',
    'BURO_closedA_COUNT',
    'BURO_activeB_COUNT',
    'BURO_type1_COUNT',
    'BURO_type3_COUNT',
    'BURO_type4_COUNT',
    'BURO_type5_COUNT',
    'BURO_type6_COUNT',
    'BURO_CREDIT_CURRENCY_NUNIQUE',
    'BURO_DDIF_MAX',
    'BURO_DDIF1_MAX',
    'BURO_longover_MAX',
    'BURO_longover1_MAX',
    'BURO_11_MAX',
    'BURO_22_MAX',
    'BURO_22_SUM',
    'BURO_33_MAX',
    'BURO_33_SUM',
    'BURO_44_MAX',
    'BURO_44_SUM',
    'BURO_DAYS_CREDIT_1_MAX',
    'BURO_DAYS_CREDIT_2_MAX',
    'BURO_DEBT_CREDIT_RATIO_1_MAX',
    'BURO_DEBT_CREDIT_RATIO_2_MAX',
    'BURO_ACS_1_MEAN',
    'BURO_ACS_2_MEAN',
    'L3',
    'history'
    ]
    main()
