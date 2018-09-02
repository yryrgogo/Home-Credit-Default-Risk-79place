import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn import preprocessing
import sys
gc.enable()

unique_id = 'SK_ID_CURR'

print('Read data and test')
data = pd.read_csv('../data/application_train.csv')
test = pd.read_csv('../data/application_test.csv')


print('Shapes : ', data.shape, test.shape)
len_data = len(data)
app = pd.concat([data, test])

app = app[['TARGET', 'SK_ID_CURR', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',

           'FLAG_DOCUMENT_3',
           'FLAG_DOCUMENT_6',

           'NAME_CONTRACT_TYPE',

           'FLAG_DOCUMENT_16',
           'FLAG_DOCUMENT_18',
           'FLAG_DOCUMENT_13',
           'FLAG_DOCUMENT_14',

           'CODE_GENDER',

           'FLAG_OWN_CAR',
           'FLAG_OWN_REALTY',

           'CNT_CHILDREN',
           'CNT_FAM_MEMBERS',
           'NAME_FAMILY_STATUS',
           'DAYS_BIRTH',
           'AMT_INCOME_TOTAL',
           'NAME_INCOME_TYPE',
           'NAME_EDUCATION_TYPE',
           'DAYS_EMPLOYED',
           'DAYS_ID_PUBLISH',
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

           'REGION_RATING_CLIENT',
           'REGION_RATING_CLIENT_W_CITY',


           'FLAG_WORK_PHONE',
           'DAYS_LAST_PHONE_CHANGE',
           ]]
app['DAYS_LAST_PHONE_CHANGE'].value_counts()
app['DAYS_LAST_PHONE_CHANGE'].describe()
app['DAYS_REGISTRATION'].isnull().sum()
#  sns.distplot(app['DAYS_REGISTRATION'].fillna(0))


le = preprocessing.LabelEncoder()
app['null1'] = app['EXT_SOURCE_1'].isnull()
app['null1'] = le.fit_transform(app['null1'].astype(int))
app['null2'] = app['EXT_SOURCE_2'].isnull()
app['null2'] = le.fit_transform(app['null2'].astype(int))
app['null3'] = app['EXT_SOURCE_3'].isnull()
app['null3'] = le.fit_transform(app['null3'].astype(int))
app['nullsum'] = app['null1']+app['null2']
del app['null1'], app['null2']  # ,app['null3']


# EXT_SOURCE features
app['EXT_SOURCE_1'].fillna(app['EXT_SOURCE_2'].median(), inplace=True)
app['mean'] = app[['EXT_SOURCE_1',
                   'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
# app['es1']=(app['EXT_SOURCE_1']-0.377668)
#del app['EXT_SOURCE_1']
app['es2'] = (app['EXT_SOURCE_2']-app['EXT_SOURCE_2'].median())
del app['EXT_SOURCE_2']
app['es3'] = (app['EXT_SOURCE_3']-app['EXT_SOURCE_3'].mean())
del app['EXT_SOURCE_3']


app['FLAG_DOCUMENT_16_cash'] = app['FLAG_DOCUMENT_16'].where(
    app['NAME_CONTRACT_TYPE'] == 'Cash loans', np.nan)
app['FLAG_DOCUMENT_18_cash'] = app['FLAG_DOCUMENT_18'].where(
    app['NAME_CONTRACT_TYPE'] == 'Cash loans', np.nan)
app['FLAG_DOCUMENT_13_cash'] = app['FLAG_DOCUMENT_13'].where(
    app['NAME_CONTRACT_TYPE'] == 'Cash loans', np.nan)
app['FLAG_DOCUMENT_14_cash'] = app['FLAG_DOCUMENT_14'].where(
    app['NAME_CONTRACT_TYPE'] == 'Cash loans', np.nan)
app['FLAG_DOCUMENT_6_cash'] = app['FLAG_DOCUMENT_6'].where(
    app['NAME_CONTRACT_TYPE'] == 'Cash loans', np.nan)
app['FLAG_DOCUMENT_16_cashFOC'] = app['FLAG_DOCUMENT_3'].where(
    app['FLAG_OWN_CAR'] == 'N', np.nan)
app['zzzzz'] = app['es3'].where(app['FLAG_OWN_REALTY'] == 'Y', np.nan)
app['CNT_FAM_MEMBERS'] = app['CNT_FAM_MEMBERS']-app['CNT_CHILDREN']
app['singlewithkids'] = app['CNT_CHILDREN'].where(
    app['NAME_FAMILY_STATUS'] != 'Married', np.nan)
app['rrrrr'] = app['AMT_INCOME_TOTAL'].where(
    app['NAME_CONTRACT_TYPE'] != 'Cash loans', np.nan)
app['mmm'] = app['DAYS_BIRTH'].where(
    app['NAME_INCOME_TYPE'] != 'Pensioner', np.nan)

app['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
app['zzzxc'] = app['CODE_GENDER'].replace('F', 58.5)
app['zzzxc'] = app['zzzxc'].replace('M', 63)
app['zzzxc'] = app['zzzxc'].replace('XNA', 63)
app['zzzxc'].value_counts()
app['bbbbb'] = (app['DAYS_BIRTH']/-365)-app['zzzxc']
del app['zzzxc']

app['wwwww'] = app['bbbbb']/app['CNT_CHILDREN']
app['oooooooo'] = (app['DAYS_BIRTH']-app['DAYS_EMPLOYED']) / \
    app['AMT_INCOME_TOTAL']

app['sasasas'] = (app['DAYS_BIRTH']/-365) - (app['DAYS_ID_PUBLISH']/-365)
app['sasauuuusas'] = app['sasasas'].apply(
    lambda x: -200 if (45.0 <= x <= 45.4) else x)
app['sasauuuusas'] = app['DAYS_ID_PUBLISH'].where(
    app['sasauuuusas'] >= 0, np.nan)
#del app['DAYS_ID_PUBLISH']
#  app['vvnn'] = (app['DAYS_REGISTRATION'])/(app['DAYS_BIRTH'])
app['OCCUPATION_TYPE'].fillna('Core staff', inplace=True)
inc_by_org = app[['AMT_INCOME_TOTAL', 'OCCUPATION_TYPE']].groupby(
    'OCCUPATION_TYPE').median()['AMT_INCOME_TOTAL']
app['NEW_INC_BY_ORG'] = app['OCCUPATION_TYPE'].map(inc_by_org)
app['NEW_INC_BY_ORG'] = app['NEW_INC_BY_ORG']/app['AMT_INCOME_TOTAL']

app['AMT_CREDIT1'] = app['AMT_CREDIT']
app['AMT_ANNUITY1'] = app['AMT_ANNUITY']
app['AMT_CREDIT'] = app['AMT_CREDIT'].where( app['NAME_CONTRACT_TYPE'] == 'Cash loans', np.nan)
app['length'] = app['AMT_CREDIT']/app['AMT_ANNUITY']
app['AMT_ANNUITY'] = app['AMT_ANNUITY'].where( app['NAME_CONTRACT_TYPE'] == 'Cash loans', np.nan)
app['length'] = app['length'].where(
    app['NAME_CONTRACT_TYPE'] == 'Cash loans', np.nan)


app['8'] = app['AMT_ANNUITY']*8-app['AMT_CREDIT']
app['8'] = app['8']/app['AMT_CREDIT']
app['8'] = app['8'].apply(lambda x: x if 0 <= x <= 0.5 else np.nan)
app['9'] = app['AMT_ANNUITY']*9-app['AMT_CREDIT']
app['9'] = app['9']/app['AMT_CREDIT']
app['9'] = app['9'].apply(lambda x: x if 0 <= x <= 0.5 else np.nan)
app['30'] = app['AMT_ANNUITY']*30-app['AMT_CREDIT']
app['30'] = app['30']/app['AMT_CREDIT']
app['30'] = app['30'].apply(lambda x: x if 0 <= x <= 0.5 else np.nan)
app['32'] = app['AMT_ANNUITY']*32-app['AMT_CREDIT']
app['32'] = app['32']/app['AMT_CREDIT']
app['32'] = app['32'].apply(lambda x: x if 0 <= x <= 0.5 else np.nan)

sys.path.append('../../../github/module/')
from preprocessing import set_validation, factorize_categoricals
from make_file import make_npy, make_feature_set, make_raw_feature
base = pd.read_csv('../data/base.csv')
prefix = 'dist_'
train = app[[unique_id, '8', '9', '30', '32']]
result = base[unique_id].to_frame().merge(train, on=unique_id, how='left')
make_raw_feature(result, prefix, select_list=[],
                 #  path='../features/gp/')
                 #  path='../features/buro_dima/')
                 path='../features/1_first_valid/')
                 #  path='../features/raw_features/')
                 #  path='../features/go/')
sys.exit()

app['12'] = app['AMT_ANNUITY']*12-app['AMT_CREDIT']
app['12'] = app['12']/app['AMT_CREDIT']
app['12'] = app['12'].apply(lambda x: x if 0 <= x <= 0.5 else np.nan)

app['18'] = app['AMT_ANNUITY']*18-app['AMT_CREDIT']
app['18'] = app['18']/app['AMT_CREDIT']
app['18'] = app['18'].apply(lambda x: x if 0 <= x <= 0.5 else np.nan)

app['24'] = app['AMT_ANNUITY']*24-app['AMT_CREDIT']
app['24'] = app['24']/app['AMT_CREDIT']
app['24'] = app['24'].apply(lambda x: x if 0 <= x <= 0.5 else np.nan)

app['36'] = app['AMT_ANNUITY']*36-app['AMT_CREDIT']
app['36'] = app['36']/app['AMT_CREDIT']
app['36'] = app['36'].apply(lambda x: x if 0 <= x <= 0.5 else np.nan)

app['48']=app['AMT_ANNUITY']*48-app['AMT_CREDIT']
app['48']=app['48']/app['AMT_CREDIT']
app['48']=app['48'].apply(lambda x: x if  0<=x<=0.5 else np.nan)

app['60']=app['AMT_ANNUITY']*60-app['AMT_CREDIT']
app['60']=app['60']/app['AMT_CREDIT']
app['60']=app['60'].apply(lambda x: x if 0<=x<=0.5 else np.nan)
app['60'].describe()


app['x'] = app['AMT_CREDIT1']/app['AMT_ANNUITY1']
app['y'] = app['x']
app['x'].value_counts()
app['y'].fillna(0, inplace=True)
app['y'] = app['y'].astype(int)
app['annuiwoperc'] = app['AMT_CREDIT1']/app['y']
app['monthadditional'] = app['AMT_ANNUITY1']-app['annuiwoperc']
app['rate'] = app['monthadditional']*100/app['annuiwoperc']
del app['x'], app['y'], app['annuiwoperc'], app['monthadditional']


app['rateCASH'] = app['rate']
app['rateCASH'].describe()
app['rateCASH'] = app['rate'].where(
    app['NAME_CONTRACT_TYPE'] == 'Cash loans', np.nan)
app['rateREVOLVING'] = app['rate'].where(
    app['NAME_CONTRACT_TYPE'] == 'Revolving loans', np.nan)
del app['rate']  # ,app['AMT_CREDIT1'],app['AMT_ANNUITY1']


app['CONSUMER_GOODS_RATIO'] = app['AMT_CREDIT'] / app['AMT_GOODS_PRICE']
app['kek'] = app['CONSUMER_GOODS_RATIO']-1
del app['CONSUMER_GOODS_RATIO']

app['AMT_REQ_CREDIT_BUREAU_HOUR'].fillna(0, inplace=True)
app['AMT_REQ_CREDIT_BUREAU_DAY'].fillna(0, inplace=True)
app['AMT_REQ_CREDIT_BUREAU_WEEK'].fillna(0, inplace=True)
app['AMT_REQ_CREDIT_BUREAU_MON'].fillna(0, inplace=True)
app['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(0, inplace=True)
app['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(0,inplace=True)

app['diff'] = app['REGION_RATING_CLIENT']+app['REGION_RATING_CLIENT_W_CITY']
del app['REGION_RATING_CLIENT']

app['FLAG_WORK_PHONE'] = app['FLAG_WORK_PHONE'].where(
    app['NAME_INCOME_TYPE'] != 'Pensioner', np.nan)

#  app['DAYS_LAST_PHONE_CHANGE'].replace(0, -4500, inplace=True)

app['ORGANIZATION_TYPE'].replace('Trade: type 6', np.nan, inplace=True)
app['ORGANIZATION_TYPE'].replace('Industry: type 5', np.nan, inplace=True)
app['ORGANIZATION_TYPE'].replace('Insurance', np.nan, inplace=True)
app['ORGANIZATION_TYPE'].replace('Telecom', np.nan, inplace=True)
app['ORGANIZATION_TYPE'].replace('Emergency', np.nan, inplace=True)
app['ORGANIZATION_TYPE'].replace('Industry: type 2', np.nan, inplace=True)
app['ORGANIZATION_TYPE'].replace('Advertising', np.nan, inplace=True)
app['ORGANIZATION_TYPE'].replace('Realtor', np.nan, inplace=True)
app['ORGANIZATION_TYPE'].replace('Industry: type 12', np.nan, inplace=True)
app['ORGANIZATION_TYPE'].replace('Culture', np.nan, inplace=True)
app['ORGANIZATION_TYPE'].replace('Trade: type 1', np.nan, inplace=True)
app['ORGANIZATION_TYPE'].replace('Mobile', np.nan, inplace=True)
app['ORGANIZATION_TYPE'].replace('Legal Services', np.nan, inplace=True)
app['ORGANIZATION_TYPE'].replace('Cleaning', np.nan, inplace=True)
app['ORGANIZATION_TYPE'].replace('Transport: type 1', np.nan, inplace=True)
app['ORGANIZATION_TYPE'].replace('Industry: type 6', np.nan, inplace=True)
app['ORGANIZATION_TYPE'].replace('Industry: type 10', np.nan, inplace=True)
app['ORGANIZATION_TYPE'].replace('Religion', np.nan, inplace=True)
app['ORGANIZATION_TYPE'].replace('Trade: type 4', np.nan, inplace=True)
app['ORGANIZATION_TYPE'].replace('Industry: type 13', np.nan, inplace=True)
app['ORGANIZATION_TYPE'].replace('Trade: type 5', np.nan, inplace=True)
app['ORGANIZATION_TYPE'].replace('Industry: type 8', np.nan, inplace=True)
del app['FLAG_OWN_CAR']

app.to_csv('../data/dima_app_feature.csv', index=False)

################################################################################################################### read prev ########################################################################
print('Read prev')
prev = pd.read_csv(
    '../data/previous_application.csv')
prev = prev[['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_FIRST_DUE', 'DAYS_TERMINATION', 'NAME_CONTRACT_STATUS',
             'NAME_CONTRACT_TYPE',
             'AMT_ANNUITY',
             'AMT_CREDIT',
             'AMT_APPLICATION',
             'AMT_GOODS_PRICE',
             'AMT_DOWN_PAYMENT',
             'RATE_DOWN_PAYMENT',
             'RATE_INTEREST_PRIMARY',
             'RATE_INTEREST_PRIVILEGED',
             'NAME_CASH_LOAN_PURPOSE',
             'CODE_REJECT_REASON',
             'DAYS_DECISION',
             'DAYS_FIRST_DRAWING',
             'CNT_PAYMENT',
             'NAME_YIELD_GROUP',
             'NFLAG_INSURED_ON_APPROVAL',
             'NAME_PAYMENT_TYPE',
             'NAME_CLIENT_TYPE',
             'NAME_TYPE_SUITE',
             'NAME_PORTFOLIO',
             'NAME_GOODS_CATEGORY'
             ]]
prev['NAME_CLIENT_TYPE'].describe()
prev['NAME_GOODS_CATEGORY'].value_counts()
prev['NAME_GOODS_CATEGORY'].isnull().sum()
prev[['AMT_ANNUITY', 'NAME_CASH_LOAN_PURPOSE']]
######################################################################################################################################################################
# prev=prev[prev['NAME_PORTFOLIO']!='XNA']
prev = prev[prev['AMT_ANNUITY'] > 0]
prev['DAYS_FIRST_DUE'].replace(365243, -30, inplace=True)

prev['RATE_INTEREST_PRIMARY'].describe()
prev['DAYS_TERMINATION'].replace(365243.000000, np.nan).describe()

prev['RATE_INTEREST_PRIVILEGED'].value_counts()
prev['RATE_INTEREST_PRIVILEGED'].isnull().sum()
prev[['AMT_ANNUITY', 'RATE_INTEREST_PRIMARY']]


prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
prevsa = prev[prev['DAYS_LAST_DUE_1ST_VERSION'] > 0]
prevsa = prevsa[prevsa['DAYS_LAST_DUE'] > 0]
prevsa = prevsa[['SK_ID_PREV', 'DAYS_LAST_DUE_1ST_VERSION']]
prevsa.rename(
    columns={'DAYS_LAST_DUE_1ST_VERSION': 'stillACTIVE'}, inplace=True)
# prevsa=prevsa.groupby('SK_ID_CURR',as_index=True).sum()
prev = prev.merge(right=prevsa, how='left', on='SK_ID_PREV')
del prevsa

prev['count'] = prev['SK_ID_PREV']

prev['cash'] = prev['SK_ID_PREV'].where(
    prev['NAME_CONTRACT_TYPE'] == 'Cash loans', np.nan)
prev['consumer'] = prev['SK_ID_PREV'].where(
    prev['NAME_CONTRACT_TYPE'] == 'Consumer loans', np.nan)
prev['revolving'] = prev['SK_ID_PREV'].where(
    prev['NAME_CONTRACT_TYPE'] == 'Revolving loans', np.nan)

prev['cash_active'] = prev['cash'].where(
    prev['stillACTIVE'].isnull() == False, np.nan)
prev['cash_not_active'] = prev['cash'].where(
    prev['stillACTIVE'].isnull() == True, np.nan)

prev['consumer_active'] = prev['consumer'].where(
    prev['DAYS_FIRST_DUE'] >= -40, np.nan)
prev['consumer_temp'] = prev['consumer'].where(
    prev['DAYS_FIRST_DUE'] < -40, np.nan)
prev['consumer_2'] = prev['consumer_temp'].where(
    prev['DAYS_FIRST_DUE'] >= -540, np.nan)
prev['consumer_3'] = prev['consumer_temp'].where(
    prev['DAYS_FIRST_DUE'] < -540, np.nan)
prev['consumer_4'] = prev['SK_ID_PREV'].where(
    prev['DAYS_FIRST_DUE'].isnull() == True, np.nan)

prev['revolving_active'] = prev['revolving'].where(
    prev['DAYS_FIRST_DUE'] >= -40, np.nan)
prev['revolving_temp'] = prev['revolving'].where(
    prev['DAYS_FIRST_DUE'] < -40, np.nan)
prev['revolving_2'] = prev['revolving_temp'].where(
    prev['DAYS_FIRST_DUE'] >= -900, np.nan)
prev['revolving_3'] = prev['revolving_temp'].where(
    prev['DAYS_FIRST_DUE'] < -900, np.nan)

prev['AMT_ANNUITY'].replace(0, np.nan, inplace=True)

prev['cash_annui'] = prev['AMT_ANNUITY'].where(
    prev['NAME_CONTRACT_TYPE'] == 'Cash loans', np.nan)
prev['consumer_annui'] = prev['AMT_ANNUITY'].where(
    prev['NAME_CONTRACT_TYPE'] == 'Consumer loans', np.nan)
prev['revolving_annui'] = prev['AMT_ANNUITY'].where(
    prev['NAME_CONTRACT_TYPE'] == 'Revolving loans', np.nan)
prev['revolving_annui'].describe()

prev_act = prev
prev_act['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
prev_act = prev_act[prev_act['DAYS_LAST_DUE_1ST_VERSION'] > 0]
prev_act = prev_act[prev_act['DAYS_LAST_DUE'] > 0]
prev_act['left2pay_sum'] = prev_act['AMT_ANNUITY'] * \
    prev_act['DAYS_LAST_DUE_1ST_VERSION']/30
prev = prev.merge(
    right=prev_act[['SK_ID_PREV', 'left2pay_sum']], how='left', on='SK_ID_PREV')

prev['AMT_CREDIT'].replace(0, np.nan, inplace=True)

prev['prev_length'] = prev['AMT_CREDIT']/prev['AMT_ANNUITY']

prev['AMT_CREDIT_1'] = prev['AMT_CREDIT'].where(
    prev['DAYS_FIRST_DUE'] >= -40, np.nan)
prev['AMT_CREDIT_temp'] = prev['AMT_CREDIT'].where(
    prev['DAYS_FIRST_DUE'] < -40, np.nan)
prev['AMT_CREDIT_2'] = prev['AMT_CREDIT_temp'].where(
    prev['DAYS_FIRST_DUE'] >= -900, np.nan)
prev['AMT_CREDIT_3'] = prev['AMT_CREDIT_temp'].where(
    prev['DAYS_FIRST_DUE'] < -900, np.nan)

prev['oblomis'] = (prev['AMT_APPLICATION']/prev['AMT_CREDIT'])
prev['oblomis'] = prev['oblomis'].where(
    prev['NAME_CONTRACT_TYPE'] != 'Revolving loans', np.nan)

prev['CRR'] = prev['AMT_ANNUITY'].where(
    prev['CODE_REJECT_REASON'] == 'XAP', np.nan)

prev['1'] = prev['DAYS_LAST_DUE_1ST_VERSION']
prev['1'].replace(365243, np.nan, inplace=True)
prev['2'] = prev['DAYS_LAST_DUE']
prev['2'] = np.where((prev['1'] > 0) & (
    prev['2'] == 365243), np.nan, prev['2'])
prev['2'].replace(365243, 100, inplace=True)
prev['2'].describe()
prev['DPD'] = prev['1']-prev['2']
prev['DPD'] = prev['DPD'].where(prev['CODE_REJECT_REASON'] == 'XAP', np.nan)
del prev['1'], prev['2']
################################################################################################################### new features #####################################################################
prev['NAME_YIELD_GROUP']

inc_by_org = prev[['prev_length', 'NAME_YIELD_GROUP']].groupby(
    'NAME_YIELD_GROUP').median()['prev_length']
prev['lebngth_by_yield'] = prev['NAME_YIELD_GROUP'].map(inc_by_org)
prev['lebngth_by_yield'] = prev['lebngth_by_yield']/prev['prev_length']
prev['high'] = prev['lebngth_by_yield'].where(
    prev['NAME_YIELD_GROUP'] == 'high', np.nan)
prev['middle'] = prev['lebngth_by_yield'].where(
    prev['NAME_YIELD_GROUP'] == 'middle', np.nan)
#prev['low_normal']=prev['lebngth_by_yield'].where(prev['NAME_YIELD_GROUP']=='low_normal', np.nan)
prev['low_action']=prev['lebngth_by_yield'].where(prev['NAME_YIELD_GROUP']=='low_action', np.nan)
prev['XNA']=prev['lebngth_by_yield'].where(prev['NAME_YIELD_GROUP']=='XNA', np.nan)

prev['lebngth_by_yield']=prev['prev_length']-prev['CNT_PAYMENT']
prev['lebngth_by_yield']=prev['lebngth_by_yield'].where(prev['NAME_CONTRACT_TYPE']!='Revolving loans', np.nan)

prev['111111']=prev['SK_ID_PREV'].apply(lambda x: 1 if  x>0 else 0)
prev['NAME_GOODS_CATEGORY']=prev['111111'].where(prev['NAME_GOODS_CATEGORY']=='XNA', 0)
prev['prev_length']=prev['prev_length'].where(prev['NAME_CONTRACT_TYPE']!='Revolving loans', np.nan)

#  prev.to_csv('../data/dima_prev_feature.csv', index=False)

######################################################################################################################################################################

#################################################################################    installments    #################################################################
print('Read inst')
inst = pd.read_csv('../data/installments_payments.csv')
inst=inst[['SK_ID_CURR','SK_ID_PREV',
'NUM_INSTALMENT_VERSION',
'NUM_INSTALMENT_NUMBER',
'DAYS_INSTALMENT',
'DAYS_ENTRY_PAYMENT',
'AMT_INSTALMENT',
'AMT_PAYMENT'
]]
inst['AMT_PAYMENT'].describe()
inst['AMT_PAYMENT'].value_counts()
inst['AMT_PAYMENT'].isnull().sum()
#################################################################################     installments features    #######################################################
inst['DPD'] = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
inst['INST_CARD']=inst['AMT_INSTALMENT'].where(inst['NUM_INSTALMENT_VERSION']==0, np.nan)

#inst['DPD3'] = inst['DPD'].where(inst['NUM_INSTALMENT_NUMBER']<=3, np.nan)


######################################################################################################################################################################
inst_aggregations = {
'NUM_INSTALMENT_VERSION': ['var'],
'NUM_INSTALMENT_NUMBER': ['max'],
'DAYS_INSTALMENT': ['min','max'],
'DPD': ['max','sum'],
'INST_CARD': ['max'],
'AMT_PAYMENT': ['sum'],
'AMT_INSTALMENT': ['sum'],
}

avg_inst = inst.groupby(['SK_ID_PREV']).agg({**inst_aggregations})
avg_inst.columns = pd.Index(['INST_' + e[0] + "_" + e[1].upper() for e in avg_inst.columns.tolist()])

######################################################################################################################################################################
######################################################################################################################################################################
print('Read pos')
pos = pd.read_csv('../data/POS_CASH_balance.csv')
pos=pos[['SK_ID_CURR','SK_ID_PREV',
'NAME_CONTRACT_STATUS',
'CNT_INSTALMENT_FUTURE',
'SK_DPD',
'MONTHS_BALANCE',
'CNT_INSTALMENT'
]]
pos['SK_DPD'].describe()
pos['SK_DPD'].value_counts()
pos['SK_DPD'].isnull().sum()
#################################################################################     POS_cash features    #######################################################
pos['CNT_INSTALMENT']=pos['CNT_INSTALMENT'].where(pos['MONTHS_BALANCE']>=-4, np.nan)
pos['SK_DPD_3m']=pos['SK_DPD'].where(pos['MONTHS_BALANCE']>=-3, np.nan)

######################################################################################################################################################################
######################################################################################################################################################################
pos_aggregations = {
'CNT_INSTALMENT': ['var'],
'SK_DPD_3m': ['mean'],
}

avg_pos = pos.groupby(['SK_ID_PREV']).agg({**pos_aggregations})
avg_pos.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in avg_pos.columns.tolist()])

######################################################################################################################################################################

######################################################################################################################################################################
print('Read cc')
cc = pd.read_csv('../data/credit_card_balance.csv')

cc=cc[['SK_ID_CURR','SK_ID_PREV',
'SK_DPD',
'MONTHS_BALANCE',
'SK_DPD_DEF',
'NAME_CONTRACT_STATUS',
'AMT_BALANCE',
'AMT_CREDIT_LIMIT_ACTUAL',
'CNT_DRAWINGS_CURRENT',
'AMT_DRAWINGS_ATM_CURRENT',
'CNT_DRAWINGS_ATM_CURRENT',
'AMT_DRAWINGS_CURRENT',
'CNT_INSTALMENT_MATURE_CUM'

]]
cc['CNT_INSTALMENT_MATURE_CUM'].describe()
cc['CNT_INSTALMENT_MATURE_CUM'].value_counts()
cc['CNT_INSTALMENT_MATURE_CUM'].isnull().sum()
#################################################################################     cc features    #######################################################
cc['USE']=(cc['AMT_CREDIT_LIMIT_ACTUAL']/cc['AMT_BALANCE'])
cc['AMT_DRAWINGS_CURRENT']=cc['AMT_DRAWINGS_CURRENT'].where(cc['MONTHS_BALANCE']>=-4, np.nan)
cc['AMT_DRAWINGS_CURRENT_1']=cc['AMT_DRAWINGS_CURRENT'].where(cc['MONTHS_BALANCE']>=-1, np.nan)
cc['CNT_DRAWINGS_ATM_CURRENT']=cc['CNT_DRAWINGS_ATM_CURRENT'].where(cc['MONTHS_BALANCE']>=-6, np.nan)
cc['AMT_BALANCE_1']=cc['AMT_BALANCE'].where(cc['MONTHS_BALANCE']>=-1, np.nan)
######################################################################################################################################################################

cc_aggregations = {

'AMT_BALANCE': ['mean','max'],
'AMT_BALANCE_1': ['max'],
'USE': ['mean'],
'AMT_DRAWINGS_CURRENT': ['mean'],
'AMT_DRAWINGS_CURRENT_1': ['mean'],
'CNT_DRAWINGS_ATM_CURRENT': ['mean'],
'CNT_DRAWINGS_ATM_CURRENT': ['mean'],
'CNT_INSTALMENT_MATURE_CUM': ['max'],


}

avg_cc = cc.groupby(['SK_ID_PREV']).agg({**cc_aggregations})
avg_cc.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in avg_cc.columns.tolist()])

######################################################################################################################################################################
prev=prev.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_PREV')
prev=prev.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_PREV')
prev=prev.merge(right=avg_cc.reset_index(), how='left', on='SK_ID_PREV')

######  inst features ##############################
prev['inst_diff']=prev['CNT_PAYMENT']/prev['INST_NUM_INSTALMENT_NUMBER_MAX']
prev['inst_diff']=prev['inst_diff'].where(prev['NAME_CONTRACT_TYPE']!='Revolving loans', np.nan)

prev['inst_diff_1']=prev['inst_diff'].where(prev['DAYS_FIRST_DUE']>=-40, np.nan)
prev['inst_diff_temp']=prev['inst_diff'].where(prev['DAYS_FIRST_DUE']<-40, np.nan)
prev['inst_diff_2']=prev['inst_diff_temp'].where(prev['DAYS_FIRST_DUE']>=-900, np.nan)
prev['inst_diff_3']=prev['inst_diff_temp'].where(prev['DAYS_FIRST_DUE']<-900, np.nan)

prev['INST_DPD_MAX_1']=prev['INST_DPD_MAX'].where(prev['DAYS_FIRST_DUE']>=-40, np.nan)
prev['INST_DPD_MAX_temp']=prev['INST_DPD_MAX'].where(prev['DAYS_FIRST_DUE']<-40, np.nan)
prev['INST_DPD_MAX_2']=prev['INST_DPD_MAX_temp'].where(prev['DAYS_FIRST_DUE']>=-900, np.nan)
prev['INST_DPD_MAX_3']=prev['INST_DPD_MAX_temp'].where(prev['DAYS_FIRST_DUE']<-900, np.nan)

prev['CARD_USE']=prev['INST_INST_CARD_MAX']/prev['AMT_ANNUITY']
######  pos features ##############################
######  cc features ###############################



######################################################################################################################################################################
prev_aggregations = {
'count': ['count'],
'cash_active': ['count'],
'cash_not_active': ['count'],
'consumer_active': ['count'],
'consumer_2': ['count'],
'consumer_3': ['count'],
'consumer_4': ['count'],
'revolving_active': ['count'],
'revolving_2': ['count'],
'revolving_3': ['count'],
'cash_annui': ['median','min','max'],
'consumer_annui': ['median','min','max'],
'revolving_annui': ['median'],
'left2pay_sum': ['sum'],
'prev_length': ['median','min','max'],
'AMT_CREDIT_1': ['max'],
'AMT_CREDIT_2': ['max'],
'AMT_CREDIT_3': ['max'],
'oblomis': ['mean'],
'CRR': ['mean'],
'DPD': ['max'],

'high': ['min'],
'middle': ['min'],
#'low_normal': ['min'],
'low_action': ['min'],
'XNA': ['min'],
'lebngth_by_yield': ['min'],
######################################  inst features ##############################
'INST_NUM_INSTALMENT_VERSION_VAR': ['max'],
'inst_diff': ['min','max'],
'inst_diff_1': ['min'],
'inst_diff_2': ['min'],
'inst_diff_3': ['min'],
'INST_NUM_INSTALMENT_NUMBER_MAX': ['max'],
'INST_DPD_MAX': ['max'],
'INST_DPD_SUM': ['sum'],
'INST_DPD_MAX_1': ['mean'],
'INST_DPD_MAX_2': ['mean'],
'INST_DPD_MAX_3': ['mean'],
'CARD_USE': ['mean'],



######################################  pos features ###############################
'POS_CNT_INSTALMENT_VAR': ['mean'],
'POS_SK_DPD_3m_MEAN': ['mean'],
######################################  cc features ################################
'CC_AMT_BALANCE_MEAN': ['mean'],
'CC_AMT_BALANCE_MAX': ['max','mean'],  
'CC_AMT_BALANCE_1_MAX': ['sum'],
'CC_USE_MEAN': ['mean'],
'CC_AMT_DRAWINGS_CURRENT_MEAN': ['mean'],
'CC_CNT_DRAWINGS_ATM_CURRENT_MEAN': ['mean'],
####################################################################################
  }

categorical_feats = [f for f in prev.columns if prev[f].dtype == 'object']
le = preprocessing.LabelEncoder()
categorical_feats
for f_ in categorical_feats:
    prev[f_] = le.fit_transform(prev[f_].astype(str))

avg_prev = prev.groupby('SK_ID_CURR').agg({**prev_aggregations})
avg_prev.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in avg_prev.columns.tolist()])
################################################################################################################### read buro ########################################################################
print('Read buro')
buro = pd.read_csv('../data/bureau.csv')

#  buro=buro[buro_list]
buro['CNT_CREDIT_PROLONG'].describe()
buro['CNT_CREDIT_PROLONG'].value_counts()
buro['CNT_CREDIT_PROLONG'].isnull().sum()
################################################################################################################### 

buro['ACS'] = buro['AMT_CREDIT_SUM'].where(buro['CREDIT_TYPE']!='Credit card', np.nan)
buro['ACS_1']=buro['ACS'].where(buro['DAYS_CREDIT_UPDATE']>= -180, np.nan)
buro['ACS_2']=buro['ACS'].where(buro['DAYS_CREDIT_UPDATE']<-180, np.nan)


buro['11']=buro['AMT_CREDIT_MAX_OVERDUE'].where(buro['CREDIT_TYPE']=='Credit card', np.nan)
buro['AMT_CREDIT_MAX_OVERDUE']=buro['AMT_CREDIT_MAX_OVERDUE'].where(buro['CREDIT_TYPE']!='Credit card', np.nan)
buro['22']=buro['AMT_CREDIT_MAX_OVERDUE'].where(buro['CREDIT_ACTIVE']=='Active', np.nan)
buro['33']=buro['AMT_CREDIT_MAX_OVERDUE'].where(buro['CREDIT_ACTIVE']=='Closed', np.nan)
buro['44']=buro['AMT_CREDIT_MAX_OVERDUE'].where(buro['CREDIT_ACTIVE']=='Sold', np.nan)

buro['DDIF']=buro['DAYS_CREDIT_ENDDATE']-buro['DAYS_CREDIT_UPDATE']
buro['DDIF']=buro['DDIF'].where(buro['DAYS_CREDIT_ENDDATE']<0, np.nan)

buro['DDIF1']=buro['DDIF'].where(buro['CREDIT_TYPE']=='Credit card', np.nan)
buro['DDIF']=buro['DDIF'].where(buro['CREDIT_TYPE']!='Credit card', np.nan)
buro['DDIF']=buro['DDIF'].where(buro['CREDIT_TYPE']!='Mortgage', np.nan)
buro['DDIF']=buro['DDIF'].where(buro['CREDIT_TYPE']!='Loan for business development', np.nan)

#buro['longover']=buro['DAYS_CREDIT_ENDDATE']+buro['CREDIT_DAY_OVERDUE']
#buro['longover']=buro['longover'].where(buro['CREDIT_ACTIVE']!='Sold', np.nan)################################3
#buro['longover1']=buro['longover'].where(buro['CREDIT_TYPE']=='Credit card', np.nan)
#buro['longover']=buro['longover'].where(buro['CREDIT_TYPE']!='Credit card', np.nan)


buro['longover']=buro['DAYS_CREDIT_ENDDATE']+buro['CREDIT_DAY_OVERDUE']
buro['longover']=buro['longover'].where(buro['CREDIT_ACTIVE']!='Sold', np.nan)################################3
buro['longover1']=buro['longover'].where(buro['CREDIT_TYPE']=='Credit card', np.nan)
buro['longover']=buro['longover'].where(buro['CREDIT_TYPE']!='Credit card', np.nan)


buro['activeA']=buro['SK_ID_BUREAU'].where(buro['CREDIT_ACTIVE']=='Active', np.nan)
buro['closedA']=buro['SK_ID_BUREAU'].where(buro['CREDIT_ACTIVE']=='Closed', np.nan)
buro['DAYS_CREDIT_ENDDATE'].fillna(0,inplace=True)
buro['activeB']=buro['SK_ID_BUREAU'].where(buro['DAYS_CREDIT_ENDDATE']>=0, np.nan)
buro['closedB']=buro['SK_ID_BUREAU'].where(buro['DAYS_CREDIT_ENDDATE']<0, np.nan)
buro['type1']=buro['SK_ID_BUREAU'].where(buro['CREDIT_TYPE']=='Consumer credit', np.nan)
buro['type2']=buro['SK_ID_BUREAU'].where(buro['CREDIT_TYPE']=='Credit card', np.nan)
buro['type3']=buro['SK_ID_BUREAU'].where(buro['CREDIT_TYPE']=='Car loan', np.nan)
buro['type4']=buro['SK_ID_BUREAU'].where(buro['CREDIT_TYPE']=='Mortgage', np.nan)
buro['type5']=buro['SK_ID_BUREAU'].where(buro['CREDIT_TYPE']=='Microloan', np.nan)
buro['type6']=buro['SK_ID_BUREAU'].where(buro['CREDIT_TYPE']=='Loan for business development', np.nan)

buro['DAYS_CREDIT_1']=buro['DAYS_CREDIT'].where(buro['DAYS_CREDIT_UPDATE']>= -75, np.nan)
buro['DAYS_CREDIT_2']=buro['DAYS_CREDIT'].where(buro['DAYS_CREDIT_UPDATE']<-75, np.nan)

buro['AMT_CREDIT_SUM_DEBT'] =buro['AMT_CREDIT_SUM_DEBT'].fillna(0) #########################
buro['AMT_CREDIT_SUM_DEBT']=buro['AMT_CREDIT_SUM_DEBT'].where(buro['CREDIT_TYPE']=='Credit card', np.nan)

buro['AMT_CREDIT_SUM'] = buro['AMT_CREDIT_SUM'].fillna(0) #######################3
buro['AMT_CREDIT_SUM']=buro['AMT_CREDIT_SUM'].where(buro['CREDIT_TYPE']=='Credit card', np.nan)

buro['AMT_CREDIT_SUM_DEBT_1']=buro['AMT_CREDIT_SUM_DEBT'].where(buro['DAYS_CREDIT_UPDATE']>= -180, np.nan)
buro['AMT_CREDIT_SUM_DEBT_2']=buro['AMT_CREDIT_SUM_DEBT'].where(buro['DAYS_CREDIT_UPDATE']<-180, np.nan)
buro['AMT_CREDIT_SUM_1']=buro['AMT_CREDIT_SUM'].where(buro['DAYS_CREDIT_UPDATE']>= -180, np.nan)
buro['AMT_CREDIT_SUM_2']=buro['AMT_CREDIT_SUM'].where(buro['DAYS_CREDIT_UPDATE']<-180, np.nan)

grp1 = buro[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT_1']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT_1'].sum().reset_index().rename( index = str, columns = { 'AMT_CREDIT_SUM_DEBT_1': 'TOTAL_CUSTOMER_DEBT_1'})
grp2 = buro[['SK_ID_CURR', 'AMT_CREDIT_SUM_1']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM_1'].sum().reset_index().rename( index = str, columns = { 'AMT_CREDIT_SUM_1': 'TOTAL_CUSTOMER_CREDIT_1'})
grp3 = buro[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT_2']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT_2'].sum().reset_index().rename( index = str, columns = { 'AMT_CREDIT_SUM_DEBT_2': 'TOTAL_CUSTOMER_DEBT_2'})
grp4 = buro[['SK_ID_CURR', 'AMT_CREDIT_SUM_2']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM_2'].sum().reset_index().rename( index = str, columns = { 'AMT_CREDIT_SUM_2': 'TOTAL_CUSTOMER_CREDIT_2'})

buro = buro.merge(grp1, on = ['SK_ID_CURR'], how = 'left')
buro = buro.merge(grp2, on = ['SK_ID_CURR'], how = 'left')
buro = buro.merge(grp3, on = ['SK_ID_CURR'], how = 'left')
buro = buro.merge(grp4, on = ['SK_ID_CURR'], how = 'left')
del grp1, grp2
gc.collect()

buro['DEBT_CREDIT_RATIO_1'] = buro['TOTAL_CUSTOMER_DEBT_1']/buro['TOTAL_CUSTOMER_CREDIT_1']
buro['DEBT_CREDIT_RATIO_2'] = buro['TOTAL_CUSTOMER_DEBT_2']/buro['TOTAL_CUSTOMER_CREDIT_2']

#buro['DEBT_CREDIT_RATIO_1']=buro['DEBT_CREDIT_RATIO'].where(buro['DAYS_CREDIT_UPDATE']>= -180, np.nan)
#buro['DEBT_CREDIT_RATIO_2']=buro['DEBT_CREDIT_RATIO'].where(buro['DAYS_CREDIT_UPDATE']<-180, np.nan)
################################################################################################################### 
buro_aggregations = {
#'activeA': ['count'],
'closedA': ['count'],
'closedA': ['count'],

'activeB': ['count'],
'closedB': ['count'],

'type1': ['count'],
#'type2': ['count'],
'type3': ['count'],
'type4': ['count'],
'type5': ['count'],
'type6': ['count'],
'CREDIT_CURRENCY': ['nunique'],

'DDIF': ['max'],
'DDIF1': ['max'],

'longover': ['max'],
'longover1': ['max'],

'11': ['max'],
'22': ['max','sum'],
'33': ['max','sum'],
'44': ['max','sum'],

'DAYS_CREDIT_1': ['max',],
'DAYS_CREDIT_2': ['max',],

'DEBT_CREDIT_RATIO_1': ['max',],
'DEBT_CREDIT_RATIO_2': ['max',],

'ACS_1': ['mean',],
'ACS_2': ['mean',],
}

avg_buro = buro.groupby(['SK_ID_CURR']).agg({**buro_aggregations})
avg_buro.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in avg_buro.columns.tolist()])
################################################################################################################### merge #####################################################################
merged_df=app.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
merged_df=merged_df.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
################################################################################################################### 
merged_df['PREV_CRR_MEAN']=merged_df['AMT_ANNUITY']/merged_df['PREV_CRR_MEAN']
merged_df['PREV_CC_USE_MEAN_MEAN']=merged_df['PREV_CC_USE_MEAN_MEAN']/merged_df['AMT_ANNUITY']
merged_df['L3']=merged_df['length']/merged_df['DAYS_LAST_PHONE_CHANGE']
del merged_df['DAYS_ID_PUBLISH']

merged_df['history']=merged_df['BURO_closedB_COUNT'].fillna(0)+merged_df['PREV_count_COUNT'].fillna(0)
del merged_df['PREV_count_COUNT'],merged_df['BURO_closedB_COUNT']
#del merged_df['AMT_ANNUITY1'], merged_df['AMT_CREDIT1']
###############################################################################################################################################################################################
categorical_feats = [f for f in merged_df.columns if merged_df[f].dtype == 'object']
le = preprocessing.LabelEncoder()
categorical_feats
for f_ in categorical_feats:
    merged_df[f_] = le.fit_transform(merged_df[f_].astype(str))
ikkdx=[merged_df.columns.get_loc(c) for c in categorical_feats]

del app,	 avg_buro,	 avg_cc,	 avg_inst,	 avg_pos,	 avg_prev,	 buro,	 cc,	 data, grp3,	 grp4,	 inst,	 pos,	 prev,	 prev_act,	 test
gc.collect()
#################################################################################################################################################


#uf1=pd.read_csv('../input/split-and-full/FULL_OLD_BURO_MMM.csv')
#merged_df=merged_df.merge(right=uf1, how='left', on='SK_ID_CURR')
#del uf1
#gc.collect()

merged_df.to_csv('../data/dima_strong_features.csv',index=False)
sys.exit()

data = merged_df[merged_df['TARGET'].notnull()]
test = merged_df[merged_df['TARGET'].isnull()]
del merged_df
gc.collect()

y = data['TARGET']
del data['TARGET'],test['TARGET']
