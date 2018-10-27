import sys
import glob
import pandas as pd
import shutil
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 600)


feim = pd.read_csv('../valid/1027_131_lgb__feat516_CV0.7616204949624903_lr0.1.csv')[['feature', 'rank']]
feim.set_index('rank', inplace=True)

rank_list = [ 1 ,2 ,4 ,5 ,9 ,11 ,14 ,15 ,16 ,17 ,18 ,19 ,20 ,22 ,26 ,27 ,28 ,36 ,37 ,40 ,41 ,51 ,53 ,54 ,55 ,58 ,66 ,68 ,70 ,73 ,74 ,76 ,77 ,79 ,81 ,82 ,84 ,85 ,86 ,89 ,90 ,92 ,95 ,96 ,99 ,100 ,102 ,103 ,107 ,109 ,110 ,111 ,116 ,117 ,118 ,120 ,127 ,129 ,133 ,134 ,137 ,140 ,147 ,148 ,149 ,157 ,158 ,161 ,163 ,168 ,169 ,170 ,172 ,173 ,174 ,175 ,177 ,178 ,182 ,183 ,188 ,194 ,197 ,200 ,203 ,208 ,209 ,212 ,213 ,217 ,218 ,221 ,223 ,225 ,227 ,228 ,229 ,231 ,232 ,234 ,235 ,238 ,240 ,243 ,250 ,251 ,252 ,257 ,259 ,260 ,267 ,270 ,277 ,278 ,281 ,283 ,288 ,289 ,302 ,305 ,307 ,309 ,310 ,311 ,320 ,322 ,328 ,329 ,334 ,336 ,338 ,341 ,347 ,349 ,351 ,357 ,359 ,360 ,362 ,364 ,366 ,377 ,378 ,384 ,385 ,397 ,403 ,404 ,406 ,412 ,413 ,429 ,431 ,434 ,439 ,441 ,450 ,453 ,456 ,466 ,472 ,480 ,481 ,483 ,487 ,496 ,498 ,502 ,505 ,507 ,515]

feim = feim.loc[rank_list, :]
feat_list = feim['feature'].values

for feat in feat_list:
    shutil.move(f'../features/4_winner/train_{feat}.gz', '../features/2_second_valid')
    shutil.move(f'../features/4_winner/test_{feat}.gz', '../features/2_second_valid')
sys.exit()



#  2               104_app_EXT_SOURCE_2_pro_EXT_SOURCE_@     3
#  5               104_app_EXT_SOURCE_1_div_EXT_SOURCE_@     6
#  6              104_app_EXT_SOURCE_2_diff_EXT_SOURCE_@     7
#  7               104_app_EXT_SOURCE_2_div_EXT_SOURCE_@     8
#  9              104_app_EXT_SOURCE_1_diff_EXT_SOURCE_@    10
#  11              104_app_EXT_SOURCE_1_pro_EXT_SOURCE_@    12
#  12              104_app_EXT_SOURCE_1_pro_EXT_SOURCE_@    13
#  20            104_app_AMT_ANNUITY_pro_DAYS_ID_PUBLIS@    21
#  22              104_app_EXT_SOURCE_1_div_EXT_SOURCE_@    23
#  23     104_app_AMT_INCOME_TOTAL_pro_DAYS_REGISTRATIO@    24
#  24       104_app_AMT_INCOME_TOTAL_pro_DAYS_ID_PUBLIS@    25
#  28      104_app_DAYS_ID_PUBLISH_pro_DAYS_REGISTRATIO@    29
#  29            104_app_AMT_INCOME_TOTAL_pro_DAYS_BIRT@    30
#  30           104_app_AMT_ANNUITY_diff_AMT_GOODS_PRIC@    31
#  31                 104_app_AMT_ANNUITY_pro_DAYS_BIRT@    32
#  32             104_app_EXT_SOURCE_1_diff_EXT_SOURCE_@    33
#  33            104_app_AMT_CREDIT_pro_AMT_INCOME_TOTA@    34
#  34   104_app_AMT_ANNUITY_pro_AMT_REQ_CREDIT_BUREAU_..    35
#  37        104_app_DAYS_EMPLOYED_pro_DAYS_REGISTRATIO@    38
#  38          104_app_AMT_GOODS_PRICE_pro_DAYS_EMPLOYE@    39
#  41          104_app_DAYS_EMPLOYED_pro_DAYS_ID_PUBLIS@    42
#  42   104_app_DAYS_EMPLOYED_pro_DAYS_LAST_PHONE_CHANG@    43
#  43             104_app_DAYS_BIRTH_pro_DAYS_ID_PUBLIS@    44
#  44              104_app_AMT_ANNUITY_pro_DAYS_EMPLOYE@    45
#  45           104_app_AMT_ANNUITY_pro_AMT_INCOME_TOTA@    46
#  46           104_app_DAYS_BIRTH_pro_DAYS_REGISTRATIO@    47
#  47   104_app_DAYS_ID_PUBLISH_pro_DAYS_LAST_PHONE_CH..    48
#  49          104_app_AMT_ANNUITY_pro_DAYS_REGISTRATIO@    50
#  48   104_app_AMT_REQ_CREDIT_BUREAU_YEAR_div_EXT_SOU..    49
#  51                104_app_AMT_ANNUITY_pro_OWN_CAR_AG@    52
#  55   104_app_AMT_CREDIT_pro_AMT_REQ_CREDIT_BUREAU_Y..    56
#  56   104_app_AMT_REQ_CREDIT_BUREAU_HOUR_diff_DAYS_I..    57
#  58        104_app_AMT_GOODS_PRICE_pro_DAYS_ID_PUBLIS@    59
#  59      104_app_EXT_SOURCE_2_var@['ORGANIZATION_TYPE]    60
#  60         104_app_AMT_INCOME_TOTAL_pro_DAYS_EMPLOYE@    61
#  61   104_app_AMT_INCOME_TOTAL_pro_DAYS_LAST_PHONE_C..    62
#  62      104_app_AMT_GOODS_PRICE_pro_DAYS_REGISTRATIO@    63
#  63   104_app_AMT_REQ_CREDIT_BUREAU_YEAR_pro_DAYS_EM..    64
#  64             104_app_AMT_GOODS_PRICE_pro_DAYS_BIRT@    65
#  66             104_app_AMT_CREDIT_pro_DAYS_ID_PUBLIS@    67
#  68   104_app_AMT_REQ_CREDIT_BUREAU_DAY_diff_DAYS_BI..    69
#  70               104_app_AMT_CREDIT_pro_DAYS_EMPLOYE@    71
#  71        104_app_EXT_SOURCE_2_var@['OCCUPATION_TYPE]    72
#  74   104_app_AMT_GOODS_PRICE_pro_DAYS_LAST_PHONE_CH..    75
#  77   104_app_AMT_REQ_CREDIT_BUREAU_DAY_diff_DAYS_RE..    78
#  79   104_app_DAYS_LAST_PHONE_CHANGE_pro_DAYS_REGIST..    80
#  82             104_app_AMT_ANNUITY_diff_DAYS_EMPLOYE@    83
#  86   104_app_AMT_REQ_CREDIT_BUREAU_YEAR_diff_DAYS_L..    87
#  87   104_app_AMT_REQ_CREDIT_BUREAU_YEAR_pro_OWN_CAR..    88
#  90                  104_app_AMT_CREDIT_pro_DAYS_BIRT@    91
#  92   104_app_AMT_REQ_CREDIT_BUREAU_DAY_diff_DAYS_LA..    93
#  93         104_app_AMT_ANNUITY_diff_DAYS_REGISTRATIO@    94
#  96                 104_app_AMT_CREDIT_pro_OWN_CAR_AG@    97
#  97           104_app_AMT_INCOME_TOTAL_diff_DAYS_BIRT@    98
#  100          104_app_AMT_ANNUITY_diff_DAYS_ID_PUBLIS@   101
#  103                104_app_DAYS_BIRTH_pro_OWN_CAR_AG@   104
#  104  104_app_AMT_REQ_CREDIT_BUREAU_YEAR_pro_DAYS_ID..   105
#  105          104_app_AMT_INCOME_TOTAL_pro_OWN_CAR_AGE@   106
#  107               104_app_AMT_ANNUITY_diff_DAYS_BIRTH@   108
#  111  104_app_AMT_GOODS_PRICE_pro_AMT_REQ_CREDIT_BUR...   112
#  112     104_app_DAYS_BIRTH_pro_DAYS_LAST_PHONE_CHANGE@   113
#  113  104_app_AMT_REQ_CREDIT_BUREAU_YEAR_pro_DAYS_BI...   114
#  114  104_app_AMT_REQ_CREDIT_BUREAU_YEAR_div_DAYS_RE...   115
#  118  104_app_AMT_REQ_CREDIT_BUREAU_MON_diff_DAYS_BI...   119
#  120    104_app_AMT_ANNUITY_pro_DAYS_LAST_PHONE_CHANGE@   121
#  121           104_app_DAYS_ID_PUBLISH_pro_OWN_CAR_AGE@   122
#  122      104_app_AMT_GOODS_PRICE_pro_AMT_INCOME_TOTAL@   123
#  123  104_app_AMT_REQ_CREDIT_BUREAU_DAY_diff_DAYS_ID...   124
#  124             104_app_DAYS_EMPLOYED_pro_OWN_CAR_AGE@   125
#  125         104_app_AMT_GOODS_PRICE_div_DAYS_EMPLOYED@   126
#  127  104_app_AMT_REQ_CREDIT_BUREAU_MON_diff_DAYS_ID...   128
#  129  104_app_AMT_REQ_CREDIT_BUREAU_YEAR_pro_DAYS_LA...   130
#  130             104_app_AMT_CREDIT_diff_DAYS_EMPLOYED@   131
#  131  104_app_AMT_REQ_CREDIT_BUREAU_QRT_diff_DAYS_BI...   132
#  134           104_app_AMT_ANNUITY_pro_AMT_GOODS_PRICE@   135
#  135  104_app_AMT_GOODS_PRICE_div_DAYS_LAST_PHONE_CH...   136
#  137              104_app_DAYS_BIRTH_pro_DAYS_EMPLOYED@   138
#  138            104_app_AMT_GOODS_PRICE_div_DAYS_BIRTH@   139
#  140  104_app_AMT_REQ_CREDIT_BUREAU_YEAR_div_DAYS_LA...   141
#  141  104_app_AMT_REQ_CREDIT_BUREAU_WEEK_diff_DAYS_I...   142
#  142  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_diff_DAYS_E...   143
#  143     104_app_AMT_CREDIT_pro_DAYS_LAST_PHONE_CHANGE@   144
#  144  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_diff_DAYS_B...   145
#  145  104_app_AMT_GOODS_PRICE_diff_DAYS_LAST_PHONE_C...   146
#  149  104_app_AMT_INCOME_TOTAL_pro_AMT_REQ_CREDIT_BU...   150
#  150  104_app_AMT_REQ_CREDIT_BUREAU_QRT_diff_DAYS_RE...   151
#  151   104_app_AMT_INCOME_TOTAL_diff_DAYS_REGISTRATION@   152
#  152  104_app_AMT_REQ_CREDIT_BUREAU_YEAR_pro_DAYS_RE...   153
#  153  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_diff_DAYS_R...   154
#  154  104_app_AMT_REQ_CREDIT_BUREAU_MON_diff_DAYS_LA...   155
#  155  104_app_AMT_REQ_CREDIT_BUREAU_YEAR_div_DAYS_ID...   156
#  158  104_app_AMT_INCOME_TOTAL_div_DAYS_LAST_PHONE_C...   159
#  159  104_app_AMT_REQ_CREDIT_BUREAU_YEAR_div_DAYS_EM...   160
#  161          104_app_AMT_CREDIT_pro_DAYS_REGISTRATION@   162
#  163        104_app_AMT_INCOME_TOTAL_div_DAYS_EMPLOYED@   164
#  164                104_app_AMT_ANNUITY_pro_AMT_CREDIT@   165
#  165  104_app_AMT_INCOME_TOTAL_diff_AMT_REQ_CREDIT_B...   166
#  166       104_app_AMT_INCOME_TOTAL_diff_DAYS_EMPLOYED@   167
#  170               104_app_AMT_ANNUITY_diff_AMT_CREDIT@   171
#  175  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_diff_DAYS_L...   176
#  178     104_app_AMT_INCOME_TOTAL_diff_DAYS_ID_PUBLISH@   179
#  179              104_app_AMT_ANNUITY_diff_OWN_CAR_AGE@   180
#  180   104_app_AMT_ANNUITY_diff_DAYS_LAST_PHONE_CHANGE@   181
#  183         104_app_DAYS_REGISTRATION_pro_OWN_CAR_AGE@   184
#  184  104_app_AMT_INCOME_TOTAL_diff_DAYS_LAST_PHONE_...   185
#  185         104_app_AMT_INCOME_TOTAL_diff_OWN_CAR_AGE@   186
#  186  104_app_AMT_ANNUITY_diff_AMT_REQ_CREDIT_BUREAU...   187
#  188  104_app_AMT_REQ_CREDIT_BUREAU_MON_diff_DAYS_RE...   189
#  189      104_app_AMT_GOODS_PRICE_diff_DAYS_ID_PUBLISH@   190
#  190               104_app_AMT_ANNUITY_div_OWN_CAR_AGE@   191
#  191  104_app_AMT_REQ_CREDIT_BUREAU_YEAR_div_DAYS_BI...   192
#  192  104_app_EXT_SOURCE_2_var@['WEEKDAY_APPR_PROCES...   193
#  194        104_app_AMT_GOODS_PRICE_diff_DAYS_EMPLOYED@   195
#  195          104_app_AMT_GOODS_PRICE_diff_OWN_CAR_AGE@   196
#  197  104_app_AMT_REQ_CREDIT_BUREAU_YEAR_diff_DAYS_E...   198
#  198  104_app_AMT_REQ_CREDIT_BUREAU_YEAR_diff_DAYS_R...   199
#  200  104_app_AMT_REQ_CREDIT_BUREAU_WEEK_diff_DAYS_L...   201
#  201   104_app_EXT_SOURCE_2_var@['NAME_EDUCATION_TYPE']   202
#  203    104_app_EXT_SOURCE_2_var@['NAME_FAMILY_STATUS']   204
#  204  104_app_AMT_REQ_CREDIT_BUREAU_QRT_diff_DAYS_ID...   205
#  205      104_app_AMT_INCOME_TOTAL_div_DAYS_ID_PUBLISH@   206
#  206  104_app_AMT_REQ_CREDIT_BUREAU_WEEK_diff_DAYS_B...   207
#  209     104_app_AMT_GOODS_PRICE_div_DAYS_REGISTRATION@   210
#  210           104_app_AMT_INCOME_TOTAL_div_DAYS_BIRTH@   211
#  213  104_app_AMT_REQ_CREDIT_BUREAU_WEEK_diff_DAYS_R...   214
#  214       104_app_AMT_GOODS_PRICE_div_DAYS_ID_PUBLISH@   215
#  215    104_app_AMT_INCOME_TOTAL_div_DAYS_REGISTRATION@   216
#  218               104_app_AMT_CREDIT_diff_OWN_CAR_AGE@   219
#  219  104_app_AMT_GOODS_PRICE_diff_AMT_REQ_CREDIT_BU...   220
#  221           104_app_AMT_GOODS_PRICE_pro_OWN_CAR_AGE@   222
#  223  104_app_AMT_REQ_CREDIT_BUREAU_QRT_diff_DAYS_EM...   224
#  225          104_app_AMT_INCOME_TOTAL_div_OWN_CAR_AGE@   226
#  229  104_app_AMT_ANNUITY_diff_AMT_REQ_CREDIT_BUREAU...   230
#  232  104_app_AMT_ANNUITY_diff_AMT_REQ_CREDIT_BUREAU...   233
#  235  104_app_AMT_REQ_CREDIT_BUREAU_QRT_diff_DAYS_LA...   236
#  236  104_app_AMT_REQ_CREDIT_BUREAU_YEAR_diff_DAYS_I...   237
#  238    104_app_AMT_ANNUITY_div_DAYS_LAST_PHONE_CHANGE@   239
#  240  104_app_AMT_REQ_CREDIT_BUREAU_QRT_div_DAYS_BIRTH@   241
#  241  104_app_AMT_REQ_CREDIT_BUREAU_DAY_diff_DAYS_EM...   242
#  243                104_app_AMT_ANNUITY_div_DAYS_BIRTH@   244
#  244  104_app_AMT_INCOME_TOTAL_div_AMT_REQ_CREDIT_BU...   245
#  245  104_app_AMT_ANNUITY_diff_AMT_REQ_CREDIT_BUREAU...   246
#  246  104_app_AMT_REQ_CREDIT_BUREAU_WEEK_diff_DAYS_E...   247
#  247  104_app_AMT_REQ_CREDIT_BUREAU_QRT_pro_DAYS_REG...   248
#  248    104_app_DAYS_LAST_PHONE_CHANGE_pro_OWN_CAR_AGE@   249
#  252  104_app_AMT_REQ_CREDIT_BUREAU_QRT_pro_DAYS_LAS...   253
#  253  104_app_AMT_REQ_CREDIT_BUREAU_YEAR_diff_DAYS_B...   254
#  254  104_app_AMT_REQ_CREDIT_BUREAU_YEAR_diff_OWN_CA...   255
#  255  104_app_AMT_REQ_CREDIT_BUREAU_QRT_pro_DAYS_BIRTH@   256
#  257  104_app_AMT_REQ_CREDIT_BUREAU_MON_diff_OWN_CAR...   258
#  260  104_app_AMT_REQ_CREDIT_BUREAU_YEAR_div_OWN_CAR...   261
#  261  104_app_AMT_GOODS_PRICE_pro_AMT_REQ_CREDIT_BUR...   262
#  262  104_app_AMT_REQ_CREDIT_BUREAU_MON_diff_DAYS_EM...   263
#  263           104_app_AMT_GOODS_PRICE_diff_DAYS_BIRTH@   264
#  264  104_app_AMT_REQ_CREDIT_BUREAU_QRT_pro_DAYS_ID_...   265
#  265  104_app_AMT_INCOME_TOTAL_diff_AMT_REQ_CREDIT_B...   266
#  267  104_app_AMT_CREDIT_pro_AMT_REQ_CREDIT_BUREAU_QRT@   268
#  268  104_app_AMT_INCOME_TOTAL_diff_AMT_REQ_CREDIT_B...   269
#  270             104_app_AMT_ANNUITY_div_DAYS_EMPLOYED@   271
#  271           104_app_AMT_CREDIT_diff_DAYS_ID_PUBLISH@   272
#  272  104_app_AMT_REQ_CREDIT_BUREAU_QRT_div_EXT_SOUR...   273
#  273  104_app_AMT_REQ_CREDIT_BUREAU_MON_div_EXT_SOUR...   274
#  274  104_app_AMT_GOODS_PRICE_div_AMT_REQ_CREDIT_BUR...   275
#  275  104_app_AMT_REQ_CREDIT_BUREAU_QRT_div_DAYS_REG...   276
#  278  104_app_AMT_REQ_CREDIT_BUREAU_MON_pro_DAYS_REG...   279
#  279  104_app_AMT_INCOME_TOTAL_div_AMT_REQ_CREDIT_BU...   280
#  281  104_app_AMT_CREDIT_diff_AMT_REQ_CREDIT_BUREAU_...   282
#  283    104_app_AMT_GOODS_PRICE_diff_DAYS_REGISTRATION@   284
#  284  104_app_AMT_CREDIT_diff_AMT_REQ_CREDIT_BUREAU_...   285
#  285  104_app_AMT_ANNUITY_diff_AMT_REQ_CREDIT_BUREAU...   286
#  286         104_app_AMT_ANNUITY_div_DAYS_REGISTRATION@   287
#  289                104_app_AMT_CREDIT_diff_DAYS_BIRTH@   290
#  290  104_app_AMT_ANNUITY_diff_AMT_REQ_CREDIT_BUREAU...   291
#  291           104_app_AMT_GOODS_PRICE_div_OWN_CAR_AGE@   292
#  292           104_app_AMT_ANNUITY_div_DAYS_ID_PUBLISH@   293
#  293  104_app_AMT_CREDIT_diff_AMT_REQ_CREDIT_BUREAU_...   294
#  294    104_app_AMT_CREDIT_diff_DAYS_LAST_PHONE_CHANGE@   295
#  295  104_app_AMT_CREDIT_diff_AMT_REQ_CREDIT_BUREAU_...   296
#  296                104_app_AMT_CREDIT_div_OWN_CAR_AGE@   297
#  297  104_app_AMT_GOODS_PRICE_diff_AMT_REQ_CREDIT_BU...   298
#  298     104_app_EXT_SOURCE_2_var@['NAME_HOUSING_TYPE']   299
#  299          104_app_AMT_CREDIT_div_DAYS_REGISTRATION@   300
#  300  104_app_AMT_INCOME_TOTAL_diff_AMT_REQ_CREDIT_B...   301
#  302  104_app_AMT_GOODS_PRICE_diff_AMT_REQ_CREDIT_BU...   303
#  303  104_app_AMT_REQ_CREDIT_BUREAU_QRT_diff_OWN_CAR...   304
#  305  104_app_AMT_REQ_CREDIT_BUREAU_QRT_div_DAYS_ID_...   306
#  307  104_app_AMT_REQ_CREDIT_BUREAU_DAY_diff_OWN_CAR...   308
#  311              104_app_AMT_CREDIT_div_DAYS_EMPLOYED@   312
#  312  104_app_AMT_GOODS_PRICE_diff_AMT_REQ_CREDIT_BU...   313
#  313  104_app_AMT_REQ_CREDIT_BUREAU_QRT_pro_DAYS_EMP...   314
#  314         104_app_AMT_CREDIT_diff_DAYS_REGISTRATION@   315
#  315  104_app_AMT_REQ_CREDIT_BUREAU_MON_pro_DAYS_EMP...   316
#  316  104_app_AMT_REQ_CREDIT_BUREAU_MON_div_DAYS_ID_...   317
#  317  104_app_AMT_REQ_CREDIT_BUREAU_MON_div_DAYS_REG...   318
#  318  104_app_AMT_ANNUITY_pro_AMT_REQ_CREDIT_BUREAU_...   319
#  320  104_app_AMT_ANNUITY_pro_AMT_REQ_CREDIT_BUREAU_...   321
#  322  104_app_AMT_REQ_CREDIT_BUREAU_QRT_diff_AMT_REQ...   323
#  323  104_app_AMT_INCOME_TOTAL_div_AMT_REQ_CREDIT_BU...   324
#  324  104_app_AMT_REQ_CREDIT_BUREAU_WEEK_diff_OWN_CA...   325
#  325            104_app_AMT_CREDIT_div_DAYS_ID_PUBLISH@   326
#  326  104_app_AMT_CREDIT_pro_AMT_REQ_CREDIT_BUREAU_MON@   327
#  329  104_app_AMT_REQ_CREDIT_BUREAU_QRT_div_DAYS_EMP...   330
#  330  104_app_AMT_REQ_CREDIT_BUREAU_MON_div_DAYS_LAS...   331
#  331  104_app_AMT_ANNUITY_div_AMT_REQ_CREDIT_BUREAU_...   332
#  332     104_app_AMT_CREDIT_div_DAYS_LAST_PHONE_CHANGE@   333
#  334  104_app_AMT_REQ_CREDIT_BUREAU_MON_pro_DAYS_ID_...   335
#  336  104_app_AMT_ANNUITY_div_AMT_REQ_CREDIT_BUREAU_...   337
#  338            104_app_AMT_CREDIT_pro_AMT_GOODS_PRICE@   339
#  339  104_app_AMT_REQ_CREDIT_BUREAU_QRT_div_DAYS_LAS...   340
#  341  104_app_AMT_REQ_CREDIT_BUREAU_MON_pro_DAYS_LAS...   342
#  342  104_app_AMT_ANNUITY_div_AMT_REQ_CREDIT_BUREAU_...   343
#  343  104_app_AMT_GOODS_PRICE_diff_AMT_REQ_CREDIT_BU...   344
#  344  104_app_AMT_ANNUITY_div_AMT_REQ_CREDIT_BUREAU_...   345
#  345  104_app_AMT_CREDIT_div_AMT_REQ_CREDIT_BUREAU_Y...   346
#  347  104_app_AMT_REQ_CREDIT_BUREAU_MON_div_DAYS_BIRTH@   348
#  349                 104_app_AMT_CREDIT_div_DAYS_BIRTH@   350
#  351  104_app_AMT_REQ_CREDIT_BUREAU_QRT_div_AMT_REQ_...   352
#  352  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_diff_OWN_CA...   353
#  353  104_app_AMT_CREDIT_diff_AMT_REQ_CREDIT_BUREAU_...   354
#  354  104_app_AMT_GOODS_PRICE_pro_AMT_REQ_CREDIT_BUR...   355
#  355  104_app_AMT_CREDIT_diff_AMT_REQ_CREDIT_BUREAU_...   356
#  357    104_app_EXT_SOURCE_2_var@['WALLSMATERIAL_MODE']   358
#  360  104_app_AMT_ANNUITY_div_AMT_REQ_CREDIT_BUREAU_...   361
#  362  104_app_AMT_REQ_CREDIT_BUREAU_MON_pro_DAYS_BIRTH@   363
#  364  104_app_AMT_GOODS_PRICE_diff_AMT_REQ_CREDIT_BU...   365
#  366      104_app_EXT_SOURCE_2_var@['NAME_INCOME_TYPE']   367
#  367  104_app_AMT_INCOME_TOTAL_pro_AMT_REQ_CREDIT_BU...   368
#  368  104_app_AMT_GOODS_PRICE_div_AMT_REQ_CREDIT_BUR...   369
#  369  104_app_AMT_REQ_CREDIT_BUREAU_WEEK_pro_DAYS_EM...   370
#  370  104_app_AMT_CREDIT_div_AMT_REQ_CREDIT_BUREAU_QRT@   371
#  371       104_app_EXT_SOURCE_2_var@['NAME_TYPE_SUITE']   372
#  372    104_app_EXT_SOURCE_2_var@['FONDKAPREMONT_MODE']   373
#  373  104_app_AMT_CREDIT_div_AMT_REQ_CREDIT_BUREAU_MON@   374
#  374  104_app_AMT_ANNUITY_div_AMT_REQ_CREDIT_BUREAU_...   375
#  375  104_app_AMT_REQ_CREDIT_BUREAU_MON_div_DAYS_EMP...   376
#  378  104_app_AMT_REQ_CREDIT_BUREAU_WEEK_div_DAYS_EM...   379
#  379  104_app_AMT_REQ_CREDIT_BUREAU_QRT_div_OWN_CAR_...   380
#  380  104_app_AMT_REQ_CREDIT_BUREAU_WEEK_div_DAYS_RE...   381
#  381  104_app_AMT_INCOME_TOTAL_diff_AMT_REQ_CREDIT_B...   382
#  382  104_app_AMT_INCOME_TOTAL_div_AMT_REQ_CREDIT_BU...   383
#  385  104_app_AMT_INCOME_TOTAL_diff_AMT_REQ_CREDIT_B...   386
#  386  104_app_AMT_REQ_CREDIT_BUREAU_MON_pro_OWN_CAR_...   387
#  387  104_app_AMT_REQ_CREDIT_BUREAU_MON_diff_AMT_REQ...   388
#  388  104_app_AMT_REQ_CREDIT_BUREAU_MON_div_OWN_CAR_...   389
#  389  104_app_AMT_GOODS_PRICE_div_AMT_REQ_CREDIT_BUR...   390
#  390  104_app_AMT_REQ_CREDIT_BUREAU_MON_diff_AMT_REQ...   391
#  391  104_app_AMT_REQ_CREDIT_BUREAU_MON_div_AMT_REQ_...   392
#  392  104_app_AMT_REQ_CREDIT_BUREAU_WEEK_div_EXT_SOU...   393
#  393  104_app_AMT_INCOME_TOTAL_div_AMT_REQ_CREDIT_BU...   394
#  394  104_app_AMT_REQ_CREDIT_BUREAU_MON_pro_AMT_REQ_...   395
#  395  104_app_AMT_ANNUITY_pro_AMT_REQ_CREDIT_BUREAU_...   396
#  397  104_app_AMT_REQ_CREDIT_BUREAU_QRT_pro_OWN_CAR_...   398
#  398  104_app_AMT_REQ_CREDIT_BUREAU_WEEK_diff_AMT_RE...   399
#  399  104_app_AMT_CREDIT_div_AMT_REQ_CREDIT_BUREAU_W...   400
#  400  104_app_AMT_GOODS_PRICE_div_AMT_REQ_CREDIT_BUR...   401
#  401  104_app_AMT_GOODS_PRICE_div_AMT_REQ_CREDIT_BUR...   402
#  404  104_app_AMT_CREDIT_div_AMT_REQ_CREDIT_BUREAU_DAY@   405
#  406  104_app_AMT_CREDIT_div_AMT_REQ_CREDIT_BUREAU_H...   407
#  407  104_app_AMT_INCOME_TOTAL_div_AMT_REQ_CREDIT_BU...   408
#  408  104_app_AMT_REQ_CREDIT_BUREAU_WEEK_pro_DAYS_RE...   409
#  409  104_app_AMT_GOODS_PRICE_pro_AMT_REQ_CREDIT_BUR...   410
#  410  104_app_AMT_REQ_CREDIT_BUREAU_QRT_pro_AMT_REQ_...   411
#  413  104_app_AMT_INCOME_TOTAL_pro_AMT_REQ_CREDIT_BU...   414
#  414  104_app_AMT_GOODS_PRICE_div_AMT_REQ_CREDIT_BUR...   415
#  415  104_app_AMT_REQ_CREDIT_BUREAU_DAY_diff_AMT_REQ...   416
#  416  104_app_AMT_REQ_CREDIT_BUREAU_MON_div_AMT_REQ_...   417
#  417  104_app_AMT_REQ_CREDIT_BUREAU_WEEK_div_DAYS_ID...   418
#  418  104_app_AMT_CREDIT_pro_AMT_REQ_CREDIT_BUREAU_W...   419
#  419  104_app_AMT_REQ_CREDIT_BUREAU_WEEK_pro_DAYS_ID...   420
#  420  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_diff_AMT_RE...   421
#  421  104_app_AMT_REQ_CREDIT_BUREAU_DAY_pro_DAYS_ID_...   422
#  422                      001_AMT_REQ_CREDIT_BUREAU_QRT   423
#  423  104_app_AMT_REQ_CREDIT_BUREAU_WEEK_div_DAYS_BI...   424
#  424  104_app_AMT_REQ_CREDIT_BUREAU_WEEK_pro_DAYS_BI...   425
#  425  104_app_AMT_REQ_CREDIT_BUREAU_WEEK_pro_DAYS_LA...   426
#  426  104_app_AMT_REQ_CREDIT_BUREAU_DAY_div_DAYS_ID_...   427
#  427  104_app_AMT_REQ_CREDIT_BUREAU_WEEK_div_DAYS_LA...   428
#  429  104_app_AMT_REQ_CREDIT_BUREAU_DAY_diff_AMT_REQ...   430
#  431  104_app_AMT_REQ_CREDIT_BUREAU_QRT_div_AMT_REQ_...   432
#  432  104_app_AMT_REQ_CREDIT_BUREAU_MON_pro_AMT_REQ_...   433
#  434  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_pro_DAYS_EM...   435
#  435  104_app_AMT_REQ_CREDIT_BUREAU_WEEK_div_AMT_REQ...   436
#  436  104_app_AMT_CREDIT_pro_AMT_REQ_CREDIT_BUREAU_H...   437
#  437  104_app_AMT_GOODS_PRICE_pro_AMT_REQ_CREDIT_BUR...   438
#  439  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_pro_DAYS_ID...   440
#  441  104_app_AMT_REQ_CREDIT_BUREAU_DAY_div_AMT_REQ_...   442
#  442  104_app_AMT_REQ_CREDIT_BUREAU_DAY_diff_AMT_REQ...   443
#  443  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_div_EXT_SOU...   444
#  444  104_app_AMT_ANNUITY_pro_AMT_REQ_CREDIT_BUREAU_...   445
#  445  104_app_AMT_REQ_CREDIT_BUREAU_QRT_diff_AMT_REQ...   446
#  446  104_app_AMT_REQ_CREDIT_BUREAU_WEEK_pro_AMT_REQ...   447
#  447  104_app_AMT_REQ_CREDIT_BUREAU_MON_div_AMT_REQ_...   448
#  448  104_app_AMT_REQ_CREDIT_BUREAU_MON_diff_AMT_REQ...   449
#  450  104_app_AMT_REQ_CREDIT_BUREAU_DAY_diff_AMT_REQ...   451
#  451  104_app_AMT_REQ_CREDIT_BUREAU_WEEK_div_OWN_CAR...   452
#  453  104_app_AMT_INCOME_TOTAL_pro_AMT_REQ_CREDIT_BU...   454
#  454  104_app_AMT_REQ_CREDIT_BUREAU_DAY_div_EXT_SOUR...   455
#  456  104_app_AMT_REQ_CREDIT_BUREAU_DAY_pro_DAYS_EMP...   457
#  457  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_diff_AMT_RE...   458
#  458  104_app_AMT_GOODS_PRICE_pro_AMT_REQ_CREDIT_BUR...   459
#  459  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_pro_DAYS_RE...   460
#  460  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_div_DAYS_LA...   461
#  461  104_app_AMT_REQ_CREDIT_BUREAU_DAY_pro_DAYS_BIRTH@   462
#  462  104_app_AMT_REQ_CREDIT_BUREAU_DAY_pro_AMT_REQ_...   463
#  463  104_app_AMT_REQ_CREDIT_BUREAU_DAY_pro_DAYS_LAS...   464
#  464  104_app_AMT_REQ_CREDIT_BUREAU_DAY_pro_DAYS_REG...   465
#  466  104_app_AMT_REQ_CREDIT_BUREAU_DAY_div_DAYS_BIRTH@   467
#  467  104_app_AMT_REQ_CREDIT_BUREAU_DAY_div_AMT_REQ_...   468
#  468  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_div_DAYS_EM...   469
#  469  104_app_AMT_REQ_CREDIT_BUREAU_DAY_div_DAYS_EMP...   470
#  470  104_app_AMT_REQ_CREDIT_BUREAU_DAY_div_AMT_REQ_...   471
#  472  104_app_AMT_REQ_CREDIT_BUREAU_DAY_div_AMT_REQ_...   473
#  473  104_app_AMT_REQ_CREDIT_BUREAU_DAY_div_OWN_CAR_...   474
#  474  104_app_AMT_REQ_CREDIT_BUREAU_WEEK_pro_OWN_CAR...   475
#  475  104_app_AMT_REQ_CREDIT_BUREAU_DAY_div_DAYS_REG...   476
#  476  104_app_AMT_REQ_CREDIT_BUREAU_DAY_diff_AMT_REQ...   477
#  477  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_pro_OWN_CAR...   478
#  478  104_app_AMT_REQ_CREDIT_BUREAU_QRT_pro_AMT_REQ_...   479
#  481  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_div_DAYS_ID...   482
#  483                      001_AMT_REQ_CREDIT_BUREAU_DAY   484
#  484  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_pro_DAYS_LA...   485
#  485  104_app_AMT_INCOME_TOTAL_pro_AMT_REQ_CREDIT_BU...   486
#  487  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_div_OWN_CAR...   488
#  488  104_app_AMT_REQ_CREDIT_BUREAU_DAY_div_AMT_REQ_...   489
#  489  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_pro_AMT_REQ...   490
#  490  104_app_AMT_CREDIT_pro_AMT_REQ_CREDIT_BUREAU_DAY@   491
#  491  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_pro_AMT_REQ...   492
#  492  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_div_DAYS_RE...   493
#  493  104_app_AMT_REQ_CREDIT_BUREAU_DAY_pro_AMT_REQ_...   494
#  494  104_app_AMT_REQ_CREDIT_BUREAU_DAY_div_DAYS_LAS...   495
#  496  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_div_DAYS_BI...   497
#  498  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_div_AMT_REQ...   499
#  499  104_app_AMT_ANNUITY_pro_AMT_REQ_CREDIT_BUREAU_...   500
#  500  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_div_AMT_REQ...   501
#  502  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_div_AMT_REQ...   503
#  503  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_div_AMT_REQ...   504
#  505  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_diff_AMT_RE...   506
#  507  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_diff_AMT_RE...   508
#  508  104_app_AMT_REQ_CREDIT_BUREAU_MON_pro_AMT_REQ_...   509
#  509  104_app_AMT_REQ_CREDIT_BUREAU_DAY_pro_OWN_CAR_...   510
#  510  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_pro_AMT_REQ...   511
#  511  104_app_AMT_REQ_CREDIT_BUREAU_DAY_pro_AMT_REQ_...   512
#  512  104_app_AMT_REQ_CREDIT_BUREAU_DAY_pro_AMT_REQ_...   513
#  513  104_app_AMT_REQ_CREDIT_BUREAU_DAY_pro_AMT_REQ_...   514
#  515  104_app_AMT_REQ_CREDIT_BUREAU_HOUR_pro_DAYS_BI...   516
