-- prefix: f008_big_ins-cur-pre-f3-
with
tmp as (
select
SK_ID_CURR
,SK_ID_PREV
, SUM(NUM_INSTALMENT_VERSION                ) AS NUM_INSTALMENT_VERSION_sum
, SUM(NUM_INSTALMENT_NUMBER                 ) AS NUM_INSTALMENT_NUMBER_sum
, SUM(DAYS_ENTRY_PAYMENT                    ) AS DAYS_ENTRY_PAYMENT_sum
, SUM(DAYS_INSTALMENT                       ) AS DAYS_INSTALMENT_sum
, SUM(AMT_INSTALMENT                        ) AS AMT_INSTALMENT_sum
, SUM(AMT_PAYMENT                           ) AS AMT_PAYMENT_sum
, SUM((AMT_INSTALMENT - AMT_PAYMENT)        ) AS DIFF_PAYMENT_sum
, AVG(NUM_INSTALMENT_VERSION                ) AS NUM_INSTALMENT_VERSION_mean
, AVG(NUM_INSTALMENT_NUMBER                 ) AS NUM_INSTALMENT_NUMBER_mean
, AVG(DAYS_ENTRY_PAYMENT                    ) AS DAYS_ENTRY_PAYMENT_mean
, AVG(DAYS_INSTALMENT                       ) AS DAYS_INSTALMENT_mean
, AVG(AMT_INSTALMENT                        ) AS AMT_INSTALMENT_mean
, AVG(AMT_PAYMENT                           ) AS AMT_PAYMENT_mean
, AVG((AMT_INSTALMENT - AMT_PAYMENT)        ) AS DIFF_PAYMENT_mean
, MAX(NUM_INSTALMENT_VERSION                ) AS NUM_INSTALMENT_VERSION_max
, MAX(NUM_INSTALMENT_NUMBER                 ) AS NUM_INSTALMENT_NUMBER_max
, MAX(DAYS_ENTRY_PAYMENT                    ) AS DAYS_ENTRY_PAYMENT_max
, MAX(DAYS_INSTALMENT                       ) AS DAYS_INSTALMENT_max
, MAX(AMT_INSTALMENT                        ) AS AMT_INSTALMENT_max
, MAX(AMT_PAYMENT                           ) AS AMT_PAYMENT_max
, MAX((AMT_INSTALMENT - AMT_PAYMENT)        ) AS DIFF_PAYMENT_max
, MIN(NUM_INSTALMENT_VERSION                ) AS NUM_INSTALMENT_VERSION_min
, MIN(NUM_INSTALMENT_NUMBER                 ) AS NUM_INSTALMENT_NUMBER_min
, MIN(DAYS_ENTRY_PAYMENT                    ) AS DAYS_ENTRY_PAYMENT_min
, MIN(DAYS_INSTALMENT                       ) AS DAYS_INSTALMENT_min
, MIN(AMT_INSTALMENT                        ) AS AMT_INSTALMENT_min
, MIN(AMT_PAYMENT                           ) AS AMT_PAYMENT_min
, MIN((AMT_INSTALMENT - AMT_PAYMENT)        ) AS DIFF_PAYMENT_min
, STDDEV_POP(NUM_INSTALMENT_VERSION         ) AS NUM_INSTALMENT_VERSION_std
, STDDEV_POP(NUM_INSTALMENT_NUMBER          ) AS NUM_INSTALMENT_NUMBER_std
, STDDEV_POP(DAYS_ENTRY_PAYMENT             ) AS DAYS_ENTRY_PAYMENT_std
, STDDEV_POP(DAYS_INSTALMENT                ) AS DAYS_INSTALMENT_std
, STDDEV_POP(AMT_INSTALMENT                 ) AS AMT_INSTALMENT_std
, STDDEV_POP(AMT_PAYMENT                    ) AS AMT_PAYMENT_std
, STDDEV_POP((AMT_INSTALMENT - AMT_PAYMENT) ) AS DIFF_PAYMENT_std

from `hori.ins_rank`
where
asc_rank_prev <=5
-- desc_rank_prev <=3
group by 
SK_ID_CURR 
,SK_ID_PREV

)

select
SK_ID_CURR
, AVG(NUM_INSTALMENT_VERSION_sum                ) AS NUM_INSTALMENT_VERSION_sum
, AVG(NUM_INSTALMENT_NUMBER_sum                 ) AS NUM_INSTALMENT_NUMBER_sum
, AVG(DAYS_ENTRY_PAYMENT_sum       ) AS DAYS_ENTRY_PAYMENT_sum
, AVG(DAYS_INSTALMENT_sum       ) AS DAYS_INSTALMENT_sum
, AVG(AMT_INSTALMENT_sum       ) AS AMT_INSTALMENT_sum
, AVG(AMT_PAYMENT_sum       ) AS AMT_PAYMENT_sum
, AVG(DIFF_PAYMENT_sum       ) AS DIFF_PAYMENT_sum
, AVG(NUM_INSTALMENT_VERSION_mean       ) AS NUM_INSTALMENT_VERSION_mean
, AVG(NUM_INSTALMENT_NUMBER_mean       ) AS NUM_INSTALMENT_NUMBER_mean
, AVG(DAYS_ENTRY_PAYMENT_mean       ) AS DAYS_ENTRY_PAYMENT_mean
, AVG(DAYS_INSTALMENT_mean       ) AS DAYS_INSTALMENT_mean
, AVG(AMT_INSTALMENT_mean       ) AS AMT_INSTALMENT_mean
, AVG(AMT_PAYMENT_mean       ) AS AMT_PAYMENT_mean
, AVG(DIFF_PAYMENT_mean       ) AS DIFF_PAYMENT_mean
, AVG(NUM_INSTALMENT_VERSION_max       ) AS NUM_INSTALMENT_VERSION_max
, AVG(NUM_INSTALMENT_NUMBER_max       ) AS NUM_INSTALMENT_NUMBER_max
, AVG(DAYS_ENTRY_PAYMENT_max       ) AS DAYS_ENTRY_PAYMENT_max
, AVG(DAYS_INSTALMENT_max       ) AS DAYS_INSTALMENT_max
, AVG(AMT_INSTALMENT_max       ) AS AMT_INSTALMENT_max
, AVG(AMT_PAYMENT_max       ) AS AMT_PAYMENT_max
, AVG(DIFF_PAYMENT_max       ) AS DIFF_PAYMENT_max
, AVG(NUM_INSTALMENT_VERSION_min       ) AS NUM_INSTALMENT_VERSION_min
, AVG(NUM_INSTALMENT_NUMBER_min       ) AS NUM_INSTALMENT_NUMBER_min
, AVG(DAYS_ENTRY_PAYMENT_min       ) AS DAYS_ENTRY_PAYMENT_min
, AVG(DAYS_INSTALMENT_min       ) AS DAYS_INSTALMENT_min
, AVG(AMT_INSTALMENT_min       ) AS AMT_INSTALMENT_min
, AVG(AMT_PAYMENT_min       ) AS AMT_PAYMENT_min
, AVG(DIFF_PAYMENT_min       ) AS DIFF_PAYMENT_min
, AVG(NUM_INSTALMENT_VERSION_std       ) AS NUM_INSTALMENT_VERSION_std
, AVG(NUM_INSTALMENT_NUMBER_std       ) AS NUM_INSTALMENT_NUMBER_std
, AVG(DAYS_ENTRY_PAYMENT_std       ) AS DAYS_ENTRY_PAYMENT_std
, AVG(DAYS_INSTALMENT_std       ) AS DAYS_INSTALMENT_std
, AVG(AMT_INSTALMENT_std       ) AS AMT_INSTALMENT_std
, AVG(AMT_PAYMENT_std       ) AS AMT_PAYMENT_std
, AVG(DIFF_PAYMENT_std       ) AS DIFF_PAYMENT_std

from tmp
group by 
SK_ID_CURR 
;
