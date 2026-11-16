/* Modelling table.
   Keeps 36 month loans issued from 2012 to 2015. All of these have finished
   by the end of the data, so every loan has a known result.
   Train on 2012 and 2013, test on 2014, monitor on 2015. */

CREATE OR REPLACE TABLE CREDIT_RISK.MART.LOANS_MODEL AS
SELECT
    ID,
    TO_DATE(ISSUE_DATE)                         AS ISSUE_DATE,
    LOAN_AMNT,
    ANNUAL_INC,
    DTI,
    EMP_LENGTH_YEARS,
    FICO_RANGE_LOW,
    DELINQ_2YRS,
    INQ_LAST_6MTHS,
    OPEN_ACC,
    PUB_REC,
    REVOL_BAL,
    REVOL_UTIL,
    TOTAL_ACC,
    CREDIT_HISTORY_YEARS,
    HOME_OWNERSHIP,
    PURPOSE,
    VERIFICATION_STATUS,
    FUNDED_AMNT,
    TOTAL_REC_PRNCP,
    RECOVERIES,
    IFF(LOAN_STATUS = 'Charged Off', 1, 0)      AS BAD,
    CASE
        WHEN YEAR(TO_DATE(ISSUE_DATE)) <= 2013 THEN 'train'
        WHEN YEAR(TO_DATE(ISSUE_DATE)) = 2014 THEN 'test'
        ELSE 'monitor'
    END                                         AS SPLIT
FROM CREDIT_RISK.RAW.LOANS
WHERE TERM_MONTHS = 36
  AND LOAN_STATUS IN ('Fully Paid', 'Charged Off')
  AND TO_DATE(ISSUE_DATE) BETWEEN '2012-01-01' AND '2015-12-31';

/* Quick check of loan counts and bad rate by quarter. */

CREATE OR REPLACE VIEW CREDIT_RISK.MART.BAD_RATE_BY_QUARTER AS
SELECT
    DATE_TRUNC('quarter', ISSUE_DATE)           AS ISSUE_QUARTER,
    SPLIT,
    COUNT(*)                                    AS LOANS,
    SUM(BAD)                                    AS BAD_LOANS,
    ROUND(AVG(BAD) * 100, 2)                    AS BAD_RATE_PERCENT,
    ROUND(SUM(FUNDED_AMNT))                     AS AMOUNT_FUNDED
FROM CREDIT_RISK.MART.LOANS_MODEL
GROUP BY 1, 2
ORDER BY 1;
