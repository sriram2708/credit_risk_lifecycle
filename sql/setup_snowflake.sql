/* Run this once in a Snowsight worksheet after you create your trial account.
   It creates a small warehouse that switches itself off after one minute of no use,
   so the free credits last, plus the database and the two schemas the project uses. */

USE ROLE ACCOUNTADMIN;

CREATE WAREHOUSE IF NOT EXISTS CREDIT_RISK_WH
    WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE;

CREATE DATABASE IF NOT EXISTS CREDIT_RISK;
CREATE SCHEMA IF NOT EXISTS CREDIT_RISK.RAW;
CREATE SCHEMA IF NOT EXISTS CREDIT_RISK.MART;

/* Check that it worked. */
SHOW WAREHOUSES LIKE 'CREDIT_RISK_WH';
SHOW SCHEMAS IN DATABASE CREDIT_RISK;
