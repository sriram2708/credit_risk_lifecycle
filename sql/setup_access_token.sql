/* Run this once in a Snowsight worksheet, one statement at a time from the top.
   It lets your user sign in from Python with a programmatic access token,
   then creates the token.

   Snowflake normally requires a network policy before a token can be used.
   The authentication policy below removes that requirement for this user only.
   Your normal sign in to Snowsight is not changed. */

USE ROLE ACCOUNTADMIN;

CREATE AUTHENTICATION POLICY IF NOT EXISTS CREDIT_RISK.PUBLIC.PAT_WITHOUT_NETWORK_POLICY
    PAT_POLICY = (NETWORK_POLICY_EVALUATION = ENFORCED_NOT_REQUIRED);

ALTER USER SRIRAMV2704
    SET AUTHENTICATION POLICY CREDIT_RISK.PUBLIC.PAT_WITHOUT_NETWORK_POLICY;

/* This returns the token in the TOKEN_SECRET column.
   It is shown only once. Copy it straight into SNOWFLAKE_PASSWORD in your .env file.
   The token works for 30 days and only with the ACCOUNTADMIN role. */

ALTER USER SRIRAMV2704 ADD PROGRAMMATIC ACCESS TOKEN CREDIT_RISK_PAT
    ROLE_RESTRICTION = 'ACCOUNTADMIN'
    DAYS_TO_EXPIRY = 30
    COMMENT = 'Credit risk lifecycle project';

/* If you lose the token, remove it and run the statement above again:
   ALTER USER SRIRAMV2704 REMOVE PROGRAMMATIC ACCESS TOKEN CREDIT_RISK_PAT; */
