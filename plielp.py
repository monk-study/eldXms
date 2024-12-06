-- Step 5A: Create base patient demographics and NDE history
DROP TABLE IF EXISTS PL_APP_RPHAI.RAW_RPHAI.NBA5_PATIENT_BASE;
CREATE TABLE PL_APP_RPHAI.RAW_RPHAI.NBA5_PATIENT_BASE AS (
    SELECT
        DRUG.*,
        PC.REQUEST_JSON,
--      FIRST create the table with entire json without parsing it. we will do this in next subsequent table.
--        -- Basic patient demographics - Location for geographic analysis
--        PARSE_JSON(PC.REQUEST_JSON):demographic.latitude::float AS PATIENT_LATITUDE,     -- Patient location latitude
--        PARSE_JSON(PC.REQUEST_JSON):demographic.longitude::float AS PATIENT_LONGITUDE,   -- Patient location longitude
--
--        -- NDE History - Patient behavior patterns
--        PARSE_JSON(PC.REQUEST_JSON):history.ndeHistory.newPatientFlag730d::string AS NEW_PATIENT_FLAG,  -- New patient indicator
--        PARSE_JSON(PC.REQUEST_JSON):history.ndeHistory.numOfCashScripts90d::int AS CASH_DISCOUNT_SCRIPTS_90D,  -- Cash discount usage
--        PARSE_JSON(PC.REQUEST_JSON):history.ndeHistory.numOfScripts90d::int AS CASH_SCRIPTS_90D,       -- Cash script count
--        PARSE_JSON(PC.REQUEST_JSON):history.ndeHistory.numOfScripts90d::int AS TOTAL_SCRIPTS_90D,      -- Total script volume
--        PARSE_JSON(PC.REQUEST_JSON):history.ndeHistory.pptCntWithSameCC90d::int AS SAME_CREDIT_CARD_COUNT_90D,  -- Payment method consistency

        PC.PATIENTID,              -- Patient identifier for joins
        PC.LOAD_DATE as PATIENT_HISTORY_LOAD_DATE  -- History snapshot date
    FROM PL_APP_RPHAI.RAW_RPHAI.NBA5_WITH_DRUG DRUG
    LEFT JOIN APP_RPHAI.CURATED_RPHAI.PATIENTCARD_JSON_HIST_V6 PC
        ON PC.PATIENTID = DRUG.CVS_CENTR_PAT_NBR
        AND PC.LOAD_DATE < DRUG.CLM_DT
    QUALIFY row_number() OVER (PARTITION BY CVS_CENTR_PAT_NBR ORDER BY LOAD_DATE desc) =1
--         DATEADD('DAY', -1, DRUG.CLM_DT)
--        DONT look at all load dates and claim dates. Only the claim date corresponding to specific claim
--        AND PC.LOAD_DATE = (
--            SELECT MAX(LOAD_DATE)
--            FROM APP_RPHAI.CURATED_RPHAI.PATIENTCARD_JSON_HIST_V6
--            WHERE LOAD_DATE < DRUG.CLM_DT
--        )
);

-- Step 5B: Add 180-day patient history
DROP TABLE IF EXISTS PL_APP_RPHAI.RAW_RPHAI.NBA5_PATIENT_180D;
CREATE TABLE PL_APP_RPHAI.RAW_RPHAI.NBA5_PATIENT_180D AS (
    SELECT
        BASE.*,
        -- Patient payment patterns over 180 days
        PARSE_JSON(PC.REQUEST_JSON):history.history180d.prcctTimesPaidCash::float AS PATIENT_CASH_RATE,        -- Cash payment frequency
        PARSE_JSON(PC.REQUEST_JSON):history.history180d.ttlTimesPaid::int AS PATIENT_TOTAL_PAID,              -- Total paid claims
        PARSE_JSON(PC.REQUEST_JSON):history.history180d.avgPntPayAmmt::float AS PATIENT_AVG_PAY_180D,         -- Average payment amount
        PARSE_JSON(PC.REQUEST_JSON):history.history180d.medianPntPayAmmt::float AS PATIENT_MEDIAN_PAY_180D,   -- Median payment amount
        PARSE_JSON(PC.REQUEST_JSON):history.history180d.highPntPayAmmt::float AS PATIENT_MAX_PAY_180D,        -- Maximum payment amount
        PARSE_JSON(PC.REQUEST_JSON):history.history180d.denialCount::int AS PATIENT_DENIAL_COUNT_180D         -- Number of denials
    FROM PL_APP_RPHAI.RAW_RPHAI.NBA5_PATIENT_BASE BASE
    LEFT JOIN APP_RPHAI.CURATED_RPHAI.PATIENTCARD_JSON_HIST_V6 PC
        ON PC.PATIENTID = BASE.PATIENTID
        AND PC.LOAD_DATE = BASE.PATIENT_HISTORY_LOAD_DATE
);
