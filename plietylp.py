-- 1. Check for duplicates in join keys for each table
SELECT PATIENTID, CLM_DT, COUNT(*) as dupe_count
FROM PL_APP_RPHAI.RAW_RPHAI.NBA5_PATIENT_90D
GROUP BY PATIENTID, CLM_DT
HAVING COUNT(*) > 1;

SELECT PATIENTID, CLM_DT, COUNT(*) as dupe_count
FROM PL_APP_RPHAI.RAW_RPHAI.NBA5_PATIENT_180D
GROUP BY PATIENTID, CLM_DT
HAVING COUNT(*) > 1;

SELECT PATIENTID, CLM_DT, COUNT(*) as dupe_count
FROM PL_APP_RPHAI.RAW_RPHAI.NBA5_PATIENT_GCN_HIST
GROUP BY PATIENTID, CLM_DT
HAVING COUNT(*) > 1;

SELECT PATIENTID, CLM_DT, COUNT(*) as dupe_count
FROM PL_APP_RPHAI.RAW_RPHAI.NBA5_PATIENT_HIC_HIST
GROUP BY PATIENTID, CLM_DT
HAVING COUNT(*) > 1;

-- 2. Find which records are dropping in INNER JOIN
WITH inner_join_results AS (
    SELECT DISTINCT PATIENTID, CLM_DT
    FROM PL_APP_RPHAI.RAW_RPHAI.NBA5_WITH_PATIENT  -- your inner join version
),
missing_records AS (
    SELECT b.PATIENTID, b.CLM_DT, 
           CASE WHEN i.PATIENTID IS NULL THEN 'Missing' ELSE 'Present' END as status
    FROM PL_APP_RPHAI.RAW_RPHAI.NBA5_PATIENT_BASE b
    LEFT JOIN inner_join_results i
        ON b.PATIENTID = i.PATIENTID
        AND b.CLM_DT = i.CLM_DT
)
SELECT status, COUNT(*) as record_count
FROM missing_records
GROUP BY status;

-- 3. Identify which join is causing records to drop
SELECT 
    'BASE' as table_name,
    COUNT(*) as record_count
FROM PL_APP_RPHAI.RAW_RPHAI.NBA5_PATIENT_BASE
UNION ALL
SELECT 
    '90D',
    COUNT(*)
FROM PL_APP_RPHAI.RAW_RPHAI.NBA5_PATIENT_90D
UNION ALL
SELECT 
    '180D',
    COUNT(*)
FROM PL_APP_RPHAI.RAW_RPHAI.NBA5_PATIENT_180D
UNION ALL
SELECT 
    'GCN',
    COUNT(*)
FROM PL_APP_RPHAI.RAW_RPHAI.NBA5_PATIENT_GCN_HIST
UNION ALL
SELECT 
    'HIC',
    COUNT(*)
FROM PL_APP_RPHAI.RAW_RPHAI.NBA5_PATIENT_HIC_HIST;

-- 4. Sample the duplicated records to understand pattern
WITH duplicates AS (
    SELECT PATIENTID, CLM_DT, COUNT(*) as appear_count
    FROM PL_APP_RPHAI.RAW_RPHAI.NBA5_WITH_PATIENT
    GROUP BY PATIENTID, CLM_DT
    HAVING COUNT(*) > 1
)
SELECT w.*, d.appear_count
FROM PL_APP_RPHAI.RAW_RPHAI.NBA5_WITH_PATIENT w
JOIN duplicates d
    ON w.PATIENTID = d.PATIENTID
    AND w.CLM_DT = d.CLM_DT
LIMIT 10;