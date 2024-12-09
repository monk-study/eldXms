-- 1. Check the exact structure of duplication
WITH dupe_check AS (
    SELECT PATIENTID, CLM_DT, COUNT(*) as ct
    FROM PL_APP_RPHAI.RAW_RPHAI.NBA5_WITH_PATIENT
    GROUP BY PATIENTID, CLM_DT
    HAVING COUNT(*) > 1
)
SELECT ct, COUNT(*) as freq
FROM dupe_check
GROUP BY ct
ORDER BY ct;

-- 2. For one specific PATIENTID with duplicates, let's see all columns
WITH sample_dupe AS (
    SELECT PATIENTID, CLM_DT
    FROM PL_APP_RPHAI.RAW_RPHAI.NBA5_WITH_PATIENT
    GROUP BY PATIENTID, CLM_DT
    HAVING COUNT(*) > 1
    LIMIT 1
)
SELECT *
FROM PL_APP_RPHAI.RAW_RPHAI.NBA5_WITH_PATIENT w
WHERE EXISTS (
    SELECT 1 
    FROM sample_dupe d 
    WHERE w.PATIENTID = d.PATIENTID 
    AND w.CLM_DT = d.CLM_DT
);
