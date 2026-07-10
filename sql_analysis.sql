-- ============================================================
-- Healthcare Patient Risk & Readmission Analytics
-- SQL script (MySQL syntax) — schema, cleaning, and analysis
-- ============================================================

-- 1. SCHEMA -----------------------------------------------------
CREATE DATABASE IF NOT EXISTS healthcare_analytics;
USE healthcare_analytics;

DROP TABLE IF EXISTS encounters;
CREATE TABLE encounters (
    patient_id          VARCHAR(10),
    age                 INT,
    gender              VARCHAR(10),
    department          VARCHAR(30),
    branch              VARCHAR(30),
    insurance_type      VARCHAR(30),
    admission_type      VARCHAR(20),
    admit_date          DATE,
    discharge_date      DATE,
    length_of_stay_days INT,
    comorbidity_count   INT,
    systolic_bp         INT,
    bmi                 DECIMAL(4,1),
    glucose_level       INT,
    billing_amount      DECIMAL(10,2),
    satisfaction_score  DECIMAL(3,1),
    readmitted_30d      TINYINT,
    CONSTRAINT chk_age CHECK (age BETWEEN 0 AND 120),
    CONSTRAINT chk_los CHECK (length_of_stay_days > 0)
);

-- Load with: LOAD DATA LOCAL INFILE 'healthcare_encounters.csv'
-- INTO TABLE encounters FIELDS TERMINATED BY ',' ENCLOSED BY '"'
-- LINES TERMINATED BY '\n' IGNORE 1 ROWS;

-- 2. DATA CLEANING -----------------------------------------------
-- Check nulls
SELECT
    SUM(bmi IS NULL) AS missing_bmi,
    SUM(satisfaction_score IS NULL) AS missing_satisfaction
FROM encounters;

-- Impute BMI and satisfaction_score with department-level median (illustrative — do the real
-- imputation in Python/pandas, this is here to demonstrate NULL-handling SQL)
UPDATE encounters e
JOIN (
    SELECT department, AVG(bmi) AS avg_bmi
    FROM encounters
    WHERE bmi IS NOT NULL
    GROUP BY department
) d ON e.department = d.department
SET e.bmi = d.avg_bmi
WHERE e.bmi IS NULL;

-- 3. CORE BUSINESS QUESTIONS --------------------------------------

-- Q1: Readmission rate by department, ranked
SELECT
    department,
    COUNT(*) AS total_encounters,
    SUM(readmitted_30d) AS readmissions,
    ROUND(100 * SUM(readmitted_30d) / COUNT(*), 2) AS readmit_rate_pct,
    RANK() OVER (ORDER BY SUM(readmitted_30d) / COUNT(*) DESC) AS risk_rank
FROM encounters
GROUP BY department
ORDER BY readmit_rate_pct DESC;

-- Q2: Average length of stay & billing by branch and admission type
SELECT
    branch,
    admission_type,
    ROUND(AVG(length_of_stay_days), 1) AS avg_los,
    ROUND(AVG(billing_amount), 0) AS avg_billing
FROM encounters
GROUP BY branch, admission_type
ORDER BY branch, avg_billing DESC;

-- Q3: Monthly readmission trend (window function: running average)
SELECT
    DATE_FORMAT(admit_date, '%Y-%m') AS admit_month,
    COUNT(*) AS encounters,
    SUM(readmitted_30d) AS readmissions,
    ROUND(100 * SUM(readmitted_30d) / COUNT(*), 2) AS readmit_rate_pct,
    ROUND(AVG(100 * SUM(readmitted_30d) / COUNT(*)) OVER (
        ORDER BY DATE_FORMAT(admit_date, '%Y-%m')
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 2) AS rolling_3mo_avg_rate
FROM encounters
GROUP BY admit_month
ORDER BY admit_month;

-- Q4: High-risk patient segments (comorbidity + age bucket)
SELECT
    CASE
        WHEN age < 30 THEN '<30'
        WHEN age BETWEEN 30 AND 50 THEN '30-50'
        WHEN age BETWEEN 51 AND 70 THEN '51-70'
        ELSE '70+'
    END AS age_bucket,
    comorbidity_count,
    COUNT(*) AS patients,
    ROUND(100 * SUM(readmitted_30d) / COUNT(*), 2) AS readmit_rate_pct
FROM encounters
GROUP BY age_bucket, comorbidity_count
HAVING COUNT(*) >= 20
ORDER BY readmit_rate_pct DESC
LIMIT 10;

-- Q5: Top 5 costliest patients per department (window function: ROW_NUMBER)
SELECT department, patient_id, billing_amount, dept_rank
FROM (
    SELECT
        department, patient_id, billing_amount,
        ROW_NUMBER() OVER (PARTITION BY department ORDER BY billing_amount DESC) AS dept_rank
    FROM encounters
) ranked
WHERE dept_rank <= 5;

-- Q6: Insurance type vs satisfaction and readmission (business-relevant cross-tab)
SELECT
    insurance_type,
    COUNT(*) AS patients,
    ROUND(AVG(satisfaction_score), 2) AS avg_satisfaction,
    ROUND(100 * SUM(readmitted_30d) / COUNT(*), 2) AS readmit_rate_pct,
    ROUND(AVG(billing_amount), 0) AS avg_billing
FROM encounters
GROUP BY insurance_type
ORDER BY readmit_rate_pct DESC;
