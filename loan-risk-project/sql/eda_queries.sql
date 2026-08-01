-- ============================================================
-- Bank Loan Prediction & Risk Analysis — SQL Reference Queries
-- Table: loan_applicants (loaded from cs-training.csv)
-- ============================================================

-- 1. Overall default rate
SELECT
    ROUND(100.0 * SUM(CASE WHEN "SeriousDlqin2yrs" = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS default_rate_pct
FROM loan_applicants;

-- 2. Default rate by age group
SELECT
    CASE
        WHEN age < 25 THEN '<25'
        WHEN age BETWEEN 25 AND 34 THEN '25-34'
        WHEN age BETWEEN 35 AND 44 THEN '35-44'
        WHEN age BETWEEN 45 AND 54 THEN '45-54'
        WHEN age BETWEEN 55 AND 64 THEN '55-64'
        ELSE '65+'
    END AS age_group,
    COUNT(*) AS total_applicants,
    ROUND(100.0 * SUM("SeriousDlqin2yrs") / COUNT(*), 2) AS default_rate_pct
FROM loan_applicants
GROUP BY 1
ORDER BY 1;

-- 3. Default rate by number of dependents
SELECT
    "NumberOfDependents",
    COUNT(*) AS total_applicants,
    ROUND(100.0 * SUM("SeriousDlqin2yrs") / COUNT(*), 2) AS default_rate_pct
FROM loan_applicants
GROUP BY 1
ORDER BY 1;

-- 4. Applicants with any past-due history vs default rate
SELECT
    CASE
        WHEN ("NumberOfTime30-59DaysPastDueNotWorse" +
              "NumberOfTime60-89DaysPastDueNotWorse" +
              "NumberOfTimes90DaysLate") = 0 THEN 'No Past Due History'
        ELSE 'Has Past Due History'
    END AS past_due_flag,
    COUNT(*) AS total_applicants,
    ROUND(100.0 * SUM("SeriousDlqin2yrs") / COUNT(*), 2) AS default_rate_pct
FROM loan_applicants
GROUP BY 1;

-- 5. Income quartile vs default rate
WITH income_quartiles AS (
    SELECT *,
           NTILE(4) OVER (ORDER BY "MonthlyIncome") AS income_quartile
    FROM loan_applicants
    WHERE "MonthlyIncome" IS NOT NULL
)
SELECT
    income_quartile,
    ROUND(AVG("MonthlyIncome"), 0) AS avg_income,
    COUNT(*) AS total_applicants,
    ROUND(100.0 * SUM("SeriousDlqin2yrs") / COUNT(*), 2) AS default_rate_pct
FROM income_quartiles
GROUP BY 1
ORDER BY 1;

-- 6. High revolving utilization (>0.8) vs default rate
SELECT
    CASE WHEN "RevolvingUtilizationOfUnsecuredLines" > 0.8 THEN 'High Utilization (>80%)'
         ELSE 'Normal Utilization' END AS utilization_flag,
    COUNT(*) AS total_applicants,
    ROUND(100.0 * SUM("SeriousDlqin2yrs") / COUNT(*), 2) AS default_rate_pct
FROM loan_applicants
GROUP BY 1;

-- 7. Real estate loans vs default rate (does owning property reduce risk?)
SELECT
    "NumberRealEstateLoansOrLines",
    COUNT(*) AS total_applicants,
    ROUND(100.0 * SUM("SeriousDlqin2yrs") / COUNT(*), 2) AS default_rate_pct
FROM loan_applicants
GROUP BY 1
ORDER BY 1
LIMIT 10;
