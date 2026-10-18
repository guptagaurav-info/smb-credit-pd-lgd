-- Portfolio analytics: cumulative charge-off rate by months-on-book for each approval vintage.
-- (Classic vintage curve used in SMB portfolio management and loss forecasting.)
WITH base AS (
    SELECT approval_fy,
           CASE WHEN is_default = 1 THEN DATEDIFF('month', disbursement_date, chgoff_date) END AS mob_at_chgoff
    FROM loans
    WHERE is_default IS NOT NULL AND approval_fy BETWEEN 1995 AND 2007 AND disbursement_date IS NOT NULL
), cohort AS (SELECT approval_fy, COUNT(*) AS n FROM base GROUP BY 1),
grid AS (SELECT range AS mob FROM range(0, 121, 6))
SELECT c.approval_fy, g.mob,
       COUNT(b.mob_at_chgoff) FILTER (WHERE b.mob_at_chgoff <= g.mob) * 1.0 / c.n AS cum_chgoff_rate
FROM cohort c CROSS JOIN grid g
LEFT JOIN base b ON b.approval_fy = c.approval_fy
GROUP BY c.approval_fy, g.mob, c.n
ORDER BY 1, 2;
