SELECT
    is_ultra,
    COUNT(DISTINCT user_id) AS total_users,
    SUM(CASE WHEN last_activity = 0 THEN 1 ELSE 0 END) AS churned_users,
    ROUND(SUM(CASE WHEN last_activity = 0 THEN 1 ELSE 0 END) / COUNT(DISTINCT user_id), 4) AS churn_rate
FROM (
    SELECT 
        user_id,
        is_ultra,
        CASE 
            WHEN SUM(calls) = 0 AND SUM(messages) = 0 AND SUM(mb_used) = 0 THEN 0
            ELSE 1
        END AS last_activity
    FROM users_behavior
    GROUP BY user_id, is_ultra
) AS user_activity
GROUP BY is_ultra
;
