-- Megaline Analysis & ML Integration
-- Megaline Analysis Views
-- This script creates views for monthly usage summaries, revenue,
-- and a combined feature set to support ML pipelines.

-- Monthly Usage Aggregation
CREATE OR REPLACE VIEW monthly_usage AS
SELECT
    u.user_id,
    u.plan,
    u.city,
    DATE_FORMAT(c.call_date, '%Y-%m') AS month,
    COALESCE(SUM(c.duration), 0) AS total_minutes,
    (
        SELECT COUNT(*)
        FROM messages m
        WHERE m.user_id = u.user_id
          AND DATE_FORMAT(m.message_date, '%Y-%m') = DATE_FORMAT(c.call_date, '%Y-%m')
    ) AS messages_sent,
    (
        SELECT COALESCE(SUM(i.mb_used), 0)
        FROM internet i
        WHERE i.user_id = u.user_id
          AND DATE_FORMAT(i.session_date, '%Y-%m') = DATE_FORMAT(c.call_date, '%Y-%m')
    ) AS data_volume_mb
FROM users u
LEFT JOIN calls c ON u.user_id = c.user_id
GROUP BY u.user_id, u.plan, u.city, month;

-- Main Features View (joins plan pricing/limits)
CREATE OR REPLACE VIEW megaline_features AS
SELECT
    mu.user_id,
    mu.plan,
    mu.city,
    mu.month,
    COALESCE(mu.total_minutes, 0)    AS total_minutes,
    COALESCE(mu.messages_sent, 0)    AS messages_sent,
    COALESCE(mu.data_volume_mb, 0)   AS data_volume_mb,
    p.minutes_included,
    p.messages_included,
    p.mb_per_month_included,
    p.usd_monthly_pay,
    p.usd_per_minute,
    p.usd_per_message,
    p.usd_per_gb
FROM monthly_usage mu
LEFT JOIN plans p ON p.plan_name = mu.plan;

-- Average Usage per Plan
CREATE OR REPLACE VIEW average_usage AS
SELECT
    plan,
    AVG(total_minutes)  AS avg_minutes,
    AVG(messages_sent)  AS avg_messages,
    AVG(data_volume_mb) AS avg_data_mb
FROM megaline_features
GROUP BY plan;

-- Plan Comparison (Base Fee Overview)
CREATE OR REPLACE VIEW plan_comparison AS
SELECT
    plan,
    AVG(usd_monthly_pay) AS avg_base_fee
FROM megaline_features
GROUP BY plan;

-- Total Usage Summary (Aggregate Metrics)
CREATE OR REPLACE VIEW total_usage_summary AS
SELECT
    plan,
    SUM(total_minutes)   AS total_minutes,
    SUM(messages_sent)   AS total_messages,
    SUM(data_volume_mb)  AS total_data_mb
FROM megaline_features
GROUP BY plan;

-- AB Testing Metrics (Example)
CREATE OR REPLACE VIEW ab_testing_metrics AS
SELECT
    plan,
    AVG(data_volume_mb) AS avg_mb,
    AVG(total_minutes)  AS avg_minutes
FROM megaline_features
GROUP BY plan;
