-- Megaline Analysis & ML Integration
-- Megaline Analysis Views
-- This script creates views for monthly usage summaries, revenue,
-- and a combined feature set to support ML pipelines.

USE megaline_db;

# Drop views if they exist to avoid conflicts during re-runs
DROP VIEW IF EXISTS monthly_usage_summary;
DROP VIEW IF EXISTS user_revenue_summary;
DROP VIEW IF EXISTS megaline_features;

# Create a monthly usage summary view
CREATE VIEW monthly_usage_summary AS
SELECT 
    user_id,
    AVG(calls) AS avg_calls,
    AVG(minutes) AS avg_minutes,
    AVG(messages) AS avg_messages,
    AVG(mb_used) AS avg_mb
FROM users_behavior
GROUP BY user_id;

# Create a revenue metrics view
CREATE VIEW user_revenue_summary AS
SELECT 
    user_id,
    SUM(CASE WHEN is_ultra = 1 THEN 70 ELSE 20 END) AS total_revenue,
    AVG(CASE WHEN is_ultra = 1 THEN 70 ELSE 20 END) AS avg_revenue
FROM users_behavior
GROUP BY user_id
;

# Create a combined feature view
CREATE VIEW megaline_features AS
SELECT 
    u.user_id,
    u.calls,
    u.minutes,
    u.messages,
    u.mb_used,
    u.is_ultra,
    mus.avg_calls,
    mus.avg_minutes,
    mus.avg_messages,
    mus.avg_mb,
    urs.total_revenue,
    urs.avg_revenue
FROM users_behavior u
LEFT JOIN monthly_usage_summary mus ON u.user_id = mus.user_id
LEFT JOIN user_revenue_summary urs ON u.user_id = urs.user_id
;

# Quick Check
SELECT * 
FROM megaline_features
LIMIT 10
;

SELECT * FROM megaline_features LIMIT 10
;