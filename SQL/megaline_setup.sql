-- Megaline Revenue Plan 
# Replicate the monthly revenue and user behavior calculations 

-- Megaline Database Setup
CREATE DATABASE IF NOT EXISTS megaline_db;
USE megaline_db;

-- Drop table if it already exists (clean start)
DROP TABLE IF EXISTS users_behavior;

-- Create users_behavior table
CREATE TABLE users_behavior (
    user_id INT,
    calls INT,
    minutes FLOAT,
    messages INT,
    mb_used FLOAT,
    is_ultra INT
);

-- Verify creation
SELECT * FROM users_behavior LIMIT 10;
