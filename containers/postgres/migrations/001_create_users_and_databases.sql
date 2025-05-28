-- 001_create_users_and_databases.sql
-- Creates database users and the verification database
-- Passwords will be updated via secrets in entrypoint script

-- Create database users without passwords (will be set from secrets in entrypoint)
CREATE USER verification_api;
CREATE USER verification_staff;
CREATE USER verification_admin;
CREATE USER verification_audit;
CREATE USER verification_cleanup;

-- Create the main verification database
CREATE DATABASE verification_db OWNER postgres;

-- Grant connect privileges to the database
GRANT CONNECT ON DATABASE verification_db TO verification_api;
GRANT CONNECT ON DATABASE verification_db TO verification_staff;
GRANT CONNECT ON DATABASE verification_db TO verification_admin;
GRANT CONNECT ON DATABASE verification_db TO verification_audit;
GRANT CONNECT ON DATABASE verification_db TO verification_cleanup;

-- Grant usage on public schema (required for table access)
\c verification_db
GRANT USAGE ON SCHEMA public TO verification_api;
GRANT USAGE ON SCHEMA public TO verification_staff;
GRANT USAGE ON SCHEMA public TO verification_admin;
GRANT USAGE ON SCHEMA public TO verification_audit;
GRANT USAGE ON SCHEMA public TO verification_cleanup;