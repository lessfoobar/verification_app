-- 003_grant_permissions.sql  
-- Role-specific permissions for database users

-- =============================================================================
-- verification_api - Main API backend operations
-- =============================================================================

-- Table permissions
GRANT SELECT, INSERT, UPDATE ON verification_requests TO verification_api;
GRANT SELECT, INSERT, UPDATE ON video_submissions TO verification_api;
GRANT SELECT, INSERT, UPDATE ON webhook_deliveries TO verification_api;
GRANT SELECT, INSERT ON audit_logs TO verification_api;
GRANT SELECT ON system_config TO verification_api;
GRANT SELECT ON encryption_keys TO verification_api;

-- Sequence permissions (for SERIAL/BIGSERIAL columns)
GRANT USAGE, SELECT ON SEQUENCE audit_logs_id_seq TO verification_api;

-- NO ACCESS to staff_users table (security isolation)

-- =============================================================================
-- verification_staff - Staff dashboard and manual review
-- =============================================================================

-- Read-only access to verification data for review
GRANT SELECT ON verification_requests TO verification_staff;
GRANT SELECT ON video_submissions TO verification_staff;
GRANT SELECT, INSERT, UPDATE ON verification_reviews TO verification_staff;
GRANT SELECT, INSERT ON audit_logs TO verification_staff;
GRANT SELECT ON system_config TO verification_staff;

-- Sequence permissions
GRANT USAGE, SELECT ON SEQUENCE audit_logs_id_seq TO verification_staff;

-- NO ACCESS to encryption_keys, webhook_deliveries, or staff_users

-- =============================================================================
-- verification_admin - System administration and management
-- =============================================================================

-- Full administrative access
GRANT SELECT, INSERT, UPDATE, DELETE ON staff_users TO verification_admin;
GRANT SELECT, INSERT, UPDATE, DELETE ON encryption_keys TO verification_admin;
GRANT SELECT, UPDATE ON system_config TO verification_admin;
GRANT SELECT ON verification_requests TO verification_admin;
GRANT SELECT ON video_submissions TO verification_admin;
GRANT SELECT ON verification_reviews TO verification_admin;
GRANT SELECT ON webhook_deliveries TO verification_admin;
GRANT SELECT, INSERT ON audit_logs TO verification_admin;

-- Sequence permissions
GRANT USAGE, SELECT ON SEQUENCE audit_logs_id_seq TO verification_admin;

-- =============================================================================
-- verification_audit - GDPR compliance and external audits
-- =============================================================================

-- Read-only access to all tables for audit purposes
GRANT SELECT ON verification_requests TO verification_audit;
GRANT SELECT ON video_submissions TO verification_audit;
GRANT SELECT ON verification_reviews TO verification_audit;
GRANT SELECT ON webhook_deliveries TO verification_audit;
GRANT SELECT ON audit_logs TO verification_audit;
GRANT SELECT ON staff_users TO verification_audit;
GRANT SELECT ON encryption_keys TO verification_audit;
GRANT SELECT ON system_config TO verification_audit;

-- Can insert audit logs for their own access tracking
GRANT INSERT ON audit_logs TO verification_audit;
GRANT USAGE, SELECT ON SEQUENCE audit_logs_id_seq TO verification_audit;

-- =============================================================================
-- verification_cleanup - Automated GDPR data retention
-- =============================================================================

-- Permissions for automated cleanup operations
GRANT DELETE ON video_submissions TO verification_cleanup;
GRANT DELETE ON verification_requests TO verification_cleanup;
GRANT SELECT ON system_config TO verification_cleanup;
GRANT INSERT ON audit_logs TO verification_cleanup;

-- Sequence permissions
GRANT USAGE, SELECT ON SEQUENCE audit_logs_id_seq TO verification_cleanup;

-- Grant execution permission for cleanup function
GRANT EXECUTE ON FUNCTION cleanup_expired_data() TO verification_cleanup;

-- =============================================================================
-- Row Level Security Policies (Enhanced Security)
-- =============================================================================

-- Enable RLS on sensitive tables
ALTER TABLE staff_users ENABLE ROW LEVEL SECURITY;
ALTER TABLE encryption_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE verification_reviews ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;

-- Staff users can only see their own record and active users for assignment
CREATE POLICY staff_users_own_record ON staff_users 
    FOR SELECT TO verification_staff 
    USING (username = current_user OR is_active = true);

CREATE POLICY staff_users_admin_full_access ON staff_users 
    FOR ALL TO verification_admin 
    USING (true);

-- Encryption keys: only admin and system can access active keys
CREATE POLICY encryption_keys_admin_access ON encryption_keys 
    FOR ALL TO verification_admin 
    USING (true);

CREATE POLICY encryption_keys_api_active_only ON encryption_keys 
    FOR SELECT TO verification_api 
    USING (is_active = true);

-- Reviews: staff can only see/modify their own reviews
CREATE POLICY verification_reviews_own_reviews ON verification_reviews 
    FOR ALL TO verification_staff 
    USING (reviewer_id = current_setting('app.current_staff_id', true)::UUID);

CREATE POLICY verification_reviews_admin_access ON verification_reviews 
    FOR SELECT TO verification_admin 
    USING (true);

-- Audit logs: prevent tampering, only inserts allowed for most users
CREATE POLICY audit_logs_insert_only ON audit_logs 
    FOR INSERT TO verification_api, verification_staff, verification_cleanup, verification_audit
    WITH CHECK (true);

CREATE POLICY audit_logs_admin_read ON audit_logs 
    FOR SELECT TO verification_admin, verification_audit
    USING (true);

-- Verification requests: API can access all, staff only those under review
CREATE POLICY verification_requests_api_access ON verification_requests 
    FOR ALL TO verification_api 
    USING (true);

CREATE POLICY verification_requests_staff_under_review ON verification_requests 
    FOR SELECT TO verification_staff 
    USING (status IN ('under_review', 'processing'));

-- Video submissions: restrict based on processing status
CREATE POLICY video_submissions_api_access ON video_submissions 
    FOR ALL TO verification_api 
    USING (true);

CREATE POLICY video_submissions_staff_for_review ON video_submissions 
    FOR SELECT TO verification_staff 
    USING (processing_status = 'completed');

-- =============================================================================
-- Application Context Functions for RLS
-- =============================================================================

-- Function to set current staff member context
CREATE OR REPLACE FUNCTION set_current_staff_id(staff_id UUID)
RETURNS void AS $
BEGIN
    PERFORM set_config('app.current_staff_id', staff_id::text, true);
END;
$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant execution to staff users
GRANT EXECUTE ON FUNCTION set_current_staff_id(UUID) TO verification_staff;
GRANT EXECUTE ON FUNCTION set_current_staff_id(UUID) TO verification_admin;

-- =============================================================================
-- Additional Security Settings
-- =============================================================================

-- Revoke default public permissions
REVOKE ALL ON verification_requests FROM PUBLIC;
REVOKE ALL ON video_submissions FROM PUBLIC;
REVOKE ALL ON verification_reviews FROM PUBLIC;
REVOKE ALL ON webhook_deliveries FROM PUBLIC;
REVOKE ALL ON audit_logs FROM PUBLIC;
REVOKE ALL ON staff_users FROM PUBLIC;
REVOKE ALL ON encryption_keys FROM PUBLIC;
REVOKE ALL ON system_config FROM PUBLIC;

-- Prevent users from creating objects in public schema
REVOKE CREATE ON SCHEMA public FROM verification_api;
REVOKE CREATE ON SCHEMA public FROM verification_staff;
REVOKE CREATE ON SCHEMA public FROM verification_audit;
REVOKE CREATE ON SCHEMA public FROM verification_cleanup;

-- Only admin can create objects
GRANT CREATE ON SCHEMA public TO verification_admin;