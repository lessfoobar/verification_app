-- Video Verification Service - Database Schema
-- This file is executed on first container startup

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Main verification requests table
CREATE TABLE verification_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_app_id VARCHAR(255) NOT NULL,
    callback_url VARCHAR(500) NOT NULL,
    user_metadata JSONB NOT NULL DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'initiated' CHECK (status IN ('initiated', 'recording', 'processing', 'under_review', 'approved', 'denied', 'expired')),
    language VARCHAR(10) DEFAULT 'en',
    required_document_types TEXT[] NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Video submissions and processing results
CREATE TABLE video_submissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    verification_id UUID NOT NULL REFERENCES verification_requests(id) ON DELETE CASCADE,
    video_path VARCHAR(500) NOT NULL,
    encryption_key_id VARCHAR(255) NOT NULL,
    document_type VARCHAR(50) NOT NULL,
    file_size_mb DECIMAL(10,2) NOT NULL,
    duration_seconds INTEGER NOT NULL,
    resolution VARCHAR(20) NOT NULL,
    upload_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    face_detection_result JSONB DEFAULT '{}',
    processing_status VARCHAR(50) DEFAULT 'pending' CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
    quality_score DECIMAL(3,2),
    attempt_number INTEGER DEFAULT 1 CHECK (attempt_number BETWEEN 1 AND 3),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Manual review and approval workflow
CREATE TABLE verification_reviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    verification_id UUID NOT NULL REFERENCES verification_requests(id) ON DELETE CASCADE,
    video_id UUID NOT NULL REFERENCES video_submissions(id) ON DELETE CASCADE,
    reviewer_id UUID NOT NULL,
    review_status VARCHAR(50) NOT NULL CHECK (review_status IN ('approved', 'denied', 'needs_review', 'escalated')),
    confidence_score DECIMAL(3,2) CHECK (confidence_score BETWEEN 0.0 AND 1.0),
    reviewer_notes TEXT,
    reviewed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    review_duration_seconds INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Webhook delivery tracking
CREATE TABLE webhook_deliveries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    verification_id UUID NOT NULL REFERENCES verification_requests(id) ON DELETE CASCADE,
    callback_url VARCHAR(500) NOT NULL,
    payload JSONB NOT NULL,
    http_status INTEGER,
    response_body TEXT,
    attempt_number INTEGER DEFAULT 1 CHECK (attempt_number BETWEEN 1 AND 5),
    delivered_at TIMESTAMP WITH TIME ZONE,
    next_retry_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- GDPR audit logging
CREATE TABLE audit_logs (
    id BIGSERIAL PRIMARY KEY,
    verification_id UUID,
    action VARCHAR(100) NOT NULL,
    actor_id UUID,
    actor_type VARCHAR(50) CHECK (actor_type IN ('user', 'staff', 'system', 'admin')),
    resource_type VARCHAR(50) CHECK (resource_type IN ('video', 'data', 'verification', 'config')),
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    details JSONB DEFAULT '{}'
);

-- Staff user management
CREATE TABLE staff_users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL CHECK (role IN ('reviewer', 'senior_reviewer', 'admin', 'auditor')),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    password_changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    failed_login_attempts INTEGER DEFAULT 0
);

-- Encryption key management
CREATE TABLE encryption_keys (
    key_id VARCHAR(255) PRIMARY KEY,
    encrypted_key BYTEA NOT NULL,
    key_version INTEGER NOT NULL,
    algorithm VARCHAR(50) DEFAULT 'AES-256-GCM',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    rotated_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    purpose VARCHAR(50) DEFAULT 'video_encryption' CHECK (purpose IN ('video_encryption', 'data_encryption', 'master_key'))
);

-- System configuration table
CREATE TABLE system_config (
    key VARCHAR(100) PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    is_sensitive BOOLEAN DEFAULT false,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_by UUID
);

-- Create indexes for performance
CREATE INDEX idx_verification_requests_status ON verification_requests(status);
CREATE INDEX idx_verification_requests_created_at ON verification_requests(created_at);
CREATE INDEX idx_verification_requests_external_app ON verification_requests(external_app_id);
CREATE INDEX idx_verification_requests_expires_at ON verification_requests(expires_at);

CREATE INDEX idx_video_submissions_verification_id ON video_submissions(verification_id);
CREATE INDEX idx_video_submissions_processing_status ON video_submissions(processing_status);
CREATE INDEX idx_video_submissions_upload_timestamp ON video_submissions(upload_timestamp);

CREATE INDEX idx_verification_reviews_verification_id ON verification_reviews(verification_id);
CREATE INDEX idx_verification_reviews_reviewer_id ON verification_reviews(reviewer_id);
CREATE INDEX idx_verification_reviews_review_status ON verification_reviews(review_status);
CREATE INDEX idx_verification_reviews_reviewed_at ON verification_reviews(reviewed_at);

CREATE INDEX idx_webhook_deliveries_verification_id ON webhook_deliveries(verification_id);
CREATE INDEX idx_webhook_deliveries_next_retry_at ON webhook_deliveries(next_retry_at);
CREATE INDEX idx_webhook_deliveries_created_at ON webhook_deliveries(created_at);

CREATE INDEX idx_audit_logs_verification_id ON audit_logs(verification_id);
CREATE INDEX idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX idx_audit_logs_actor_id ON audit_logs(actor_id);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);

CREATE INDEX idx_staff_users_username ON staff_users(username);
CREATE INDEX idx_staff_users_email ON staff_users(email);
CREATE INDEX idx_staff_users_role ON staff_users(role);
CREATE INDEX idx_staff_users_is_active ON staff_users(is_active);

CREATE INDEX idx_encryption_keys_is_active ON encryption_keys(is_active);
CREATE INDEX idx_encryption_keys_purpose ON encryption_keys(purpose);

-- Insert default system configuration
INSERT INTO system_config (key, value, description, is_sensitive) VALUES 
    ('verification_price_eur', '0.50', 'Price per verification in EUR', false),
    ('max_recording_attempts', '3', 'Maximum video recording attempts per verification', false),
    ('recording_duration_seconds', '30', 'Required video recording duration', false),
    ('face_detection_confidence_threshold', '0.8', 'Minimum confidence for face detection', false),
    ('video_retention_days', '90', 'Days to retain encrypted videos', false),
    ('verification_result_retention_days', '2555', 'Days to retain verification results (7 years)', false),
    ('auto_delete_enabled', 'true', 'Enable automatic deletion of expired data', false),
    ('webhook_timeout_seconds', '30', 'Timeout for webhook delivery attempts', false),
    ('webhook_retry_attempts', '3', 'Number of webhook retry attempts', false),
    ('rate_limit_requests', '100', 'Rate limit requests per window', false),
    ('rate_limit_window_seconds', '60', 'Rate limit window in seconds', false)
ON CONFLICT (key) DO NOTHING;

-- Create a default admin user (password: admin123 - CHANGE IN PRODUCTION!)
INSERT INTO staff_users (username, email, password_hash, role) VALUES 
    ('admin', 'admin@verification-service.local', '$2a$10$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC/.og/at2.uheWG/igi', 'admin')
ON CONFLICT (username) DO NOTHING;

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_verification_requests_updated_at BEFORE UPDATE ON verification_requests FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_webhook_deliveries_updated_at BEFORE UPDATE ON webhook_deliveries FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function for automatic data cleanup (GDPR)
CREATE OR REPLACE FUNCTION cleanup_expired_data()
RETURNS INTEGER AS $$
DECLARE
    video_retention_days INTEGER;
    verification_retention_days INTEGER;
    deleted_videos INTEGER := 0;
    deleted_verifications INTEGER := 0;
BEGIN
    -- Get retention settings
    SELECT value::INTEGER INTO video_retention_days FROM system_config WHERE key = 'video_retention_days';
    SELECT value::INTEGER INTO verification_retention_days FROM system_config WHERE key = 'verification_result_retention_days';
    
    -- Delete expired videos (but keep verification results)
    DELETE FROM video_submissions 
    WHERE upload_timestamp < NOW() - INTERVAL '1 day' * video_retention_days;
    GET DIAGNOSTICS deleted_videos = ROW_COUNT;
    
    -- Delete very old verification requests (after legal retention period)
    DELETE FROM verification_requests 
    WHERE created_at < NOW() - INTERVAL '1 day' * verification_retention_days;
    GET DIAGNOSTICS deleted_verifications = ROW_COUNT;
    
    -- Log cleanup activity
    INSERT INTO audit_logs (action, actor_type, resource_type, details) VALUES (
        'automatic_cleanup',
        'system',
        'data',
        jsonb_build_object(
            'deleted_videos', deleted_videos,
            'deleted_verifications', deleted_verifications,
            'video_retention_days', video_retention_days,
            'verification_retention_days', verification_retention_days
        )
    );
    
    RETURN deleted_videos + deleted_verifications;
END;
$$ LANGUAGE plpgsql;

COMMENT ON DATABASE verification_db IS 'Video Verification Service Database - GDPR Compliant Identity Verification';
COMMENT ON TABLE verification_requests IS 'Main verification requests from external applications';
COMMENT ON TABLE video_submissions IS 'Encrypted video files uploaded by users';
COMMENT ON TABLE verification_reviews IS 'Manual review results by staff members';
COMMENT ON TABLE webhook_deliveries IS 'Webhook delivery tracking for external app callbacks';
COMMENT ON TABLE audit_logs IS 'GDPR audit trail for all data access and modifications';
COMMENT ON TABLE staff_users IS 'Staff members who can review and approve verifications';
COMMENT ON TABLE encryption_keys IS 'Encryption key management for video storage';
COMMENT ON TABLE system_config IS 'System-wide configuration parameters';