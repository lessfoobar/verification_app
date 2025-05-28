#!/bin/bash

set -e

echo "Creating Podman secrets for verification service..."

gen_secret() {
	local prefix="$1"
	local name="$2"

	rand=$(openssl rand -hex 32)
	podman secret create "${name}" <(echo "${prefix}_${rand}")
}

# Secret definitions
gen_secret "apikey" "api_secret_key"
gen_secret "dbenc" "db_encryption_key"
gen_secret "jwt" "jwt_secret"
gen_secret "pg_admin_pass" "pg_admin_pass"
gen_secret "pg_api_pass" "pg_api_pass"
gen_secret "pg_audit_pass" "pg_audit_pass"
gen_secret "pg_cleanup_pass" "pg_cleanup_pass"
gen_secret "pg_postgres_pass" "pg_postgres_pass"
gen_secret "pg_staff_pass" "pg_staff_pass"
gen_secret "redis_pass" "redis_pass"

echo
echo "✅ Podman secrets created successfully!"
podman secret ls
