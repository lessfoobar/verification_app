#!/bin/bash

set -e

echo "🔐 Creating Podman secrets for verification service..."
echo "====================================================="

# Function to generate a secret with error handling
gen_secret() {
	local prefix="$1"
	local name="$2"

	# Check if secret already exists
	if podman secret exists "${name}" 2>/dev/null; then
		echo "⚠️  Secret '${name}' already exists, skipping..."
		return 0
	fi

	# Generate random secret
	local rand=""
  rand=$(openssl rand -hex 32)
	local secret_value="${prefix}_${rand}"

	# Create the secret
	if echo "${secret_value}" | podman secret create "${name}" -; then
		echo "✅ Created secret: ${name}"
	else
		echo "❌ Failed to create secret: ${name}"
		return 1
	fi
}

# Generate database passwords (these need to be secure)
echo ""
echo "📋 Generating database user passwords..."
gen_secret "pgpass" "pg_postgres_pass"
gen_secret "pgapi" "pg_api_pass"
gen_secret "pgstaff" "pg_staff_pass"
gen_secret "pgadmin" "pg_admin_pass"
gen_secret "pgaudit" "pg_audit_pass"
gen_secret "pgclean" "pg_cleanup_pass"

# Generate application secrets
echo ""
echo "📋 Generating application secrets..."
gen_secret "apikey" "api_secret_key"
gen_secret "dbenc" "db_encryption_key"
gen_secret "jwt" "jwt_secret"
gen_secret "redis" "redis_pass"

echo ""
echo "✅ All secrets created successfully!"
echo ""
echo "📋 Current Podman secrets:"
podman secret ls

echo ""
echo "⚠️  IMPORTANT SECURITY NOTES:"
echo "   • These secrets are stored securely in Podman"
echo "   • Never commit secrets to version control"
echo "   • In production, use a proper secret management system"
echo "   • Rotate secrets regularly"