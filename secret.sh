#!/bin/bash

set -euo pipefail

echo "🔐 Podman Secret Manager"
echo "========================"

# Define secrets and prefixes
declare -A SECRETS=(
	["ca_password"]=""
	["pg_postgres_pass"]="pgpass"
	["pg_api_pass"]="pgapi"
	["pg_staff_pass"]="pgstaff"
	["pg_admin_pass"]="pgadmin"
	["pg_audit_pass"]="pgaudit"
	["pg_cleanup_pass"]="pgclean"
	["api_secret_key"]="apikey"
	["db_encryption_key"]="dbenc"
	["jwt_secret"]="jwt"
	["redis_pass"]="redis"
)

# Default values
REGEN_ONE=""
CLEAN=false
SHOW_INFO=false
SECRET_LENGTH=32

# Parse arguments
while [[ $# -gt 0 ]]; do
	case "$1" in
	--regen)
		REGEN_ONE="$2"
		if [[ -z ${REGEN_ONE} ]]; then
			echo "❌ Error: --regen requires a secret name or 'all'"
			exit 1
		fi
		shift
		;;
	--clean)
		CLEAN=true
		;;
	--info)
		SHOW_INFO=true
		;;
	--length)
		SECRET_LENGTH="$2"
		if ! [[ ${SECRET_LENGTH} =~ ^[0-9]+$ ]]; then
			echo "❌ Error: --length must be a number"
			exit 1
		fi
		shift
		;;
	*)
		echo "❌ Unknown option: $1"
		exit 1
		;;
	esac
	shift
done

# Optional info block
if [[ ${SHOW_INFO} == true ]]; then
	echo ""
	echo "⚠️  IMPORTANT SECURITY NOTES:"
	echo "   • These secrets are stored securely in Podman"
	echo "   • Use '--clean' to remove all generated secrets"
	echo "   • Use '--regen <name>' to regenerate just one secret"
	echo "   • Use '--regen all' to regenerate all secrets"
	echo "   • Use '--length <N>' to override default length (default is 32)"
	echo "   • Never commit secrets to version control"
	echo ""
fi

# Clean all secrets
if [[ ${CLEAN} == true ]]; then
	echo "🧹 Cleaning up all secrets..."
	for name in "${!SECRETS[@]}"; do
		if podman secret exists "${name}" 2>/dev/null; then
			secret_id=""
			if secret_id=$(podman secret inspect "${name}" --format '{{.ID}}' 2>/dev/null); then
				secret_id=$(echo "${secret_id}" | cut -c1-12)
				podman secret rm "${name}" >/dev/null && echo "❌ Deleted secret: ${name} (ID: ${secret_id})"
			else
				podman secret rm "${name}" >/dev/null && echo "❌ Deleted secret: ${name}"
			fi
		fi
	done
	echo "✅ Cleanup complete!"
	exit 0
fi

# Function to (re)generate a secret
gen_secret() {
	local prefix="$1"
	local name="$2"
	local length="${3:-${SECRET_LENGTH}}"
	local should_create=false
	local secret_id=""

	# Determine if we should create/regenerate this secret
	if [[ ${REGEN_ONE} == "all" || ${REGEN_ONE} == "${name}" ]]; then
		# Force regeneration requested
		should_create=true
		if podman secret exists "${name}" 2>/dev/null; then
			secret_id=""
			if secret_id=$(podman secret inspect "${name}" --format '{{.ID}}' 2>/dev/null); then
				secret_id=$(echo "${secret_id}" | cut -c1-12)
				podman secret rm "${name}" >/dev/null && echo "♻️  Regenerating secret: ${name} (ID: ${secret_id})"
			else
				podman secret rm "${name}" >/dev/null && echo "♻️  Regenerating secret: ${name}"
			fi
		fi
	elif [[ -z ${REGEN_ONE} ]]; then
		# No regeneration specified - create if doesn't exist
		if ! podman secret exists "${name}" 2>/dev/null; then
			should_create=true
		else
			secret_id=""
			if secret_id=$(podman secret inspect "${name}" --format '{{.ID}}' 2>/dev/null); then
				secret_id=$(echo "${secret_id}" | cut -c1-12)
				echo "✅ Secret already exists: ${name} (ID: ${secret_id})"
				return 0
			fi
		fi
	else
		# Regenerating specific secret, but this isn't it
		return 0
	fi

	# Create the secret if needed
	if [[ ${should_create} == true ]]; then
		local rand
		rand=$(openssl rand -base64 "${length}")
		local secret_value="${prefix:+${prefix}_}${rand}"

		if secret_id=$(echo "${secret_value}" | podman secret create "${name}" - 2>/dev/null); then
			secret_id=$(echo "${secret_id}" | cut -c1-12)
			echo "✅ Created secret: ${name} (ID: ${secret_id})"
		else
			echo "❌ Failed to create secret: ${name}"
			return 1
		fi
	fi
}

# Main loop - create all secrets
echo "🔧 Processing secrets..."
for name in "${!SECRETS[@]}"; do
	gen_secret "${SECRETS[${name}]}" "${name}" "${SECRET_LENGTH}"
done

echo ""
echo "✅ All secrets processed successfully!"