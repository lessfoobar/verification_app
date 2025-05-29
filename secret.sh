#!/bin/bash

set -e

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
	echo "   • Use '--length <N>' to override default length (default is 64)"
	echo "   • Never commit secrets to version control"
	echo ""
fi

# Clean all secrets
if [[ ${CLEAN} == true ]]; then
	echo "🧹 Cleaning up all secrets..."
	for name in "${!SECRETS[@]}"; do
		if podman secret exists "${name}" 2>/dev/null; then
			podman secret rm "${name}" >/dev/null && echo "❌ Deleted secret: ${name}"
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
	local should_regen=false

	if [[ ${REGEN_ONE} == "all" || ${REGEN_ONE} == "${name}" ]]; then
		should_regen=true
	fi

	# Skip if not regenerating and secret exists
	if [[ ${should_regen} == true ]]; then
		podman secret exists "${name}" 2>/dev/null && {
			echo "♻️  Regenerating secret: ${name}"
			podman secret rm "${name}" >/dev/null
		}
	else
		return 0
	fi

	local rand
	rand=$(openssl rand -base64 "${length}")
	local secret_value="${prefix:+${prefix}_}${rand}"

	if echo "${secret_value}" | podman secret create "${name}" -; then
		echo "✅ Created secret: ${name}"
	else
		echo "❌ Failed to create secret: ${name}"
		return 1
	fi
}

# Main loop
for name in "${!SECRETS[@]}"; do
	gen_secret "${SECRETS[${name}]}" "${name}" "${SECRET_LENGTH}"
done

echo ""
