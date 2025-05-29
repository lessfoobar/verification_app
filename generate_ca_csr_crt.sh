#!/bin/bash

set -e

# Configuration
if [[ -f .env ]]; then
  source .env
fi

# Colors for output
RED="$(tput setaf 1 2>/dev/null)"
GREEN="$(tput setaf 2 2>/dev/null)"
YELLOW="$(tput setaf 3 2>/dev/null)"
BLUE="$(tput setaf 4 2>/dev/null)"
RESET="$(tput sgr0 2>/dev/null)"

# Usage function
usage() {
	echo "Usage: $0 [OPTIONS] [DOMAIN]"
	echo ""
	echo "Generate SSL certificates with CA signing"
	echo ""
	echo "OPTIONS:"
	echo "  -h, --help              Show this help message"
	echo "  -d, --domain DOMAIN     Server domain/hostname (default: localhost)"
	echo "  -p, --ca-pass PASSWORD  CA password (default: changeit)"
	echo "  -o, --output-dir DIR    Output directory (default: certificates)"
	echo "  --ca-only               Only create CA certificates"
	echo "  --san SANS              Subject Alternative Names (comma-separated)"
	echo "                          Format: DNS:example.com,DNS:www.example.com,IP:192.168.1.1"
	echo "  --service SERVICE       Generate for specific service (postgres, nginx, etc.)"
	echo ""
	echo "EXAMPLES:"
	echo "  $0                                   # Create CA and localhost cert"
	echo "  $0 --ca-only                         # Create only CA certificates"
	echo "  $0 --service postgres                # Create postgres server cert"
	echo "  $0 --service nginx --domain mysite.com"
	echo "  $0 --domain mysite.com --san 'DNS:www.mysite.com,IP:10.0.0.1'"
	echo ""
}

# Parse command line arguments
DOMAIN="${DEFAULT_DOMAIN}"
CA_PASS="${CA_PASSWORD}"
OUTPUT_DIR="${CERTS_BASE_DIR}"
CA_ONLY=false
SANS=""
SERVICE=""

while [[ $# -gt 0 ]]; do
	case $1 in
	-h | --help)
		usage
		exit 0
		;;
	-d | --domain)
		DOMAIN="$2"
		shift 2
		;;
	-p | --ca-pass)
		CA_PASS="$2"
		shift 2
		;;
	-o | --output-dir)
		OUTPUT_DIR="$2"
		CERTS_BASE_DIR="$2"
		CA_DIR="${OUTPUT_DIR}/CA"
		shift 2
		;;
	--ca-only)
		CA_ONLY=true
		shift
		;;
	--san)
		SANS="$2"
		shift 2
		;;
	--service)
		SERVICE="$2"
		shift 2
		;;
	-*)
		echo -e "${RED}Error: Unknown option $1${RESET}" >&2
		usage
		exit 1
		;;
	*)
		# Positional argument - treat as domain
		if [[ -z ${DOMAIN} ]]; then
			DOMAIN="$1"
		else
			echo -e "${RED}Error: Multiple domains specified${RESET}" >&2
			usage
			exit 1
		fi
		shift
		;;
	esac
done

# Set hostname and output directory based on service and domain
if [[ -n ${SERVICE} ]]; then
	# Generate service-specific hostname and separate directory for each service
	case "${SERVICE}" in
	postgres | postgresql)
		HOSTNAME="postgres.${DOMAIN}"
		SERVICE_OUTPUT_DIR="${OUTPUT_DIR}/${SERVICE}"
		;;
	nginx | proxy)
		HOSTNAME="nginx.${DOMAIN}"
		SERVICE_OUTPUT_DIR="${OUTPUT_DIR}/${SERVICE}"
		;;
	redis)
		HOSTNAME="redis.${DOMAIN}"
		SERVICE_OUTPUT_DIR="${OUTPUT_DIR}/${SERVICE}"
		;;
	api | grpc)
		HOSTNAME="api.${DOMAIN}"
		SERVICE_OUTPUT_DIR="${OUTPUT_DIR}/${SERVICE}"
		;;
	frontend | react)
		HOSTNAME="frontend.${DOMAIN}"
		SERVICE_OUTPUT_DIR="${OUTPUT_DIR}/${SERVICE}"
		;;
	*)
		HOSTNAME="${SERVICE}.${DOMAIN}"
		SERVICE_OUTPUT_DIR="${OUTPUT_DIR}/${SERVICE}"
		;;
	esac
else
	# No service specified - use domain as hostname and put in domain directory
	HOSTNAME="${DOMAIN}"
	SERVICE_OUTPUT_DIR="${OUTPUT_DIR}/${DOMAIN}"
fi

# Function to create CA certificates
create_ca() {
	echo -e "${BLUE}Creating Certificate Authority...${RESET}"

	mkdir -p "${CA_DIR}"

	# Generate CA private key
	if [[ ! -f "${CA_DIR}/ca.key" ]]; then
		echo -e "${YELLOW}Generating CA private key...${RESET}"
		openssl genpkey -algorithm RSA \
			-out "${CA_DIR}/ca.key" \
			-aes256 \
			-pass "pass:${CA_PASS}" \
			-pkeyopt rsa_keygen_bits:4096
		chmod 400 "${CA_DIR}/ca.key"
	else
		echo -e "${GREEN}CA private key already exists${RESET}"
	fi

	# Generate CA certificate
	if [[ ! -f "${CA_DIR}/ca.crt" ]]; then
		echo -e "${YELLOW}Generating CA certificate...${RESET}"
		openssl req -x509 -new -nodes \
			-key "${CA_DIR}/ca.key" \
			-sha256 \
			-days 3650 \
			-out "${CA_DIR}/ca.crt" \
			-passin "pass:${CA_PASS}" \
			-subj "/C=US/ST=State/L=City/O=Verification Service/OU=Certificate Authority/CN=Verification Service CA"
		chmod 444 "${CA_DIR}/ca.crt"
	else
		echo -e "${GREEN}CA certificate already exists${RESET}"
	fi
	echo -e "${YELLOW}⚠️  CA password used: ${CA_PASS}${RESET}"
	echo -e "${GREEN}✅ Certificate Authority created successfully${RESET}"
}

# Function to create server certificate
create_server_cert() {
	local hostname="$1"
	local output_dir="$2"

	echo -e "${BLUE}Creating server certificate for: ${hostname}${RESET}"
	echo -e "${BLUE}Output directory: ${output_dir}${RESET}"

	# Ensure CA exists
	if [[ ! -f "${CA_DIR}/ca.key" ]] || [[ ! -f "${CA_DIR}/ca.crt" ]]; then
		echo -e "${YELLOW}CA certificates not found, creating them first...${RESET}"
		create_ca
	fi

	mkdir -p "${output_dir}"

	# Prepare SAN configuration
	local san_config=""
	if [[ -n ${SANS} ]]; then
		san_config="${SANS}"
	else
		# Default SANs based on hostname
		if [[ ${hostname} == "localhost" ]]; then
			san_config="DNS:localhost,DNS:127.0.0.1,IP:127.0.0.1,IP:::1"
		elif [[ ${hostname} =~ \. ]]; then
			# FQDN - add hostname and base domain
			base_domain="${hostname#*.}"
			san_config="DNS:${hostname},DNS:${base_domain},DNS:localhost,IP:127.0.0.1"
		else
			# Simple hostname
			san_config="DNS:${hostname},DNS:localhost,IP:127.0.0.1"
		fi
	fi

	# Create OpenSSL configuration file
	cat >"${output_dir}/openssl.cnf" <<EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = State
L = City
O = Verification Service
OU = Server Certificate
CN = ${hostname}

[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = @alt_names

[alt_names]
EOF

	# Add SAN entries
	IFS=',' read -ra SAN_ARRAY <<<"${san_config}"
	index=1
	for san in "${SAN_ARRAY[@]}"; do
		san_type=${san%%:*}
		san_value=${san#*:}
		echo "${san_type}.${index} = ${san_value}" >>"${output_dir}/openssl.cnf"
		((index++))
	done

	# Generate server private key
	echo -e "${YELLOW}Generating server private key...${RESET}"
	openssl genpkey -algorithm RSA \
		-out "${output_dir}/server.key" \
		-pkeyopt rsa_keygen_bits:2048
	chmod 400 "${output_dir}/server.key"

	# Generate certificate signing request
	echo -e "${YELLOW}Generating certificate signing request...${RESET}"
	openssl req -new \
		-key "${output_dir}/server.key" \
		-out "${output_dir}/server.csr" \
		-config "${output_dir}/openssl.cnf"

	# Sign the certificate with CA
	echo -e "${YELLOW}Signing certificate with CA...${RESET}"
	openssl x509 -req \
		-in "${output_dir}/server.csr" \
		-CA "${CA_DIR}/ca.crt" \
		-CAkey "${CA_DIR}/ca.key" \
		-CAcreateserial \
		-out "${output_dir}/server.crt" \
		-days 825 \
		-sha256 \
		-extensions v3_req \
		-extfile "${output_dir}/openssl.cnf" \
		-passin "pass:${CA_PASS}"
	chmod 444 "${output_dir}/server.crt"

	# Clean up
	rm -f "${output_dir}/server.csr" "${output_dir}/openssl.cnf"

	echo -e "${GREEN}✅ Server certificate created successfully${RESET}"
	echo -e "${BLUE}📁 Certificate files:${RESET}"
	echo -e "   🔐 Private key: ${output_dir}/server.key"
	echo -e "   📜 Certificate: ${output_dir}/server.crt"
	echo -e "   🏛️  CA Certificate: ${CA_DIR}/ca.crt"

	echo -e "${YELLOW}⚠️  CA password used: ${CA_PASS}${RESET}"

	# Verify certificate
	echo -e "${YELLOW}Verifying certificate...${RESET}"
	if openssl verify -CAfile "${CA_DIR}/ca.crt" "${output_dir}/server.crt" >/dev/null 2>&1; then
		echo -e "${GREEN}✅ Certificate verification successful${RESET}"
	else
		echo -e "${RED}❌ Certificate verification failed${RESET}"
		return 1
	fi

	# Display certificate details
	echo -e "${BLUE}📋 Certificate details:${RESET}"
	openssl x509 -in "${output_dir}/server.crt" -noout -subject -dates -ext subjectAltName
}

# Main execution
main() {
	echo -e "${BLUE}🔐 SSL Certificate Generator${RESET}"
	echo -e "${BLUE}===========================${RESET}"
	echo ""

	# Create CA certificates
	create_ca

	# Create server certificate if not CA-only
	if [[ ${CA_ONLY} != true ]]; then
		echo ""
		create_server_cert "${HOSTNAME}" "${SERVICE_OUTPUT_DIR}"
	fi

	echo ""
	echo -e "${GREEN}🎉 Certificate generation completed successfully!${RESET}"

	if [[ -n ${SERVICE} ]]; then
		echo -e "${BLUE}📝 For service '${SERVICE}', certificates are ready at:${RESET}"
		echo -e "   ${SERVICE_OUTPUT_DIR}/"
		echo -e "   Certificate hostname: ${HOSTNAME}"
	fi
}

# Check dependencies
check_dependencies() {
	local missing_deps=()

	if ! command -v openssl >/dev/null 2>&1; then
		missing_deps+=("openssl")
	fi

	if [[ ${#missing_deps[@]} -gt 0 ]]; then
		echo -e "${RED}❌ Missing dependencies: ${missing_deps[*]}${RESET}" >&2
		echo "Please install the missing dependencies and try again." >&2
		exit 1
	fi
}

# Run dependency check
check_dependencies

# Execute main function
main "$@"