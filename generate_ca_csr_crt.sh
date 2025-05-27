#!/bin/bash

set -e

mkdir -p certificates/CA

# Create CA key and cert if not exists (use ca.key and ca.crt)
if [ ! -f certificates/CA/ca.key ] || [ ! -f certificates/CA/ca.crt ]; then
  openssl genpkey -algorithm RSA -out certificates/CA/ca.key -aes256 -pass pass:changeit -pkeyopt rsa_keygen_bits:4096
  openssl req -x509 -new -nodes -key certificates/CA/ca.key -sha256 -days 3650 -out certificates/CA/ca.crt -subj "/CN=My Custom CA" -passin pass:changeit
fi

read -rp "Enter server domain (Common Name for CSR): " server_domain

read -rp "Do you want to add Subject Alternative Names (SAN)? (y/N): " add_san

san_config=""
if [[ "$add_san" =~ ^[Yy]$ ]]; then
  echo "Enter SANs separated by commas (e.g. DNS:example.com,DNS:www.example.com,IP:192.168.1.1):"
  read -r sans_input

  san_config="
[ req ]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[ req_distinguished_name ]
CN = $server_domain

[ v3_req ]
subjectAltName = $sans_input
"
else
  san_config="
[ req ]
distinguished_name = req_distinguished_name
prompt = no

[ req_distinguished_name ]
CN = $server_domain
"
fi

mkdir -p "certificates/$server_domain"

echo "$san_config" > "certificates/$server_domain/openssl.cnf"

openssl genpkey -algorithm RSA -out "certificates/$server_domain/server.key" -pkeyopt rsa_keygen_bits:2048
openssl req -new -key "certificates/$server_domain/server.key" -out "certificates/$server_domain/server.csr" -config "certificates/$server_domain/openssl.cnf"

cat > certificates/CA/v3_ca_ext.cnf <<EOF
basicConstraints = CA:FALSE
keyUsage = digitalSignature, keyEncipherment
subjectAltName = @alt_names

[alt_names]
EOF

if [[ "$add_san" =~ ^[Yy]$ ]]; then
  IFS=',' read -ra SAN_ARRAY <<< "$sans_input"
  index=1
  for san in "${SAN_ARRAY[@]}"; do
    san_type=${san%%:*}
    san_value=${san#*:}
    echo "$san_type.$index = $san_value" >> certificates/CA/v3_ca_ext.cnf
    ((index++))
  done
else
  echo "DNS.1 = $server_domain" >> certificates/CA/v3_ca_ext.cnf
fi

openssl x509 -req -in "certificates/$server_domain/server.csr" \
  -CA certificates/CA/ca.crt -CAkey certificates/CA/ca.key -CAcreateserial \
  -out "certificates/$server_domain/server.crt" -days 825 -sha256 \
  -extfile certificates/CA/v3_ca_ext.cnf -passin pass:changeit

rm -f "certificates/$server_domain/server.csr" "certificates/$server_domain/openssl.cnf"

echo "Server key and certificate created in certificates/$server_domain/"
