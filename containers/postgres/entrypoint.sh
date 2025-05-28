#!/bin/bash

set -e

# Defaults
PGDATA=${PGDATA:-/var/lib/pgsql/data}
CERTS_DIR=${CERTS_DIR:-/etc/ssl/postgresql}

echo "[INFO] PostgreSQL Verification Service - Starting initialization..."
echo "======================================================================"

# ------------------------------------------------------------------------------
# Load Secrets
# ------------------------------------------------------------------------------
echo "[INFO] Reading secrets from Podman secrets..."

if [ ! -f /run/secrets/pg_postgres_pass ]; then
    echo "[ERROR] ❌ Secrets not found in /run/secrets/"
    echo "[ERROR] Available files in /run/secrets/:"
    ls -la /run/secrets/ || echo "No secrets directory found"
    exit 1
fi

API_PASS=$(tr -d '\n' </run/secrets/pg_api_pass)
STAFF_PASS=$(tr -d '\n' </run/secrets/pg_staff_pass)
ADMIN_PASS=$(tr -d '\n' </run/secrets/pg_admin_pass)
AUDIT_PASS=$(tr -d '\n' </run/secrets/pg_audit_pass)
CLEANUP_PASS=$(tr -d '\n' </run/secrets/pg_cleanup_pass)
POSTGRES_PASS=$(tr -d '\n' </run/secrets/pg_postgres_pass)

echo "[INFO] ✅ All secrets loaded successfully"

# Clean secrets
rm -f /run/secrets/* 2>/dev/null || true

# ------------------------------------------------------------------------------
# Database Initialization
# ------------------------------------------------------------------------------
need_init=false

if [[ ! -s "${PGDATA}/PG_VERSION" ]]; then
    echo "[INFO] Database not initialized, performing first-time setup..."
    
    /usr/bin/initdb \
        --encoding=UTF-8 \
        --lc-collate=C \
        --lc-ctype=C \
        --pgdata="${PGDATA}" \
        --auth-local=peer \
        --auth-host=scram-sha-256 \
        --username=postgres

    echo "[INFO] ✅ Database initialized successfully"

    echo "[INFO] Updating PostgreSQL configuration..."
    mv /configs/postgresql.conf "${PGDATA}/"
    mv /configs/pg_hba.conf "${PGDATA}/"

    need_init=true
else
    echo "[INFO] Database already initialized, updating configuration..."
fi

# ------------------------------------------------------------------------------
# SSL Certificate Check
# ------------------------------------------------------------------------------
echo "[INFO] Verifying SSL certificates..."

if [ ! -f "$CERTS_DIR/server.crt" ] || [ ! -f "$CERTS_DIR/server.key" ] || [ ! -f "/etc/ssl/ca.crt" ]; then
    echo "[ERROR] ❌ SSL certificates missing!"
    echo "[ERROR] Expected:"
    echo "   - $CERTS_DIR/server.crt"
    echo "   - $CERTS_DIR/server.key"
    echo "   - /etc/ssl/ca.crt"
    ls -la "$CERTS_DIR/" || echo "Certificate directory not found"
    exit 1
fi

# ------------------------------------------------------------------------------
# Start PostgreSQL Temporarily
# ------------------------------------------------------------------------------
echo "[INFO] Starting PostgreSQL temporarily for setup..."
/usr/bin/pg_ctl -D "${PGDATA}" -l "${PGDATA}/setup.log" -o "-c listen_addresses=''" -w start || {
    echo "[ERROR] ❌ PostgreSQL failed to start. Log output:"
    cat "${PGDATA}/setup.log" || echo "No log file found"
    exit 1
}

# ------------------------------------------------------------------------------
# Run Migrations (on first init only)
# ------------------------------------------------------------------------------
if $need_init && [[ -d "/migrations" ]]; then
    echo "[INFO] Running database migrations..."

    for f in /migrations/001_*.sql; do
        [[ -f $f ]] || continue
        echo "[INFO] Applying $(basename "$f") to postgres database..."
        if ! psql -U postgres -d postgres -v ON_ERROR_STOP=1 -f "$f"; then
            echo "[ERROR] ❌ Migration failed: $(basename "$f")"
            pg_ctl -D "${PGDATA}" -m fast -w stop
            exit 1
        fi
    done

    for f in /migrations/002_*.sql /migrations/003_*.sql; do
        [[ -f $f ]] || continue
        echo "[INFO] Applying $(basename "$f") to verification_db database..."
        if ! psql -U postgres -d verification_db -v ON_ERROR_STOP=1 -f "$f"; then
            echo "[ERROR] ❌ Migration failed: $(basename "$f")"
            pg_ctl -D "${PGDATA}" -m fast -w stop
            exit 1
        fi
    done

    echo "[INFO] ✅ All migrations completed successfully"
fi

# ------------------------------------------------------------------------------
# Update User Passwords
# ------------------------------------------------------------------------------
echo "[INFO] Updating user passwords from secrets..."

psql -U postgres -d postgres -v ON_ERROR_STOP=1 <<-EOSQL || {
    echo "[ERROR] ❌ Failed to update user passwords"
    pg_ctl -D "${PGDATA}" -m fast -w stop
    exit 1
}
    ALTER USER verification_api PASSWORD '${API_PASS}';
    ALTER USER verification_staff PASSWORD '${STAFF_PASS}';
    ALTER USER verification_admin PASSWORD '${ADMIN_PASS}';
    ALTER USER verification_audit PASSWORD '${AUDIT_PASS}';
    ALTER USER verification_cleanup PASSWORD '${CLEANUP_PASS}';
    ALTER USER postgres PASSWORD '${POSTGRES_PASS}';
EOSQL

echo "[INFO] ✅ User passwords updated successfully"

# ------------------------------------------------------------------------------
# Stop Temporary PostgreSQL
# ------------------------------------------------------------------------------
echo "[INFO] Stopping temporary PostgreSQL..."
pg_ctl -D "${PGDATA}" -m fast -w stop

# ------------------------------------------------------------------------------
# Start PostgreSQL (Production)
# ------------------------------------------------------------------------------
echo "[INFO] ✅ Database initialization and setup completed!"
echo ""
echo "🚀 Starting PostgreSQL server with SSL enabled..."
echo "================================================"

exec /usr/bin/postgres \
    -D "${PGDATA}" \
    -c hba_file="${PGDATA}/pg_hba.conf" \
    -c config_file="${PGDATA}/postgresql.conf" \
    -c ssl=on \
    -c ssl_cert_file="${CERTS_DIR}/server.crt" \
    -c ssl_key_file="${CERTS_DIR}/server.key" \
    -c ssl_ca_file="/etc/ssl/ca.crt" \
    -c log_destination=stderr \
    -c logging_collector=off \
    -c log_statement=all
