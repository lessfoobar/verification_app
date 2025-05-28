#!/bin/bash

set -e

PGDATA=${PGDATA:-/var/lib/pgsql/data}

echo "[INFO] Reading secrets..."
API_PASS=$(tr -d '\n' </run/secrets/pg_api_pass)
STAFF_PASS=$(tr -d '\n' </run/secrets/pg_staff_pass)
ADMIN_PASS=$(tr -d '\n' </run/secrets/pg_admin_pass)
AUDIT_PASS=$(tr -d '\n' </run/secrets/pg_audit_pass)
CLEANUP_PASS=$(tr -d '\n' </run/secrets/pg_cleanup_pass)
POSTGRES_PASS=$(tr -d '\n' </run/secrets/pg_postgres_pass)

rm -f /run/secrets/*

need_init=false
if [[ ! -s "${PGDATA}/PG_VERSION" ]]; then
	echo "Initializing PostgreSQL database..."
	/usr/bin/initdb --encoding=UTF-8 --lc-collate=C --lc-ctype=C --pgdata="${PGDATA}" --auth-local=peer --auth-host=scram-sha-256

	cp /configs/postgresql.conf "${PGDATA}/"
	cp /configs/pg_hba.conf "${PGDATA}/"

	need_init=true
else
	echo "PostgreSQL database already initialized."
	echo "Configuring passwords."
fi

echo "[INFO] Starting PostgreSQL temporarily for migrations and password setup..."
/usr/bin/pg_ctl -D "${PGDATA}" -l "${PGDATA}/logfile" -o "-c listen_addresses=''" -w start || {
	echo "[ERROR] PostgreSQL failed to start. Log output:"
	cat "${PGDATA}/logfile" || true
	exit 1
}

if ${need_init} && [[ -d "/migrations" ]]; then
	echo "[INFO] Running initial setup migrations on postgres database..."
	for f in /migrations/001_*.sql; do
		if [[ -f ${f} ]]; then
			echo "[INFO] Applying $(basename "${f}") to postgres database"
			psql -U postgres -d postgres -v ON_ERROR_STOP=1 -f "${f}" || {
				echo "[ERROR] Initial migration failed: $(basename "${f}")"
				pg_ctl -D "${PGDATA}" -m fast -w stop
				exit 1
			}
		fi
	done

	echo "[INFO] Running remaining migrations on verification_db database..."
	for f in /migrations/002_*.sql /migrations/003_*.sql; do
		if [[ -f ${f} ]]; then
			echo "[INFO] Applying $(basename "${f}") to verification_db database"
			psql -U postgres -d verification_db -v ON_ERROR_STOP=1 -f "${f}" || {
				echo "[ERROR] Migration failed: $(basename "${f}")"
				pg_ctl -D "${PGDATA}" -m fast -w stop
				exit 1
			}
		fi
	done
fi

echo "[INFO] Updating user passwords from secrets..."
psql -U postgres -d postgres -v ON_ERROR_STOP=1 <<-EOSQL
	ALTER USER verification_api PASSWORD '${API_PASS}';
	ALTER USER verification_staff PASSWORD '${STAFF_PASS}';
	ALTER USER verification_admin PASSWORD '${ADMIN_PASS}';
	ALTER USER verification_audit PASSWORD '${AUDIT_PASS}';
	ALTER USER verification_cleanup PASSWORD '${CLEANUP_PASS}';
	ALTER USER postgres PASSWORD '${POSTGRES_PASS}';
EOSQL

echo "[INFO] Stopping temporary PostgreSQL..."
pg_ctl -D "${PGDATA}" -m fast -w stop

echo "[INFO] Database initialization complete."

echo "Starting PostgreSQL server..."
exec /usr/bin/postgres \
	-D "${PGDATA}" \
	-c hba_file="${PGDATA}/pg_hba.conf" \
	-c config_file="${PGDATA}/postgresql.conf" \
	-c ssl=on \
	-c ssl_cert_file='/var/lib/postgresql/certs/server.crt' \
	-c ssl_key_file='/var/lib/postgresql/certs/server.key'
