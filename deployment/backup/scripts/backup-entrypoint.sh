#!/bin/bash
# Automated PostgreSQL Backup Script for brAIn v2.0
# Runs daily backups with rotation and S3 upload

set -e

# Configuration
BACKUP_DIR="/backup/postgres"
DB_HOST="${DB_HOST:-postgres}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-brain_prod}"
DB_USER="${DB_USER:-brain_user}"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"
S3_BUCKET="${S3_BACKUP_BUCKET}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Create backup directory if it doesn't exist
mkdir -p "${BACKUP_DIR}"

# Function to perform backup
perform_backup() {
    local backup_file="${BACKUP_DIR}/backup_${DB_NAME}_${TIMESTAMP}.sql.gz"
    local backup_info="${BACKUP_DIR}/backup_${DB_NAME}_${TIMESTAMP}.info"

    log "Starting backup of database: ${DB_NAME}"

    # Check database connectivity
    if ! PGPASSWORD="${POSTGRES_PASSWORD}" pg_isready -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}"; then
        error "Failed to connect to database"
        return 1
    fi

    # Perform the backup
    if PGPASSWORD="${POSTGRES_PASSWORD}" pg_dump \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${DB_NAME}" \
        --verbose \
        --no-owner \
        --no-privileges \
        --clean \
        --if-exists \
        --format=plain | gzip -9 > "${backup_file}"; then

        # Create backup info file
        cat > "${backup_info}" <<EOF
Backup Information
==================
Database: ${DB_NAME}
Host: ${DB_HOST}
Port: ${DB_PORT}
User: ${DB_USER}
Timestamp: ${TIMESTAMP}
File: $(basename ${backup_file})
Size: $(du -h ${backup_file} | cut -f1)
MD5: $(md5sum ${backup_file} | cut -d' ' -f1)
EOF

        log "Backup completed successfully: ${backup_file}"
        log "Backup size: $(du -h ${backup_file} | cut -f1)"

        # Upload to S3 if configured
        if [ -n "${S3_BUCKET}" ] && command -v aws &> /dev/null; then
            upload_to_s3 "${backup_file}" "${backup_info}"
        fi

        return 0
    else
        error "Backup failed"
        return 1
    fi
}

# Function to upload backup to S3
upload_to_s3() {
    local backup_file="$1"
    local backup_info="$2"

    log "Uploading backup to S3: ${S3_BUCKET}"

    if aws s3 cp "${backup_file}" "s3://${S3_BUCKET}/postgres/$(basename ${backup_file})" \
        --storage-class STANDARD_IA \
        --metadata "backup-date=${TIMESTAMP},database=${DB_NAME}"; then

        aws s3 cp "${backup_info}" "s3://${S3_BUCKET}/postgres/$(basename ${backup_info})"
        log "Successfully uploaded backup to S3"
    else
        warning "Failed to upload backup to S3"
    fi
}

# Function to clean old backups
cleanup_old_backups() {
    log "Cleaning up old backups (retention: ${RETENTION_DAYS} days)"

    # Local cleanup
    find "${BACKUP_DIR}" -name "backup_${DB_NAME}_*.sql.gz" -type f -mtime +${RETENTION_DAYS} -delete
    find "${BACKUP_DIR}" -name "backup_${DB_NAME}_*.info" -type f -mtime +${RETENTION_DAYS} -delete

    # S3 cleanup if configured
    if [ -n "${S3_BUCKET}" ] && command -v aws &> /dev/null; then
        log "Cleaning up old S3 backups"
        aws s3 ls "s3://${S3_BUCKET}/postgres/" | while read -r line; do
            createDate=$(echo $line | awk '{print $1" "$2}')
            createDate=$(date -d "$createDate" +%s)
            olderThan=$(date -d "${RETENTION_DAYS} days ago" +%s)

            if [[ $createDate -lt $olderThan ]]; then
                fileName=$(echo $line | awk '{print $4}')
                if [[ $fileName == backup_${DB_NAME}_* ]]; then
                    aws s3 rm "s3://${S3_BUCKET}/postgres/$fileName"
                    log "Deleted old S3 backup: $fileName"
                fi
            fi
        done
    fi
}

# Function to verify backup
verify_backup() {
    local latest_backup=$(ls -t "${BACKUP_DIR}"/backup_${DB_NAME}_*.sql.gz 2>/dev/null | head -1)

    if [ -f "${latest_backup}" ]; then
        log "Verifying backup: ${latest_backup}"

        # Test if the file can be decompressed
        if gzip -t "${latest_backup}" 2>/dev/null; then
            log "Backup verification successful"
            return 0
        else
            error "Backup verification failed - corrupted file"
            return 1
        fi
    else
        error "No backup file found to verify"
        return 1
    fi
}

# Main backup loop
main() {
    log "PostgreSQL Backup Service Started"
    log "Database: ${DB_NAME}"
    log "Host: ${DB_HOST}:${DB_PORT}"
    log "Retention: ${RETENTION_DAYS} days"

    if [ -n "${S3_BUCKET}" ]; then
        log "S3 Backup: Enabled (${S3_BUCKET})"
    else
        log "S3 Backup: Disabled"
    fi

    # Run initial backup
    perform_backup
    verify_backup
    cleanup_old_backups

    # Schedule daily backups
    while true; do
        # Sleep until 3 AM
        current_hour=$(date +%H)
        if [ "$current_hour" -ge 3 ]; then
            # Past 3 AM today, wait until tomorrow
            sleep_seconds=$(($(date -d "tomorrow 03:00:00" +%s) - $(date +%s)))
        else
            # Before 3 AM today
            sleep_seconds=$(($(date -d "today 03:00:00" +%s) - $(date +%s)))
        fi

        log "Next backup scheduled in ${sleep_seconds} seconds"
        sleep ${sleep_seconds}

        # Perform daily backup
        log "Starting scheduled daily backup"
        perform_backup
        verify_backup
        cleanup_old_backups
    done
}

# Trap signals for graceful shutdown
trap 'log "Backup service shutting down"; exit 0' SIGTERM SIGINT

# Start the backup service
main