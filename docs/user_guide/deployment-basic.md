# Basic Deployment Guide - brAIn v2.0

This guide walks you through deploying brAIn v2.0 step-by-step, assuming no prior DevOps experience. Perfect for small teams or personal use.

## 🎯 What You'll Achieve

By the end of this guide, you'll have:
- ✅ brAIn v2.0 running on a server accessible from anywhere
- ✅ SSL/HTTPS security enabled
- ✅ Automatic backups configured
- ✅ Basic monitoring in place

## 📋 Before You Start

### What You Need
- **Server**: VPS or cloud server (DigitalOcean, Linode, AWS EC2, etc.)
- **Domain**: A domain name pointing to your server (e.g., brain.yourcompany.com)
- **API Keys**: OpenAI account with API access
- **Time**: About 2-3 hours for first-time setup

### Server Requirements
- **OS**: Ubuntu 20.04 or newer (this guide assumes Ubuntu)
- **CPU**: 2+ cores (4+ recommended)
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 50GB SSD minimum
- **Network**: Public IP address

## 🖥️ Step 1: Set Up Your Server

### Option A: DigitalOcean (Recommended for beginners)

1. **Create Account**: Go to [DigitalOcean](https://digitalocean.com)
2. **Create Droplet**:
   - Choose Ubuntu 22.04 LTS
   - Select "Basic" plan
   - Choose $20/month (4GB RAM, 2 CPUs) or higher
   - Add your SSH key or create a password
   - Choose a datacenter near your users

3. **Note Your Server Details**:
   ```
   Server IP: 203.0.113.123 (example)
   Username: root
   Password: (if you chose password auth)
   ```

### Option B: Other Providers
- **AWS EC2**: Choose t3.medium or larger
- **Linode**: Choose "Nanode 1GB" or larger
- **Vultr**: Choose "Regular Performance" 1GB or larger

## 🌐 Step 2: Point Your Domain to Your Server

1. **Get Your Server's IP Address** (from Step 1)
2. **Log into Your Domain Provider** (GoDaddy, Namecheap, etc.)
3. **Create DNS Records**:
   ```
   Type: A Record
   Name: @ (or your subdomain like "brain")
   Value: YOUR_SERVER_IP_ADDRESS
   TTL: 300 (5 minutes)
   ```

4. **Test DNS** (may take 5-60 minutes):
   ```bash
   # On your computer, test if DNS is working
   ping yourdomain.com
   ```

## 🔐 Step 3: Connect to Your Server

### Using Terminal/Command Prompt

**On Mac/Linux**:
```bash
ssh root@YOUR_SERVER_IP
# Example: ssh root@203.0.113.123
```

**On Windows** (use PowerShell or download PuTTY):
```powershell
ssh root@YOUR_SERVER_IP
```

**First Time Connection**:
- You'll see a security warning - type "yes" and press Enter
- Enter your password when prompted

## 🛠️ Step 4: Prepare Your Server

Copy and paste these commands one by one into your server:

### Update the System
```bash
# Update package list
apt update

# Upgrade all packages (this may take 5-10 minutes)
apt upgrade -y

# Install essential tools
apt install -y curl git unzip software-properties-common
```

### Install Docker
```bash
# Download Docker installation script
curl -fsSL https://get.docker.com -o get-docker.sh

# Run the installation script
sh get-docker.sh

# Add current user to docker group (if not root)
usermod -aG docker $USER

# Test Docker installation
docker --version
```

### Install Docker Compose
```bash
# Download Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# Make it executable
chmod +x /usr/local/bin/docker-compose

# Test installation
docker-compose --version
```

## 📁 Step 5: Get brAIn v2.0 Code

### Download the Application
```bash
# Create directory for the application
mkdir -p /opt/brain
cd /opt/brain

# Clone the repository
git clone https://github.com/stablemischief/brAIn.git .

# Verify files downloaded
ls -la
```

You should see files like:
- `docker-compose.yml`
- `Dockerfile`
- `.env.example`
- `README.md`

## ⚙️ Step 6: Configure Your Environment

### Create Configuration File
```bash
# Copy the example configuration
cp .env.example .env

# Edit the configuration file
nano .env
```

### Edit Your Configuration
In the nano editor, update these values:

**Required Settings**:
```env
# Change these values:
ENVIRONMENT=production
DEBUG=false

# Your domain (replace with your actual domain)
CORS_ORIGINS=https://yourdomain.com

# Your OpenAI API key (get from https://platform.openai.com/api-keys)
OPENAI_API_KEY=sk-your_actual_openai_key_here

# Create a strong JWT secret (30+ random characters)
JWT_SECRET=your_very_long_random_secret_string_here_abc123

# Database password (create a strong password)
POSTGRES_PASSWORD=your_strong_database_password_here
```

**Save and Exit**:
- Press `Ctrl + X`
- Press `Y` to confirm
- Press `Enter` to save

### Verify Your Configuration
```bash
# Check that your file looks correct (without showing sensitive data)
grep -v "API_KEY\|PASSWORD\|SECRET" .env
```

## 🚀 Step 7: Launch brAIn v2.0

### Start the Application
```bash
# Navigate to the application directory
cd /opt/brain

# Start all services (this will take 5-10 minutes the first time)
docker-compose up -d

# Watch the startup process
docker-compose logs -f
```

**What's Happening**:
- Docker is downloading and building the application
- Database is being created and configured
- Web services are starting up

**Look for Success Messages**:
- "Application startup complete"
- "Server started on port 8000"
- "Database migration completed"

### Test Basic Functionality
```bash
# Test that the application is responding
curl http://localhost:8000/api/health

# You should see something like:
# {"status": "healthy", "timestamp": "2025-09-22T..."}
```

## 🔒 Step 8: Set Up SSL/HTTPS

### Install SSL Certificate Tool
```bash
# Install Certbot for SSL certificates
apt install -y snapd
snap install --classic certbot
ln -s /snap/bin/certbot /usr/bin/certbot
```

### Get SSL Certificate
```bash
# Stop the application temporarily
docker-compose down

# Get SSL certificate (replace yourdomain.com with your actual domain)
certbot certonly --standalone -d yourdomain.com

# Follow the prompts:
# - Enter your email address
# - Agree to terms (type 'y')
# - Choose whether to share email (type 'y' or 'n')
```

### Configure nginx for SSL
```bash
# Create nginx configuration
mkdir -p /opt/brain/nginx
cat > /opt/brain/nginx/nginx.conf << 'EOF'
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://brain-app:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api/ {
        proxy_pass http://brain-app:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws/ {
        proxy_pass http://brain-app:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
    }
}
EOF

# Replace yourdomain.com with your actual domain in the config
sed -i 's/yourdomain.com/YOUR_ACTUAL_DOMAIN/g' /opt/brain/nginx/nginx.conf
```

### Update Docker Compose for SSL
```bash
# Create production docker-compose file
cat > /opt/brain/docker-compose.prod.yml << 'EOF'
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/conf.d/default.conf
      - /etc/letsencrypt:/etc/letsencrypt:ro
    depends_on:
      - brain-app
    restart: unless-stopped

  brain-app:
    build: .
    environment:
      - ENVIRONMENT=production
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: brain_db
      POSTGRES_USER: brain_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

volumes:
  postgres_data:
EOF
```

### Start with SSL
```bash
# Start the production version
docker-compose -f docker-compose.prod.yml up -d

# Check that everything is running
docker-compose -f docker-compose.prod.yml ps
```

## 🧪 Step 9: Test Your Deployment

### Test HTTP to HTTPS Redirect
```bash
# Test that HTTP redirects to HTTPS
curl -I http://yourdomain.com
# Should show: Location: https://yourdomain.com/
```

### Test HTTPS
```bash
# Test HTTPS is working
curl -I https://yourdomain.com
# Should show: HTTP/2 200
```

### Test in Browser
1. Open your browser
2. Go to `https://yourdomain.com`
3. You should see the brAIn v2.0 dashboard!

## 🔄 Step 10: Set Up Automatic SSL Renewal

### Create Renewal Script
```bash
# Create renewal script
cat > /opt/brain/renew-ssl.sh << 'EOF'
#!/bin/bash
# Renew SSL certificates

# Stop nginx to free port 80
docker-compose -f /opt/brain/docker-compose.prod.yml stop nginx

# Renew certificates
certbot renew --quiet

# Start nginx again
docker-compose -f /opt/brain/docker-compose.prod.yml start nginx

echo "SSL renewal completed at $(date)"
EOF

# Make script executable
chmod +x /opt/brain/renew-ssl.sh
```

### Schedule Automatic Renewal
```bash
# Add to crontab to run monthly
(crontab -l 2>/dev/null; echo "0 3 1 * * /opt/brain/renew-ssl.sh >> /opt/brain/ssl-renewal.log 2>&1") | crontab -

# Verify crontab was added
crontab -l
```

## 💾 Step 11: Set Up Basic Backups

### Create Backup Script
```bash
# Create backup directory
mkdir -p /opt/brain/backups

# Create backup script
cat > /opt/brain/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/brain/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup database
docker-compose -f /opt/brain/docker-compose.prod.yml exec -T postgres pg_dump -U brain_user brain_db > "${BACKUP_DIR}/database_${DATE}.sql"

# Backup configuration
cp /opt/brain/.env "${BACKUP_DIR}/env_${DATE}.backup"

# Compress old backups
find ${BACKUP_DIR} -name "*.sql" -mtime +1 -exec gzip {} \;

# Delete backups older than 30 days
find ${BACKUP_DIR} -name "*.gz" -mtime +30 -delete

echo "Backup completed: ${DATE}"
EOF

# Make script executable
chmod +x /opt/brain/backup.sh

# Test backup script
./backup.sh
```

### Schedule Daily Backups
```bash
# Add to crontab for daily backups at 2 AM
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/brain/backup.sh >> /opt/brain/backup.log 2>&1") | crontab -
```

## 📊 Step 12: Basic Monitoring Setup

### Create Health Check Script
```bash
cat > /opt/brain/health-check.sh << 'EOF'
#!/bin/bash
# Simple health monitoring

URL="https://yourdomain.com/api/health"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$URL")

if [ "$STATUS" = "200" ]; then
    echo "$(date): brAIn v2.0 is healthy"
else
    echo "$(date): WARNING - brAIn v2.0 health check failed (HTTP $STATUS)"
    # Restart services if unhealthy
    docker-compose -f /opt/brain/docker-compose.prod.yml restart
fi
EOF

chmod +x /opt/brain/health-check.sh

# Add to crontab for every 5 minutes
(crontab -l 2>/dev/null; echo "*/5 * * * * /opt/brain/health-check.sh >> /opt/brain/health.log 2>&1") | crontab -
```

## ✅ Step 13: Final Verification

### Complete Deployment Checklist
- [ ] Server is accessible via HTTPS
- [ ] SSL certificate is working (no browser warnings)
- [ ] Dashboard loads and shows login screen
- [ ] Can create user account through interface
- [ ] Health check endpoint returns 200 OK
- [ ] Backups are configured and tested
- [ ] SSL renewal is scheduled
- [ ] Basic monitoring is active

### Test Complete Workflow
1. **Access Dashboard**: Go to `https://yourdomain.com`
2. **Create Account**: Use the signup process
3. **Run Configuration Wizard**: Complete the AI-assisted setup
4. **Add Test Folder**: Connect a Google Drive folder
5. **Process Documents**: Verify document processing works
6. **Test Search**: Try semantic search functionality

## 🎉 Congratulations!

Your brAIn v2.0 deployment is complete! Here's what you now have:

### ✅ Production Features
- **Secure HTTPS access** with automatic SSL renewal
- **Automatic daily backups** with 30-day retention
- **Basic health monitoring** with auto-restart
- **Production-optimized configuration**

### 🔗 Important URLs
- **Dashboard**: `https://yourdomain.com`
- **API Documentation**: `https://yourdomain.com/docs`
- **Health Check**: `https://yourdomain.com/api/health`

### 📁 Important Files on Your Server
- **Application**: `/opt/brain/`
- **Configuration**: `/opt/brain/.env`
- **Backups**: `/opt/brain/backups/`
- **Logs**: `/opt/brain/logs/`

## 🆘 Common Issues & Quick Fixes

### "Site Can't Be Reached"
```bash
# Check if services are running
docker-compose -f /opt/brain/docker-compose.prod.yml ps

# Restart if needed
docker-compose -f /opt/brain/docker-compose.prod.yml restart
```

### SSL Certificate Errors
```bash
# Check certificate status
certbot certificates

# Renew if needed
./renew-ssl.sh
```

### Application Not Starting
```bash
# Check logs
docker-compose -f /opt/brain/docker-compose.prod.yml logs

# Common fix: restart everything
docker-compose -f /opt/brain/docker-compose.prod.yml down
docker-compose -f /opt/brain/docker-compose.prod.yml up -d
```

### Need to Update Configuration
```bash
# Edit configuration
nano /opt/brain/.env

# Restart to apply changes
docker-compose -f /opt/brain/docker-compose.prod.yml restart
```

## 📞 Getting Help

- **Check Logs**: `docker-compose -f /opt/brain/docker-compose.prod.yml logs`
- **Health Status**: Visit `https://yourdomain.com/api/health`
- **Documentation**: See the [Troubleshooting Guide](../troubleshooting/)

---

**Your brAIn v2.0 deployment is now live and ready for your team!** 🚀