# Quick Start Guide - brAIn v2.0

Get brAIn v2.0 running in 10 minutes! This guide takes you from zero to processing your first documents.

## 🎯 What You'll Accomplish

By the end of this guide, you'll have:
- ✅ brAIn v2.0 running locally with Docker
- ✅ A Google Drive folder connected and processing
- ✅ Real-time dashboard showing processing status
- ✅ Your first semantic search results

## 📋 Prerequisites Checklist

Before starting, ensure you have:
- [ ] Docker and Docker Compose installed
- [ ] Google Drive account with test documents (5-10 files recommended)
- [ ] OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- [ ] Text editor for configuration files

## 🚀 Step 1: Get the Code

```bash
# Clone the repository
git clone https://github.com/stablemischief/brain-rag-v2
cd brain-rag-v2

# Check Docker is running
docker --version
docker-compose --version
```

## ⚙️ Step 2: Basic Configuration

```bash
# Copy the environment template
cp .env.example .env

# Edit with your favorite editor
nano .env  # or code .env, or vim .env
```

**Essential settings to configure:**
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Basic Setup
ENVIRONMENT=development
DEBUG=true

# Database (defaults work for local development)
DATABASE_URL=postgresql://brain_user:brain_password@postgres:5432/brain_db

# Google Drive (we'll configure this in the UI)
GOOGLE_DRIVE_FOLDER_ID=  # Leave empty for now
```

💡 **Pro tip**: You can configure Google Drive through the AI assistant after starting the system.

## 🐳 Step 3: Launch the System

```bash
# Start all services (this will take 2-3 minutes the first time)
docker-compose up --build

# You'll see logs from multiple services starting up
# Wait for: "Application startup complete" message
```

**What's happening:**
- 🐘 PostgreSQL database starting with pgvector extension
- 🐍 Python backend building with AI dependencies
- ⚛️ React frontend building with TypeScript
- 🔧 Supervisor managing multiple processes

## 🌐 Step 4: Access the Dashboard

Open your browser and navigate to:
- **Main App**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

You should see the brAIn v2.0 dashboard loading! 🎉

## 🤖 Step 5: Run the AI Configuration Wizard

1. **Click "Configuration Wizard"** in the dashboard
2. **Follow the AI assistant prompts**:
   - Verify your OpenAI API key
   - Test database connection
   - Configure Google Drive access
   - Set daily cost budgets

3. **Google Drive Setup**:
   - The wizard will guide you through OAuth setup
   - Select a test folder with 5-10 documents
   - Choose your processing preferences

## 📊 Step 6: Process Your First Documents

1. **Go to "Folder Management"**
2. **Add your test folder**:
   - Paste the Google Drive folder URL
   - Click "Add Folder"
   - Watch real-time processing begin!

3. **Monitor Progress**:
   - Processing status updates live
   - Cost tracking in real-time
   - Success/failure notifications

## 🔍 Step 7: Try Semantic Search

Once processing completes (usually 1-2 minutes for 5-10 documents):

1. **Navigate to "Search"**
2. **Try these example queries**:
   - "project timeline" (finds schedule-related content)
   - "budget costs" (finds financial information)
   - "team responsibilities" (finds role assignments)

3. **Explore Results**:
   - See semantic similarity scores
   - Click through to original documents
   - Notice context-aware ranking

## 🕸️ Step 8: Explore Knowledge Relationships

1. **Click "Knowledge Graph"**
2. **Explore the visualization**:
   - Nodes represent concepts and entities
   - Edges show relationships between documents
   - Zoom and pan to explore connections

3. **Try Interactive Features**:
   - Click nodes to see related documents
   - Filter by relationship types
   - Search for specific entities

## 🎯 Verification Checklist

Ensure everything is working:
- [ ] Dashboard loads and shows real-time updates
- [ ] Documents processed successfully (green status)
- [ ] Search returns relevant results
- [ ] Knowledge graph shows document relationships
- [ ] Cost tracking displays usage information

## 🎉 Congratulations!

You've successfully set up brAIn v2.0! Here's what you can do next:

### Immediate Next Steps
- **Add more folders**: Scale up to your full document collection
- **Adjust budgets**: Set appropriate daily/monthly limits
- **Explore search**: Try different query types and filters
- **Review costs**: Understand your usage patterns

### Learn More
- **[Dashboard Guide](./dashboard.md)** - Master the real-time interface
- **[Document Processing](./document-processing.md)** - Advanced processing options
- **[Cost Management](./cost-management.md)** - Optimize your AI spending

## 🆘 Troubleshooting Quick Fixes

### Container won't start?
```bash
# Check logs
docker-compose logs brain-app

# Reset and rebuild
docker-compose down -v
docker-compose up --build
```

### API key not working?
1. Verify key in .env file
2. Check OpenAI account has credits
3. Restart services: `docker-compose restart`

### Google Drive connection failed?
1. Use the Configuration Wizard to re-authenticate
2. Ensure folder is publicly accessible or properly shared
3. Check folder URL format

### No search results?
1. Verify documents processed successfully (green status)
2. Wait for indexing to complete (usually 30 seconds)
3. Try simpler search terms first

## 💡 Pro Tips

1. **Start with test data**: Use a small folder first to understand costs
2. **Monitor the logs**: `docker-compose logs -f brain-app` shows detailed info
3. **Use the AI assistant**: It can help diagnose most configuration issues
4. **Set budgets early**: Prevent unexpected costs during experimentation

---

**Next**: Ready to dive deeper? Check out the [Dashboard Guide](./dashboard.md) to master the interface!