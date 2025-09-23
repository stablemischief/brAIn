# brAIn v2.0 - Intelligent RAG Pipeline Management System

> 🧠 **AI-Enhanced Document Processing Platform** - Transform your document vectorization workflow with intelligent monitoring, cost optimization, and real-time collaboration capabilities.

[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](./docker/) [![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?logo=fastapi)](./api/) [![React](https://img.shields.io/badge/React-Frontend-cyan?logo=react)](./frontend/) [![Supabase](https://img.shields.io/badge/Supabase-Realtime-orange?logo=supabase)](https://supabase.com)

## 🚀 What is brAIn v2.0?

brAIn v2.0 is an intelligent RAG (Retrieval-Augmented Generation) pipeline management system that transforms document processing from a CLI-only tool into a collaborative, AI-enhanced web platform. Built for teams who need reliable, cost-conscious document vectorization with real-time monitoring and intelligent automation.

### 🎯 Key Features

- **🔄 Real-time Dashboard**: Live WebSocket updates for processing status, costs, and system health
- **🤖 AI-Powered Configuration**: Pydantic AI assistant guides setup and validates configurations
- **💰 Cost Intelligence**: Token tracking, budget management, and optimization recommendations
- **🕸️ Knowledge Graph**: Automatic document relationship discovery and visualization
- **🔍 Semantic Search**: Context-aware search with hybrid vector + keyword matching
- **📊 Predictive Monitoring**: Failure prediction and automatic recovery systems
- **🛡️ Security-First**: JWT authentication, CORS protection, and comprehensive security middleware

### 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React Frontend │◄──►│  FastAPI Backend │◄──►│ PostgreSQL+pgvector│
│   (Port 3000)   │    │   (Port 8000)    │    │ Vector Database │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  WebSocket      │    │ Background       │    │  Langfuse       │
│  Real-time      │    │ Processing       │    │  LLM Monitoring │
│  Updates        │    │ Engine           │    │  & Analytics    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🎬 Quick Start

### Prerequisites
- Docker & Docker Compose
- Google Drive folder with documents to process
- OpenAI API key

### 1️⃣ Setup Environment
```bash
# Clone repository
git clone https://github.com/stablemischief/brAIn.git
cd brAIn

# Copy environment template
cp .env.example .env

# Edit configuration (see Configuration Guide)
nano .env
```

### 2️⃣ Launch System
```bash
# Start all services
docker-compose up --build

# Access the application
open http://localhost:3000
```

### 3️⃣ Configure with AI Assistant
1. Navigate to **Configuration Wizard** 📝
2. Follow the AI-guided setup process
3. Test your Google Drive connection
4. Set cost budgets and monitoring alerts

### 4️⃣ Process Documents
1. Add Google Drive folders 📁
2. Monitor real-time processing status 📊
3. Explore knowledge graph relationships 🕸️
4. Search with semantic understanding 🔍

## 📚 Documentation

### 👤 For Users
- **[User Guide](./docs/user_guide/)** - Complete setup and usage instructions
- **[Basic Deployment Guide](./docs/user_guide/deployment-basic.md)** - Step-by-step server deployment for beginners
- **[Quick Start Tutorial](./docs/user_guide/quick-start.md)** - Get running in 10 minutes
- **[Configuration Guide](./docs/user_guide/configuration.md)** - Environment setup and AI assistant

### 👩‍💻 For Developers
- **[API Documentation](./docs/api/)** - REST API and WebSocket endpoints
- **[Docker Setup](./docker/README.md)** - Development and production deployment
- **[Database Schema](./migrations/README.md)** - PostgreSQL schema and migrations

### 🎓 For Teams
- **[Onboarding Guide](./docs/onboarding/)** - Team setup and training materials
- **[Admin Guide](./docs/admin/)** - System administration and maintenance
- **[Troubleshooting](./docs/troubleshooting/)** - Common issues and solutions

### 📖 Advanced Topics
- **[Knowledge Graph Guide](./docs/tutorials/knowledge-graph.md)** - Understanding document relationships
- **[Cost Optimization](./docs/tutorials/cost-optimization.md)** - Managing AI processing costs
- **[Security Guidelines](./docs/security_guidelines.md)** - Production security best practices

## 🏢 Use Cases

### Internal Teams
- **Research Organizations**: Process academic papers and build searchable knowledge bases
- **Consulting Firms**: Analyze client documents and extract insights automatically
- **Tech Companies**: Create internal documentation search with context awareness

### Document-Heavy Workflows
- **Legal Teams**: Process contracts and case documents with relationship tracking
- **Medical Organizations**: Analyze research papers and clinical documentation
- **Financial Services**: Process reports and regulatory documents with cost control

## 🛠️ Technology Stack

### Backend
- **FastAPI** - High-performance Python web framework
- **Pydantic AI** - AI-powered data validation and configuration
- **PostgreSQL + pgvector** - Vector database for embeddings
- **Supabase** - Real-time subscriptions and authentication
- **Langfuse** - LLM operation monitoring and cost tracking

### Frontend
- **React 18** - Modern user interface framework
- **TypeScript** - Type-safe JavaScript development
- **Tailwind CSS** - Utility-first styling
- **D3.js** - Interactive knowledge graph visualization

### AI/ML
- **OpenAI Embeddings** - text-embedding-3-small model
- **Vector Similarity Search** - HNSW indexing with cosine similarity
- **Knowledge Graph Engine** - Automated relationship discovery
- **Cost Optimization** - Token counting and budget management

### Infrastructure
- **Docker** - Containerized deployment
- **Supervisor** - Multi-process management
- **nginx** - Production reverse proxy (optional)
- **Redis** - Caching and real-time data

## 📊 Performance & Scale

- **Processing Speed**: 10-30 documents per minute
- **Search Latency**: <100ms for semantic search
- **Real-time Updates**: <50ms WebSocket latency
- **Cost Efficiency**: <$0.01 per document average
- **Concurrent Users**: 10+ simultaneous users supported

## 🚨 System Requirements

### Development
- **CPU**: 2+ cores
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB available space
- **Network**: Stable internet for AI API calls

### Production
- **CPU**: 4+ cores
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB+ SSD storage
- **Database**: PostgreSQL 15+ with pgvector extension

## 🤝 Contributing

We welcome contributions! Please see our:
- [Development Setup](./docs/dev/setup.md)
- [Code Standards](./docs/dev/standards.md)
- [Testing Guide](./docs/dev/testing.md)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Full documentation](./docs/)
- **Issues**: [GitHub Issues](https://github.com/stablemischief/brAIn/issues)
- **Community**: [Discussions](https://github.com/stablemischief/brAIn/discussions)

## 🗺️ Roadmap

### Completed ✅
- Real-time dashboard with WebSocket updates
- AI-powered configuration wizard
- Cost tracking and budget management
- Knowledge graph visualization
- Comprehensive security implementation

### In Progress 🚧
- Advanced analytics dashboard
- Multi-user collaboration features
- Production deployment automation

### Planned 📋
- Plugin architecture for custom processors
- Advanced AI model switching
- Enterprise SSO integration
- Multi-tenant support

---

**Built with ❤️ using the BMad Method** - Breakthrough Method of Agile AI-driven Development