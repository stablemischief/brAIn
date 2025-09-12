# brAIn Monitoring Architecture

## AI-First Observability & Intelligence System

### Version: 2.0 Enhanced
### Focus: Real-time, Predictive, Cost-Aware Monitoring

---

## 🎯 Overview

The brAIn monitoring architecture implements a comprehensive observability system that goes beyond traditional metrics to provide AI-powered insights, predictive analytics, and cost optimization. This system leverages Langfuse for LLM operations, Supabase real-time for instant updates, and custom intelligence for predictive monitoring.

### Core Principles
- **Real-time Visibility:** Instant status updates via WebSocket subscriptions
- **Predictive Analytics:** Failure prediction and performance optimization
- **Cost Intelligence:** Token tracking, budget management, and optimization
- **Context Awareness:** Knowledge graph integration for intelligent insights
- **Self-Healing:** Automatic recovery and optimization recommendations

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                 Monitoring Dashboard                     │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────┐│
│  │  System Health  │ │  Cost Analytics │ │ Knowledge   ││
│  │  Real-time      │ │  Predictions    │ │ Graph Viz   ││
│  │  Indicators     │ │  Optimization   │ │ Insights    ││
│  └─────────────────┘ └─────────────────┘ └─────────────┘│
└─────────────────────────────────────────────────────────┘
                           │ WebSocket
┌─────────────────────────────────────────────────────────┐
│                Monitoring Engine                         │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────┐│
│  │  Real-time      │ │  Predictive     │ │ Knowledge   ││
│  │  Collector      │ │  Analytics      │ │ Graph       ││
│  │                 │ │  Engine         │ │ Monitor     ││
│  └─────────────────┘ └─────────────────┘ └─────────────┘│
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────┐│
│  │  Cost           │ │  LLM            │ │ System      ││
│  │  Optimizer      │ │  Tracker        │ │ Health      ││
│  │                 │ │  (Langfuse)     │ │ Monitor     ││
│  └─────────────────┘ └─────────────────┘ └─────────────┘│
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                   Data Layer                            │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────┐│
│  │   Supabase      │ │    Langfuse     │ │   Custom    ││
│  │   Real-time     │ │    Traces       │ │   Metrics   ││
│  │   Tables        │ │    & Spans      │ │   Store     ││
│  └─────────────────┘ └─────────────────┘ └─────────────┘│
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Monitoring Components

### 1. Real-time Data Collector

**Purpose:** Collect and aggregate monitoring data in real-time

**Implementation:**
```python
class RealtimeCollector:
    """Real-time monitoring data collection"""
    
    def __init__(self):
        self.supabase = create_client(...)
        self.langfuse = Langfuse(...)
        self.metrics_buffer = {}
        
    async def collect_system_metrics(self):
        """Collect system health metrics"""
        metrics = {
            'timestamp': datetime.now(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_mb': psutil.virtual_memory().used // 1024 // 1024,
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'active_connections': len(self.active_websockets),
            'queue_size': await self.get_processing_queue_size()
        }
        
        await self.store_metrics('system_health', metrics)
        await self.broadcast_metrics(metrics)
    
    async def collect_processing_metrics(self, file_operation):
        """Collect processing-specific metrics"""
        metrics = {
            'file_id': file_operation.file_id,
            'operation_type': file_operation.type,
            'start_time': file_operation.start_time,
            'duration_ms': file_operation.duration_ms,
            'tokens_used': file_operation.tokens_used,
            'cost': file_operation.cost,
            'success': file_operation.success,
            'error_type': file_operation.error_type if not file_operation.success else None
        }
        
        await self.store_metrics('processing_metrics', metrics)
        await self.update_real_time_dashboard(metrics)
```

### 2. LLM Operations Tracking (Langfuse)

**Purpose:** Comprehensive tracking of all LLM operations with cost analysis

**Integration:**
```python
from langfuse import Langfuse
from langfuse.decorators import observe

class LLMTracker:
    """Enhanced LLM operation tracking"""
    
    def __init__(self):
        self.langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )
        
    @observe(name="document-embedding")
    async def track_embedding_operation(self, document_data, chunks, embeddings):
        """Track embedding generation with detailed metrics"""
        
        # Calculate costs
        total_tokens = sum(self.count_tokens(chunk) for chunk in chunks)
        total_cost = self.calculate_embedding_cost(total_tokens, self.model)
        
        # Create comprehensive trace
        trace = self.langfuse.trace(
            name="document-embedding",
            input={
                "document_id": document_data['id'],
                "document_name": document_data['name'],
                "document_size": len(document_data['content']),
                "chunk_count": len(chunks)
            },
            output={
                "embeddings_generated": len(embeddings),
                "total_tokens": total_tokens,
                "total_cost": float(total_cost),
                "avg_tokens_per_chunk": total_tokens / len(chunks) if chunks else 0
            },
            metadata={
                "model": self.model,
                "folder_id": document_data.get('folder_id'),
                "file_type": document_data.get('file_type')
            }
        )
        
        # Track individual chunk processing
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            generation = trace.generation(
                name=f"chunk-embedding-{i}",
                model=self.model,
                input=chunk[:100],  # First 100 chars for reference
                output={"embedding_dimensions": len(embedding)},
                usage={
                    "input_tokens": self.count_tokens(chunk),
                    "total_tokens": self.count_tokens(chunk),
                    "unit": "TOKENS"
                },
                metadata={
                    "chunk_index": i,
                    "chunk_size": len(chunk)
                }
            )
        
        # Store in local database for analytics
        await self.store_llm_usage({
            "trace_id": trace.id,
            "document_id": document_data['id'],
            "operation_type": "embedding",
            "model_name": self.model,
            "input_tokens": total_tokens,
            "output_tokens": 0,
            "cost": total_cost,
            "latency_ms": trace.duration_ms if hasattr(trace, 'duration_ms') else None
        })
        
        return trace.id
    
    async def track_search_operation(self, query, results, search_strategy):
        """Track semantic search operations"""
        
        with self.langfuse.trace(name="semantic-search") as trace:
            trace.update(
                input={"query": query, "strategy": search_strategy},
                output={
                    "result_count": len(results),
                    "avg_similarity": sum(r.similarity_score for r in results) / len(results) if results else 0,
                    "top_similarity": max(r.similarity_score for r in results) if results else 0
                },
                metadata={
                    "search_strategy": search_strategy,
                    "has_results": len(results) > 0
                }
            )
```

### 3. Predictive Analytics Engine

**Purpose:** AI-powered prediction of failures, capacity needs, and optimization opportunities

**Implementation:**
```python
class PredictiveAnalytics:
    """AI-powered predictive monitoring"""
    
    def __init__(self):
        self.failure_patterns = {}
        self.performance_baseline = {}
        self.cost_trends = {}
        
    async def analyze_failure_patterns(self):
        """Analyze historical failures to predict future issues"""
        
        # Get recent failure data
        failures = await self.get_recent_failures(days=30)
        
        # Pattern analysis
        patterns = {
            'time_based': self.analyze_time_patterns(failures),
            'file_type_based': self.analyze_file_type_patterns(failures),
            'size_based': self.analyze_size_patterns(failures),
            'folder_based': self.analyze_folder_patterns(failures)
        }
        
        # Predict likely failures
        predictions = await self.generate_failure_predictions(patterns)
        
        # Create alerts for high-risk scenarios
        for prediction in predictions:
            if prediction['probability'] > 0.7:
                await self.create_predictive_alert(prediction)
        
        return predictions
    
    async def analyze_performance_trends(self):
        """Analyze performance trends and predict capacity needs"""
        
        # Get performance metrics
        metrics = await self.get_performance_metrics(days=14)
        
        # Trend analysis
        trends = {
            'processing_time': self.calculate_trend(metrics, 'avg_processing_time'),
            'throughput': self.calculate_trend(metrics, 'files_per_hour'),
            'error_rate': self.calculate_trend(metrics, 'error_rate'),
            'cost_per_document': self.calculate_trend(metrics, 'cost_per_document')
        }
        
        # Capacity predictions
        capacity_forecast = await self.predict_capacity_needs(trends)
        
        # Performance recommendations
        recommendations = await self.generate_performance_recommendations(trends)
        
        return {
            'trends': trends,
            'capacity_forecast': capacity_forecast,
            'recommendations': recommendations
        }
    
    async def cost_optimization_analysis(self):
        """Analyze cost patterns and suggest optimizations"""
        
        # Get cost data
        cost_data = await self.get_cost_metrics(days=30)
        
        # Analyze spending patterns
        analysis = {
            'daily_trends': self.analyze_daily_spending(cost_data),
            'model_efficiency': self.analyze_model_efficiency(cost_data),
            'folder_cost_analysis': self.analyze_folder_costs(cost_data),
            'optimization_opportunities': await self.identify_optimizations(cost_data)
        }
        
        # Generate savings recommendations
        recommendations = await self.generate_cost_recommendations(analysis)
        
        return {
            'analysis': analysis,
            'recommendations': recommendations,
            'projected_savings': self.calculate_potential_savings(recommendations)
        }
```

### 4. Real-time Dashboard Backend

**Purpose:** WebSocket-based real-time updates for monitoring dashboard

**Implementation:**
```python
class MonitoringWebSocket:
    """Real-time monitoring WebSocket handler"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscription_manager = SupabaseSubscriptionManager()
        
    async def connect(self, websocket: WebSocket):
        """Accept new monitoring connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Send current status immediately
        current_status = await self.get_current_system_status()
        await websocket.send_json({
            'type': 'initial_status',
            'data': current_status
        })
        
        # Set up real-time subscriptions
        await self.setup_monitoring_subscriptions(websocket)
    
    async def setup_monitoring_subscriptions(self, websocket: WebSocket):
        """Set up Supabase subscriptions for monitoring data"""
        
        # System health updates
        health_channel = self.supabase.channel('system-health-monitor')
        health_channel.on_postgres_changes(
            event='*',
            schema='public',
            table='system_health',
            callback=lambda payload: asyncio.create_task(
                self.broadcast_health_update(websocket, payload)
            )
        ).subscribe()
        
        # Processing status updates
        processing_channel = self.supabase.channel('processing-monitor')
        processing_channel.on_postgres_changes(
            event='*',
            schema='public',
            table='processing_logs',
            callback=lambda payload: asyncio.create_task(
                self.broadcast_processing_update(websocket, payload)
            )
        ).subscribe()
        
        # Cost tracking updates
        cost_channel = self.supabase.channel('cost-monitor')
        cost_channel.on_postgres_changes(
            event='INSERT',
            schema='public',
            table='llm_usage',
            callback=lambda payload: asyncio.create_task(
                self.broadcast_cost_update(websocket, payload)
            )
        ).subscribe()
    
    async def broadcast_health_update(self, websocket: WebSocket, payload):
        """Broadcast system health updates"""
        health_data = payload.get('new', payload)
        
        # Enrich with predictions
        enriched_data = await self.enrich_health_data(health_data)
        
        message = {
            'type': 'health_update',
            'data': enriched_data,
            'timestamp': datetime.now().isoformat()
        }
        
        await self.safe_send(websocket, message)
    
    async def broadcast_cost_update(self, websocket: WebSocket, payload):
        """Broadcast cost updates with budget analysis"""
        cost_data = payload.get('new', payload)
        
        # Calculate running totals and projections
        daily_total = await self.get_daily_cost_total()
        monthly_projection = await self.calculate_monthly_projection()
        
        # Check budget limits
        budget_status = await self.check_budget_status(daily_total)
        
        message = {
            'type': 'cost_update',
            'data': {
                'latest_operation': cost_data,
                'daily_total': float(daily_total),
                'monthly_projection': float(monthly_projection),
                'budget_status': budget_status
            },
            'timestamp': datetime.now().isoformat()
        }
        
        await self.safe_send(websocket, message)
        
        # Send alert if budget exceeded
        if budget_status['status'] == 'exceeded':
            await self.send_budget_alert(websocket, budget_status)
```

### 5. Knowledge Graph Monitor

**Purpose:** Monitor and analyze document relationships and knowledge patterns

**Implementation:**
```python
class KnowledgeGraphMonitor:
    """Monitor knowledge graph health and insights"""
    
    async def analyze_graph_metrics(self):
        """Analyze knowledge graph metrics"""
        
        metrics = {
            'total_nodes': await self.count_knowledge_nodes(),
            'total_edges': await self.count_knowledge_edges(),
            'connected_components': await self.analyze_connected_components(),
            'average_degree': await self.calculate_average_node_degree(),
            'clustering_coefficient': await self.calculate_clustering(),
            'graph_density': await self.calculate_graph_density()
        }
        
        # Analyze graph health
        health_score = await self.calculate_graph_health_score(metrics)
        
        # Identify insights
        insights = await self.extract_graph_insights(metrics)
        
        return {
            'metrics': metrics,
            'health_score': health_score,
            'insights': insights,
            'recommendations': await self.generate_graph_recommendations(metrics)
        }
    
    async def monitor_relationship_quality(self):
        """Monitor the quality of detected relationships"""
        
        # Sample recent relationships
        recent_edges = await self.get_recent_edges(limit=100)
        
        # Quality assessment
        quality_metrics = {
            'confidence_distribution': self.analyze_confidence_distribution(recent_edges),
            'relationship_type_diversity': self.analyze_relationship_types(recent_edges),
            'temporal_consistency': await self.check_temporal_consistency(recent_edges)
        }
        
        return quality_metrics
```

---

## 📈 Monitoring Dashboards

### 1. System Health Dashboard

**Real-time Indicators:**
- **Processing Status:** Current operations, queue size, throughput
- **System Resources:** CPU, memory, disk usage with trends
- **Service Health:** API response times, database connections, external service status
- **Error Rates:** Real-time error tracking with categorization

**Predictive Elements:**
- **Failure Probability:** AI-predicted failure likelihood for next 24 hours
- **Capacity Forecast:** Predicted resource needs based on trends
- **Performance Degradation Alerts:** Early warning for performance issues

### 2. Cost Analytics Dashboard

**Cost Tracking:**
- **Real-time Spend:** Current daily/monthly costs with budget comparison
- **Token Usage:** Detailed breakdown by operation, model, and folder
- **Cost per Document:** Trends and efficiency metrics
- **Budget Alerts:** Visual indicators for budget status

**Optimization Insights:**
- **Model Efficiency:** Comparison of different embedding models
- **Batch Processing Benefits:** Cost savings from optimization
- **Unused Resources:** Identification of waste and inefficiencies

### 3. Knowledge Graph Insights

**Graph Metrics:**
- **Relationship Discovery:** New connections found over time
- **Graph Growth:** Node and edge creation trends
- **Connectivity Analysis:** Highly connected documents and clusters
- **Quality Metrics:** Relationship confidence and accuracy

**Business Insights:**
- **Knowledge Patterns:** Most common document relationships
- **Content Gaps:** Areas with low connectivity
- **Recommendation Engine:** Suggested documents for processing

### 4. Performance Analytics

**Processing Metrics:**
- **Throughput Trends:** Documents processed per hour/day
- **Processing Time Distribution:** Performance by file type and size
- **Success/Failure Rates:** Detailed breakdown with root cause analysis
- **Queue Analysis:** Processing backlog and wait times

**Optimization Opportunities:**
- **Bottleneck Identification:** Slowest processing stages
- **Resource Utilization:** Efficiency metrics and recommendations
- **Scaling Recommendations:** When and how to scale resources

---

## 🔔 Alerting System

### Alert Categories

#### 1. Critical System Alerts
- **Service Down:** Any core service unavailable
- **High Error Rate:** Error rate >5% for 5+ minutes
- **Resource Exhaustion:** CPU >90% or memory >95% for 10+ minutes
- **Database Issues:** Connection failures or slow queries

#### 2. Cost Management Alerts
- **Budget Exceeded:** Daily or monthly budget limits reached
- **Cost Spike:** Unusual spending patterns detected
- **Quota Warnings:** Approaching API limits
- **Inefficient Operations:** High cost per document ratios

#### 3. Performance Alerts
- **Slow Processing:** Processing time >2x baseline
- **Queue Backlog:** Processing queue >100 items
- **Failed Operations:** Multiple consecutive failures
- **Degraded Search:** Search performance below threshold

#### 4. Predictive Alerts
- **Predicted Failure:** AI predicts failure within 24 hours
- **Capacity Warning:** Resource needs forecast to exceed capacity
- **Maintenance Required:** Predictive maintenance recommendations

### Alert Implementation

```python
class AlertManager:
    """Intelligent alert management system"""
    
    async def evaluate_alerts(self, metrics: Dict[str, Any]):
        """Evaluate metrics and trigger appropriate alerts"""
        
        alerts = []
        
        # System health alerts
        if metrics['error_rate'] > 0.05:
            alerts.append(await self.create_alert(
                type='high_error_rate',
                severity='critical',
                message=f"Error rate {metrics['error_rate']:.1%} exceeds threshold",
                data=metrics
            ))
        
        # Cost alerts
        if metrics['daily_cost'] > float(os.getenv('DAILY_BUDGET', '10.0')):
            alerts.append(await self.create_alert(
                type='budget_exceeded',
                severity='high',
                message=f"Daily cost ${metrics['daily_cost']:.2f} exceeds budget",
                data=metrics
            ))
        
        # Predictive alerts
        failure_probability = await self.get_failure_probability()
        if failure_probability > 0.7:
            alerts.append(await self.create_alert(
                type='predicted_failure',
                severity='medium',
                message=f"High failure probability: {failure_probability:.1%}",
                data={'probability': failure_probability}
            ))
        
        # Send alerts
        for alert in alerts:
            await self.send_alert(alert)
        
        return alerts
    
    async def send_alert(self, alert: Dict[str, Any]):
        """Send alert through multiple channels"""
        
        # Real-time dashboard
        await self.broadcast_alert_to_dashboard(alert)
        
        # Email for critical alerts
        if alert['severity'] in ['critical', 'high']:
            await self.send_email_alert(alert)
        
        # Slack/Discord webhook
        if os.getenv('SLACK_WEBHOOK_URL'):
            await self.send_slack_alert(alert)
        
        # Log alert
        logger.warning(f"Alert triggered: {alert['type']} - {alert['message']}")
```

---

## 📊 Analytics & Reporting

### Automated Reports

#### Daily Summary Report
- **Processing Summary:** Files processed, success rate, average time
- **Cost Analysis:** Daily spend, budget status, cost per document
- **System Health:** Uptime, error summary, performance metrics
- **Insights:** Key findings and recommendations

#### Weekly Trend Report
- **Performance Trends:** Week-over-week comparisons
- **Cost Optimization:** Identified savings opportunities
- **Knowledge Growth:** New relationships and insights discovered
- **System Optimizations:** Recommended improvements

#### Monthly Business Review
- **ROI Analysis:** Cost savings and efficiency gains
- **Usage Patterns:** Team adoption and usage trends
- **System Evolution:** Feature usage and optimization results
- **Future Planning:** Capacity and feature recommendations

### Custom Analytics

```python
class AnalyticsEngine:
    """Custom analytics and reporting engine"""
    
    async def generate_daily_report(self, date: datetime.date):
        """Generate comprehensive daily report"""
        
        # Gather data
        processing_data = await self.get_processing_summary(date)
        cost_data = await self.get_cost_summary(date)
        health_data = await self.get_health_summary(date)
        
        # Generate insights
        insights = await self.analyze_daily_patterns(
            processing_data, cost_data, health_data
        )
        
        # Create report
        report = {
            'date': date.isoformat(),
            'summary': {
                'files_processed': processing_data['total_files'],
                'success_rate': processing_data['success_rate'],
                'total_cost': cost_data['total_spent'],
                'avg_cost_per_document': cost_data['cost_per_document'],
                'system_uptime': health_data['uptime_percentage']
            },
            'insights': insights,
            'recommendations': await self.generate_daily_recommendations(insights)
        }
        
        return report
```

---

## 🔧 Configuration & Setup

### Environment Variables

```bash
# Langfuse Configuration
LANGFUSE_PUBLIC_KEY=pk_your_public_key
LANGFUSE_SECRET_KEY=sk_your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com

# Monitoring Configuration
MONITORING_ENABLED=true
METRICS_COLLECTION_INTERVAL=30
PREDICTIVE_ANALYTICS_ENABLED=true
REAL_TIME_UPDATES_ENABLED=true

# Alert Configuration
ALERT_EMAIL=admin@example.com
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
DAILY_BUDGET=10.00
ALERT_THRESHOLDS_ERROR_RATE=0.05
ALERT_THRESHOLDS_CPU_PERCENT=90

# Performance Configuration
METRICS_RETENTION_DAYS=90
ANALYTICS_BATCH_SIZE=1000
DASHBOARD_UPDATE_INTERVAL=5
```

### Database Setup

```sql
-- Monitoring tables
CREATE TABLE system_health (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    service_name VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    metrics JSONB NOT NULL,
    checked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE processing_analytics (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    date DATE NOT NULL,
    hour INTEGER CHECK (hour >= 0 AND hour < 24),
    metrics JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE alert_history (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    data JSONB,
    resolved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_system_health_service_time 
    ON system_health(service_name, checked_at DESC);
    
CREATE INDEX idx_processing_analytics_date_hour 
    ON processing_analytics(date, hour);
    
CREATE INDEX idx_alert_history_type_time 
    ON alert_history(alert_type, created_at DESC);
```

---

## 🚀 Deployment & Operations

### Monitoring Service Deployment

```dockerfile
# Monitoring service Dockerfile
FROM python:3.11-slim

# Install monitoring dependencies
COPY requirements-monitoring.txt .
RUN pip install -r requirements-monitoring.txt

# Copy monitoring modules
COPY monitoring/ /app/monitoring/
COPY config/ /app/config/

# Set up monitoring environment
ENV PYTHONPATH=/app
ENV MONITORING_CONFIG=/app/config/monitoring.yaml

# Start monitoring services
CMD ["python", "-m", "monitoring.main"]
```

### Health Checks

```python
# Health check endpoints
@app.get("/health/monitoring")
async def monitoring_health_check():
    """Health check for monitoring system"""
    
    checks = {
        'langfuse_connection': await check_langfuse_connection(),
        'supabase_subscriptions': await check_supabase_subscriptions(),
        'websocket_connections': len(active_websockets),
        'metrics_collection': await check_metrics_collection(),
        'alert_system': await check_alert_system()
    }
    
    all_healthy = all(checks.values())
    
    return {
        'status': 'healthy' if all_healthy else 'degraded',
        'checks': checks,
        'timestamp': datetime.now().isoformat()
    }
```

---

## 📚 Integration Examples

### Frontend Integration

```typescript
// Real-time monitoring integration
class MonitoringDashboard {
  private ws: WebSocket;
  private metrics: MonitoringMetrics = {};
  
  constructor() {
    this.setupWebSocket();
  }
  
  private setupWebSocket() {
    this.ws = new WebSocket('ws://localhost:8000/ws/monitoring');
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.handleMonitoringUpdate(data);
    };
  }
  
  private handleMonitoringUpdate(data: any) {
    switch (data.type) {
      case 'health_update':
        this.updateSystemHealth(data.data);
        break;
      case 'cost_update':
        this.updateCostMetrics(data.data);
        break;
      case 'processing_update':
        this.updateProcessingStatus(data.data);
        break;
    }
  }
}
```

### Alert Integration

```python
# Custom alert handler
class CustomAlertHandler:
    async def handle_budget_alert(self, alert_data):
        """Handle budget exceeded alert"""
        
        # Automatically pause expensive operations
        await self.pause_high_cost_operations()
        
        # Send detailed cost breakdown
        cost_breakdown = await self.generate_cost_breakdown()
        await self.send_detailed_cost_report(cost_breakdown)
        
        # Suggest optimizations
        optimizations = await self.suggest_cost_optimizations()
        await self.send_optimization_suggestions(optimizations)
```

---

*Monitoring Architecture v2.0 - Real-time, Predictive, Intelligent*
*Optimized for AI-First Operations with Cost Awareness*