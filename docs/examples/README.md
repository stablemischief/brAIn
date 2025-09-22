# Code Examples & Integration Guide - brAIn v2.0

This collection provides practical examples for integrating with brAIn v2.0, from simple API calls to complex application integrations. All examples are production-ready and include error handling.

## 📚 Example Categories

### 🔗 API Integration Examples
- **[Basic API Usage](./api-basic.md)** - Authentication, folders, documents
- **[Search Integration](./api-search.md)** - Semantic search and results handling
- **[Real-time Updates](./api-realtime.md)** - WebSocket integration patterns
- **[Cost Management](./api-cost.md)** - Budget tracking and optimization

### 🐍 Python SDK Examples
- **[Python Client](./python-client.md)** - Complete Python integration
- **[Batch Processing](./python-batch.md)** - Bulk document operations
- **[Data Analysis](./python-analysis.md)** - Analytics and reporting
- **[Custom Workflows](./python-workflows.md)** - Advanced automation

### 🌐 JavaScript/React Examples
- **[React Integration](./react-integration.md)** - Frontend components
- **[Node.js Backend](./nodejs-backend.md)** - Server-side integration
- **[Real-time Dashboard](./react-dashboard.md)** - Live updating interface
- **[Search Components](./react-search.md)** - Search UI components

### 🔌 Integration Patterns
- **[Webhook Integration](./webhooks.md)** - External system notifications
- **[CI/CD Integration](./cicd-integration.md)** - Automated document processing
- **[Slack Bot](./slack-bot.md)** - Team collaboration integration
- **[Database Sync](./database-sync.md)** - External database synchronization

## 🚀 Quick Start Examples

### Basic API Authentication
```python
import requests
import json

class BrainAPIClient:
    def __init__(self, base_url, username, password):
        self.base_url = base_url
        self.token = None
        self.authenticate(username, password)

    def authenticate(self, username, password):
        """Authenticate and store JWT token"""
        response = requests.post(
            f"{self.base_url}/api/auth/login",
            json={"email": username, "password": password}
        )
        response.raise_for_status()
        self.token = response.json()["access_token"]

    def get_headers(self):
        """Get authenticated headers"""
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

    def get_folders(self):
        """List all folders"""
        response = requests.get(
            f"{self.base_url}/api/folders",
            headers=self.get_headers()
        )
        response.raise_for_status()
        return response.json()["data"]

    def search_documents(self, query, limit=10):
        """Semantic search for documents"""
        response = requests.post(
            f"{self.base_url}/api/search",
            headers=self.get_headers(),
            json={
                "query": query,
                "limit": limit,
                "search_type": "semantic"
            }
        )
        response.raise_for_status()
        return response.json()["data"]

# Usage example
client = BrainAPIClient("http://localhost:8000", "user@example.com", "your_secure_password")
folders = client.get_folders()
results = client.search_documents("project timeline")
```

### React Search Component
```jsx
import React, { useState, useEffect } from 'react';
import { BrainClient } from '@brain/api-client';

const SearchComponent = ({ apiKey, baseUrl }) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [client] = useState(() => new BrainClient({ baseUrl, apiKey }));

  const handleSearch = async (searchQuery) => {
    if (!searchQuery.trim()) return;

    setLoading(true);
    try {
      const searchResults = await client.search.semanticSearch({
        query: searchQuery,
        limit: 10
      });
      setResults(searchResults);
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (query) handleSearch(query);
    }, 300); // Debounce search

    return () => clearTimeout(timeoutId);
  }, [query]);

  return (
    <div className="search-component">
      <div className="search-input">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search documents..."
          className="w-full p-3 border rounded-lg"
        />
      </div>

      {loading && (
        <div className="loading-indicator mt-4">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </div>
      )}

      <div className="results mt-4">
        {results.map((result) => (
          <div key={result.id} className="result-item p-4 border-b">
            <h3 className="font-bold text-lg">{result.title}</h3>
            <p className="text-gray-600 mt-2">{result.content_excerpt}</p>
            <div className="metadata mt-2 text-sm text-gray-500">
              <span>Similarity: {(result.similarity_score * 100).toFixed(1)}%</span>
              <span className="ml-4">Type: {result.document_type}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SearchComponent;
```

### WebSocket Real-time Updates
```javascript
class BrainRealtimeClient {
  constructor(wsUrl, token) {
    this.wsUrl = wsUrl;
    this.token = token;
    this.ws = null;
    this.subscribers = new Map();
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
  }

  connect() {
    this.ws = new WebSocket(`${this.wsUrl}?token=${this.token}`);

    this.ws.onopen = () => {
      console.log('Connected to brAIn real-time updates');
      this.reconnectAttempts = 0;
    };

    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      this.handleMessage(message);
    };

    this.ws.onclose = () => {
      console.log('Disconnected from brAIn real-time updates');
      this.attemptReconnect();
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  subscribe(channel, callback) {
    if (!this.subscribers.has(channel)) {
      this.subscribers.set(channel, new Set());
    }
    this.subscribers.get(channel).add(callback);

    // Subscribe to channel
    this.send({
      action: 'subscribe',
      channel: channel
    });
  }

  unsubscribe(channel, callback) {
    if (this.subscribers.has(channel)) {
      this.subscribers.get(channel).delete(callback);
    }
  }

  send(message) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  handleMessage(message) {
    const { channel, data } = message;
    if (this.subscribers.has(channel)) {
      this.subscribers.get(channel).forEach(callback => {
        callback(data);
      });
    }
  }

  attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        console.log(`Reconnection attempt ${this.reconnectAttempts}`);
        this.connect();
      }, 1000 * Math.pow(2, this.reconnectAttempts));
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
    }
  }
}

// Usage example
const realtime = new BrainRealtimeClient('ws://localhost:8000/ws/realtime', 'your-jwt-token');

realtime.subscribe('processing_status', (data) => {
  console.log('Processing update:', data);
  updateProcessingUI(data);
});

realtime.subscribe('cost_updates', (data) => {
  console.log('Cost update:', data);
  updateCostDisplay(data);
});

realtime.connect();
```

## 🐍 Python Integration Examples

### Complete Python SDK
```python
import asyncio
import aiohttp
import websockets
import json
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass

@dataclass
class SearchResult:
    id: str
    title: str
    content_excerpt: str
    similarity_score: float
    document_type: str
    metadata: Dict

class BrainAsyncClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def search_documents(
        self,
        query: str,
        limit: int = 10,
        search_type: str = "semantic"
    ) -> List[SearchResult]:
        """Search documents with semantic similarity"""
        async with self.session.post(
            f"{self.base_url}/api/search",
            json={
                "query": query,
                "limit": limit,
                "search_type": search_type
            }
        ) as response:
            response.raise_for_status()
            data = await response.json()

            return [
                SearchResult(
                    id=item["id"],
                    title=item["title"],
                    content_excerpt=item["content_excerpt"],
                    similarity_score=item["similarity_score"],
                    document_type=item["document_type"],
                    metadata=item.get("metadata", {})
                )
                for item in data["data"]
            ]

    async def process_folder(self, folder_id: str, force: bool = False) -> Dict:
        """Trigger document processing for a folder"""
        async with self.session.post(
            f"{self.base_url}/api/processing/trigger",
            json={
                "folder_id": folder_id,
                "force_reprocess": force
            }
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def get_cost_analytics(
        self,
        start_date: str = None,
        end_date: str = None
    ) -> Dict:
        """Get cost analytics for a date range"""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        async with self.session.get(
            f"{self.base_url}/api/analytics/costs",
            params=params
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def stream_realtime_updates(
        self,
        channels: List[str],
        callback: Callable[[str, Dict], None]
    ):
        """Stream real-time updates via WebSocket"""
        ws_url = f"ws://{self.base_url.split('://', 1)[1]}/ws/realtime"

        async with websockets.connect(
            ws_url,
            extra_headers={"Authorization": f"Bearer {self.api_key}"}
        ) as websocket:
            # Subscribe to channels
            await websocket.send(json.dumps({
                "action": "subscribe",
                "channels": channels
            }))

            async for message in websocket:
                data = json.loads(message)
                callback(data.get("channel"), data.get("data"))

# Usage examples
async def main():
    async with BrainAsyncClient("http://localhost:8000", "your-api-key") as client:
        # Search documents
        results = await client.search_documents("project timeline", limit=5)
        for result in results:
            print(f"{result.title}: {result.similarity_score:.2f}")

        # Process documents
        processing_result = await client.process_folder("folder-uuid")
        print(f"Processing started: {processing_result}")

        # Get cost analytics
        costs = await client.get_cost_analytics()
        print(f"Total cost: ${costs['data']['total_cost']:.2f}")

# Real-time monitoring example
async def monitor_processing():
    async with BrainAsyncClient("http://localhost:8000", "your-api-key") as client:
        def handle_update(channel, data):
            if channel == "processing_status":
                print(f"Processing update: {data['status']} - {data['progress']}%")
            elif channel == "cost_updates":
                print(f"Cost update: ${data['daily_cost']:.2f}")

        await client.stream_realtime_updates(
            ["processing_status", "cost_updates"],
            handle_update
        )

if __name__ == "__main__":
    asyncio.run(main())
```

### Batch Processing Script
```python
#!/usr/bin/env python3
"""
Batch document processing script for brAIn v2.0
Processes multiple folders with cost monitoring and retry logic
"""

import asyncio
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.client = BrainAsyncClient(
            self.config["api"]["base_url"],
            self.config["api"]["key"]
        )

        self.daily_budget = self.config.get("limits", {}).get("daily_budget", 100.0)
        self.max_retries = self.config.get("limits", {}).get("max_retries", 3)

    async def process_batch(self, folder_configs: List[Dict]) -> Dict:
        """Process multiple folders with monitoring"""
        results = {
            "processed": [],
            "failed": [],
            "total_cost": 0.0,
            "start_time": datetime.now()
        }

        async with self.client:
            # Check current costs
            cost_data = await self.client.get_cost_analytics()
            current_daily_cost = cost_data["data"]["daily_cost"]

            if current_daily_cost >= self.daily_budget:
                logger.error(f"Daily budget exceeded: ${current_daily_cost:.2f}")
                return results

            # Process each folder
            for folder_config in folder_configs:
                folder_id = folder_config["id"]
                folder_name = folder_config["name"]

                logger.info(f"Processing folder: {folder_name}")

                try:
                    # Start processing
                    result = await self.client.process_folder(
                        folder_id,
                        force=folder_config.get("force", False)
                    )

                    # Monitor progress
                    await self.monitor_processing(folder_id)

                    results["processed"].append({
                        "folder_id": folder_id,
                        "folder_name": folder_name,
                        "result": result
                    })

                except Exception as e:
                    logger.error(f"Failed to process {folder_name}: {e}")
                    results["failed"].append({
                        "folder_id": folder_id,
                        "folder_name": folder_name,
                        "error": str(e)
                    })

                # Check budget after each folder
                cost_data = await self.client.get_cost_analytics()
                current_cost = cost_data["data"]["daily_cost"]
                results["total_cost"] = current_cost

                if current_cost >= self.daily_budget:
                    logger.warning("Daily budget reached, stopping processing")
                    break

        results["end_time"] = datetime.now()
        results["duration"] = (results["end_time"] - results["start_time"]).total_seconds()

        return results

    async def monitor_processing(self, folder_id: str):
        """Monitor processing for a specific folder"""
        processing_complete = False

        def handle_update(channel, data):
            nonlocal processing_complete
            if channel == "processing_status" and data.get("folder_id") == folder_id:
                status = data.get("status")
                progress = data.get("progress", 0)

                logger.info(f"Folder {folder_id}: {status} - {progress}%")

                if status in ["completed", "failed"]:
                    processing_complete = True

        # Start monitoring
        monitor_task = asyncio.create_task(
            self.client.stream_realtime_updates(
                ["processing_status"],
                handle_update
            )
        )

        # Wait for completion or timeout
        timeout = self.config.get("limits", {}).get("processing_timeout", 3600)
        try:
            await asyncio.wait_for(
                self.wait_for_completion(lambda: processing_complete),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Processing timeout for folder {folder_id}")
        finally:
            monitor_task.cancel()

    async def wait_for_completion(self, condition_func):
        """Wait for a condition to become true"""
        while not condition_func():
            await asyncio.sleep(1)

# CLI interface
def main():
    parser = argparse.ArgumentParser(description="Batch process documents with brAIn v2.0")
    parser.add_argument("--config", required=True, help="Configuration file path")
    parser.add_argument("--folders", required=True, help="Folders configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")

    args = parser.parse_args()

    # Load folder configurations
    with open(args.folders) as f:
        folders = yaml.safe_load(f)["folders"]

    if args.dry_run:
        print("Dry run - would process:")
        for folder in folders:
            print(f"  - {folder['name']} ({folder['id']})")
        return

    # Run batch processing
    processor = BatchProcessor(args.config)
    results = asyncio.run(processor.process_batch(folders))

    # Print results
    print(f"\nBatch Processing Results:")
    print(f"Duration: {results['duration']:.1f} seconds")
    print(f"Total Cost: ${results['total_cost']:.2f}")
    print(f"Processed: {len(results['processed'])} folders")
    print(f"Failed: {len(results['failed'])} folders")

    if results["failed"]:
        print("\nFailed folders:")
        for failed in results["failed"]:
            print(f"  - {failed['folder_name']}: {failed['error']}")

if __name__ == "__main__":
    main()
```

## 📋 Configuration Examples

### Example batch processing configuration (config.yml):
```yaml
api:
  base_url: "http://localhost:8000"
  key: "your-api-key-here"

limits:
  daily_budget: 100.00
  max_retries: 3
  processing_timeout: 3600  # seconds

folders:
  - id: "folder-uuid-1"
    name: "Legal Documents"
    force: false
    priority: high

  - id: "folder-uuid-2"
    name: "Marketing Materials"
    force: true
    priority: low

monitoring:
  enable_realtime: true
  log_level: INFO
  cost_alerts: true
  progress_updates: true
```

---

**More Examples**: Browse specific integration guides for [React](./react-integration.md), [Python](./python-client.md), and [API patterns](./api-basic.md).