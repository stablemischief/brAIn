#!/usr/bin/env python3
"""
Health check script for brAIn v2.0
Validates all services are running and responsive
"""

import asyncio
import sys
import os
import json
from typing import Dict, List, Tuple
import httpx
from pydantic import BaseModel, Field
from datetime import datetime
import psycopg2
import redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ServiceHealth(BaseModel):
    """Health status model for a service"""

    name: str
    status: str = Field(default="unknown")
    message: str = Field(default="")
    latency_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthCheckResult(BaseModel):
    """Overall health check result"""

    healthy: bool
    services: List[ServiceHealth]
    timestamp: datetime = Field(default_factory=datetime.now)
    message: str = ""


class HealthChecker:
    """Main health checker class"""

    def __init__(self):
        self.results: List[ServiceHealth] = []
        self.backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        self.frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        self.websocket_url = os.getenv("WEBSOCKET_URL", "ws://localhost:8080")

    async def check_backend(self) -> ServiceHealth:
        """Check FastAPI backend health"""
        service = ServiceHealth(name="backend")
        try:
            start = datetime.now()
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.backend_url}/api/health")
                latency = (datetime.now() - start).total_seconds() * 1000

                if response.status_code == 200:
                    service.status = "healthy"
                    service.message = "Backend API is responsive"
                else:
                    service.status = "unhealthy"
                    service.message = f"Backend returned status {response.status_code}"
                service.latency_ms = latency
        except Exception as e:
            service.status = "unhealthy"
            service.message = f"Backend check failed: {str(e)}"
        return service

    async def check_frontend(self) -> ServiceHealth:
        """Check React frontend health"""
        service = ServiceHealth(name="frontend")
        try:
            start = datetime.now()
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(self.frontend_url)
                latency = (datetime.now() - start).total_seconds() * 1000

                if response.status_code == 200:
                    service.status = "healthy"
                    service.message = "Frontend is serving"
                else:
                    service.status = "unhealthy"
                    service.message = f"Frontend returned status {response.status_code}"
                service.latency_ms = latency
        except Exception as e:
            service.status = "unhealthy"
            service.message = f"Frontend check failed: {str(e)}"
        return service

    def check_postgres(self) -> ServiceHealth:
        """Check PostgreSQL database health"""
        service = ServiceHealth(name="postgres")
        try:
            start = datetime.now()
            conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=os.getenv("POSTGRES_PORT", "5432"),
                user=os.getenv("POSTGRES_USER", "brain_user"),
                password=os.getenv("POSTGRES_PASSWORD", "brain_password"),
                database=os.getenv("POSTGRES_DB", "brain_db"),
                connect_timeout=5,
            )
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            latency = (datetime.now() - start).total_seconds() * 1000

            service.status = "healthy"
            service.message = "PostgreSQL is responsive"
            service.latency_ms = latency
        except Exception as e:
            service.status = "unhealthy"
            service.message = f"PostgreSQL check failed: {str(e)}"
        return service

    def check_redis(self) -> ServiceHealth:
        """Check Redis cache health"""
        service = ServiceHealth(name="redis")
        try:
            start = datetime.now()
            r = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                password=os.getenv("REDIS_PASSWORD", None),
                socket_connect_timeout=5,
            )
            r.ping()
            latency = (datetime.now() - start).total_seconds() * 1000

            service.status = "healthy"
            service.message = "Redis is responsive"
            service.latency_ms = latency
        except Exception as e:
            service.status = "unhealthy"
            service.message = f"Redis check failed: {str(e)}"
        return service

    async def check_websocket(self) -> ServiceHealth:
        """Check WebSocket server health"""
        service = ServiceHealth(name="websocket")
        try:
            # For now, just check if the endpoint is reachable
            # Full WebSocket check would require websockets library
            start = datetime.now()
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Try to connect to WebSocket health endpoint if available
                ws_health_url = self.websocket_url.replace("ws://", "http://").replace(
                    "wss://", "https://"
                )
                response = await client.get(f"{ws_health_url}/health")
                latency = (datetime.now() - start).total_seconds() * 1000

                if response.status_code == 200:
                    service.status = "healthy"
                    service.message = "WebSocket server is responsive"
                else:
                    service.status = "unhealthy"
                    service.message = (
                        f"WebSocket server returned status {response.status_code}"
                    )
                service.latency_ms = latency
        except:
            # If no health endpoint, consider it healthy if backend is up
            service.status = "unknown"
            service.message = "WebSocket health endpoint not available"
        return service

    def check_supabase(self) -> ServiceHealth:
        """Check Supabase connectivity"""
        service = ServiceHealth(name="supabase")
        try:
            import httpx

            start = datetime.now()
            supabase_url = os.getenv("SUPABASE_URL")
            if not supabase_url:
                service.status = "unconfigured"
                service.message = "Supabase URL not configured"
                return service

            # Check Supabase health endpoint
            response = httpx.get(f"{supabase_url}/rest/v1/", timeout=5.0)
            latency = (datetime.now() - start).total_seconds() * 1000

            if response.status_code in [200, 401]:  # 401 is expected without auth
                service.status = "healthy"
                service.message = "Supabase is reachable"
            else:
                service.status = "unhealthy"
                service.message = f"Supabase returned status {response.status_code}"
            service.latency_ms = latency
        except Exception as e:
            service.status = "unhealthy"
            service.message = f"Supabase check failed: {str(e)}"
        return service

    async def run_checks(self) -> HealthCheckResult:
        """Run all health checks"""
        # Run async checks
        backend_task = asyncio.create_task(self.check_backend())
        frontend_task = asyncio.create_task(self.check_frontend())
        websocket_task = asyncio.create_task(self.check_websocket())

        # Run sync checks in executor
        loop = asyncio.get_event_loop()
        postgres_future = loop.run_in_executor(None, self.check_postgres)
        redis_future = loop.run_in_executor(None, self.check_redis)
        supabase_future = loop.run_in_executor(None, self.check_supabase)

        # Gather all results
        services = [
            await backend_task,
            await frontend_task,
            await websocket_task,
            await postgres_future,
            await redis_future,
            await supabase_future,
        ]

        # Determine overall health
        critical_services = ["backend", "postgres"]
        critical_healthy = all(
            s.status == "healthy" for s in services if s.name in critical_services
        )

        all_healthy = all(
            s.status in ["healthy", "unknown", "unconfigured"] for s in services
        )

        result = HealthCheckResult(
            healthy=critical_healthy,
            services=services,
            message=(
                "All critical services are healthy"
                if critical_healthy
                else "Some critical services are unhealthy"
            ),
        )

        return result


async def main():
    """Main health check entry point"""
    checker = HealthChecker()
    result = await checker.run_checks()

    # Print result as JSON
    print(json.dumps(result.model_dump(), indent=2, default=str))

    # Exit with appropriate code
    sys.exit(0 if result.healthy else 1)


if __name__ == "__main__":
    asyncio.run(main())
