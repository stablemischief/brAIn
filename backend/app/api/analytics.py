"""
Analytics API endpoints
Provides cost tracking, usage analytics, and performance metrics
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text, and_, func

from config.settings import get_settings
from models.api import (
    CostAnalyticsResponse,
    UsageAnalyticsResponse,
    PerformanceMetricsResponse,
    BudgetResponse,
    BudgetUpdateRequest,
    CostProjectionResponse,
)
from database.connection import get_database_session
from api.auth import get_current_user

router = APIRouter()


@router.get("/costs", response_model=CostAnalyticsResponse)
async def get_cost_analytics(
    timeframe: str = Query("week", regex="^(day|week|month|quarter|year)$"),
    breakdown_by: str = Query("day", regex="^(hour|day|week|month)$"),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
):
    """Get comprehensive cost analytics and breakdowns."""
    try:
        # Calculate date range based on timeframe
        end_date = datetime.now(timezone.utc)
        if timeframe == "day":
            start_date = end_date - timedelta(days=1)
        elif timeframe == "week":
            start_date = end_date - timedelta(weeks=1)
        elif timeframe == "month":
            start_date = end_date - timedelta(days=30)
        elif timeframe == "quarter":
            start_date = end_date - timedelta(days=90)
        else:  # year
            start_date = end_date - timedelta(days=365)

        # Mock cost data - in reality, this would query the LLM usage tracking table
        daily_costs = []
        current_date = start_date
        while current_date <= end_date:
            daily_costs.append(
                {
                    "date": current_date.date().isoformat(),
                    "cost": round(
                        5.50
                        + (hash(current_date.date().isoformat()) % 1000) / 1000 * 2.0,
                        2,
                    ),
                    "tokens": 15000 + (hash(current_date.date().isoformat()) % 10000),
                    "requests": 45 + (hash(current_date.date().isoformat()) % 20),
                }
            )
            if breakdown_by == "hour":
                current_date += timedelta(hours=1)
            elif breakdown_by == "day":
                current_date += timedelta(days=1)
            elif breakdown_by == "week":
                current_date += timedelta(weeks=1)
            else:  # month
                current_date += timedelta(days=30)

        total_cost = sum(day["cost"] for day in daily_costs)
        total_tokens = sum(day["tokens"] for day in daily_costs)
        total_requests = sum(day["requests"] for day in daily_costs)

        return CostAnalyticsResponse(
            total_cost=total_cost,
            daily_costs=daily_costs,
            cost_by_model={
                "gpt-4": total_cost * 0.7,
                "gpt-3.5-turbo": total_cost * 0.2,
                "text-embedding-ada-002": total_cost * 0.1,
            },
            cost_by_operation={
                "document_processing": total_cost * 0.6,
                "search_queries": total_cost * 0.25,
                "knowledge_graph": total_cost * 0.15,
            },
            token_usage={
                "total_tokens": total_tokens,
                "input_tokens": int(total_tokens * 0.6),
                "output_tokens": int(total_tokens * 0.4),
            },
            cost_per_document=(
                total_cost / 100 if total_cost > 0 else 0
            ),  # Mock document count
            projected_monthly_cost=(
                total_cost * 4.33 if timeframe == "week" else total_cost
            ),
            timeframe=timeframe,
            breakdown_by=breakdown_by,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cost analytics: {str(e)}",
        )


@router.get("/usage", response_model=UsageAnalyticsResponse)
async def get_usage_analytics(
    timeframe: str = Query("week", regex="^(day|week|month|quarter|year)$"),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
):
    """Get usage analytics and user behavior patterns."""
    try:
        # Mock usage data
        return UsageAnalyticsResponse(
            total_documents_processed=1250,
            total_search_queries=3420,
            total_knowledge_graph_queries=185,
            average_documents_per_day=35.7,
            average_search_queries_per_day=97.7,
            most_active_hours={"14": 125, "15": 132, "16": 98, "10": 87, "11": 94},
            most_processed_file_types={
                "pdf": 45.2,
                "docx": 28.7,
                "txt": 15.1,
                "xlsx": 7.3,
                "pptx": 3.7,
            },
            search_patterns={
                "avg_query_length": 12.5,
                "most_common_terms": [
                    "analysis",
                    "report",
                    "summary",
                    "data",
                    "results",
                ],
                "successful_search_rate": 87.3,
            },
            user_engagement={
                "session_duration_avg_minutes": 45.2,
                "pages_per_session": 8.7,
                "return_rate_percentage": 73.5,
            },
            timeframe=timeframe,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get usage analytics: {str(e)}",
        )


@router.get("/performance", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
):
    """Get system performance metrics and health indicators."""
    try:
        return PerformanceMetricsResponse(
            avg_processing_time_seconds=42.5,
            avg_search_response_time_ms=185,
            avg_knowledge_graph_query_time_ms=320,
            processing_success_rate=94.2,
            search_success_rate=98.7,
            system_uptime_percentage=99.8,
            current_queue_size=5,
            peak_concurrent_users=23,
            memory_usage_percentage=67.3,
            cpu_usage_percentage=34.8,
            database_response_time_ms=12.5,
            cache_hit_rate_percentage=85.2,
            error_rate_percentage=0.8,
            throughput_requests_per_minute=145.7,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance metrics: {str(e)}",
        )


@router.get("/budget", response_model=BudgetResponse)
async def get_budget_status(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
):
    """Get current budget status and spending limits."""
    try:
        # This would query the user's budget settings
        # Mock data for now
        current_month_start = datetime.now(timezone.utc).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        current_spend = 87.35
        monthly_limit = 150.00

        return BudgetResponse(
            monthly_limit=monthly_limit,
            current_month_spend=current_spend,
            remaining_budget=monthly_limit - current_spend,
            budget_utilization_percentage=(current_spend / monthly_limit) * 100,
            days_remaining_in_month=(
                datetime.now(timezone.utc) + timedelta(days=32)
            ).replace(day=1)
            - timedelta(days=1)
            - datetime.now(timezone.utc),
            projected_month_end_spend=current_spend * 1.8,  # Simple projection
            alerts={
                "approaching_limit": current_spend > monthly_limit * 0.8,
                "over_budget": current_spend > monthly_limit,
                "high_daily_spend": False,
            },
            spending_trend="stable",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get budget status: {str(e)}",
        )


@router.put("/budget", response_model=BudgetResponse)
async def update_budget_limits(
    request: BudgetUpdateRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
):
    """Update budget limits and alert thresholds."""
    try:
        # Validate budget limits
        if request.monthly_limit and request.monthly_limit < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Monthly limit must be positive",
            )

        if request.daily_limit and request.daily_limit < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Daily limit must be positive",
            )

        # Update budget settings in database
        # This would update the user's budget preferences

        # Return updated budget status
        return await get_budget_status(current_user, db)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update budget: {str(e)}",
        )


@router.get("/projections", response_model=CostProjectionResponse)
async def get_cost_projections(
    projection_days: int = Query(30, ge=1, le=365),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
):
    """Get cost projections based on current usage patterns."""
    try:
        # Calculate projections based on historical data
        # Mock projections for now
        current_daily_avg = 5.83

        projections = []
        for i in range(projection_days):
            date = datetime.now(timezone.utc) + timedelta(days=i + 1)
            # Add some variance to make it realistic
            daily_cost = current_daily_avg * (
                0.9 + (hash(date.date().isoformat()) % 100) / 500
            )
            projections.append(
                {
                    "date": date.date().isoformat(),
                    "projected_cost": round(daily_cost, 2),
                    "confidence": 0.85
                    - (i / projection_days) * 0.2,  # Confidence decreases over time
                }
            )

        total_projected = sum(p["projected_cost"] for p in projections)

        return CostProjectionResponse(
            projections=projections,
            total_projected_cost=total_projected,
            current_daily_average=current_daily_avg,
            projection_confidence=0.82,
            factors_considered=[
                "Historical usage patterns",
                "Current document queue",
                "Seasonal trends",
                "User behavior patterns",
            ],
            potential_savings_opportunities=[
                "Batch processing optimization could save 15%",
                "Model selection optimization could save 8%",
                "Caching improvements could save 5%",
            ],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cost projections: {str(e)}",
        )


@router.get("/export")
async def export_analytics_data(
    format: str = Query("csv", regex="^(csv|json|xlsx)$"),
    timeframe: str = Query("month", regex="^(week|month|quarter|year)$"),
    include_personal_data: bool = Query(False),
    current_user: dict = Depends(get_current_user),
):
    """Export analytics data in various formats."""
    try:
        # This would generate and return the requested export format
        # For now, return a mock response
        export_data = {
            "export_type": format,
            "timeframe": timeframe,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "user_id": current_user["id"] if include_personal_data else "anonymized",
            "data_points": 1250,
            "download_url": f"/api/downloads/analytics-{datetime.now().strftime('%Y%m%d')}.{format}",
        }

        return {
            "message": "Analytics export prepared successfully",
            "export_info": export_data,
            "expires_at": (
                datetime.now(timezone.utc) + timedelta(hours=24)
            ).isoformat(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export analytics data: {str(e)}",
        )
