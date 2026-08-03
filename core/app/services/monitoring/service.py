# app/services/monitoring/service.py
import asyncio
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from app.config import settings
from app.logger import get_logger
from app.services.monitoring.models import (
    Alert,
    AlertRule,
    AlertSeverity,
    MetricRecord,
    MetricType,
    SystemMetrics,
)
from app.utils.decorators import timed

logger = get_logger(__name__)


class MonitoringService:
    """Monitoring and alerting service.
    
    Consumes:
    - logger: structured logging
    - config: monitoring configuration
    - All services via metrics collection
    
    Provides:
    - Metric collection and querying
    - Alert rule evaluation
    - System health metrics aggregation
    """
    
    def __init__(self):
        self._metrics: Dict[str, List[MetricRecord]] = defaultdict(list)
        self._alert_rules: Dict[str, AlertRule] = {}
        self._alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._last_evaluation: Dict[str, datetime] = {}  # rule_id -> last_eval_time
        self._evaluation_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start background alert evaluation."""
        if self._evaluation_task is None:
            self._evaluation_task = asyncio.create_task(self._evaluate_alerts_loop())
            logger.info("Monitoring service started - alert evaluation running")
    
    async def stop(self):
        """Stop background tasks."""
        if self._evaluation_task:
            self._evaluation_task.cancel()
            self._evaluation_task = None
    
    @timed
    async def record_metric(
        self,
        metric_type: MetricType,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MetricRecord:
        """Record a metric data point."""
        record = MetricRecord(
            id=uuid4(),
            metric_type=metric_type,
            value=value,
            tags=tags or {},
            metadata=metadata or {},
        )
        
        self._metrics[metric_type.value].append(record)
        
        # Prune old metrics (keep last 24 hours)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        self._metrics[metric_type.value] = [
            m for m in self._metrics[metric_type.value]
            if m.timestamp > cutoff
        ]
        
        logger.debug(f"Metric recorded: {metric_type.value}={value} tags={tags}")
        return record
    
    @timed
    async def query_metrics(
        self,
        metric_type: MetricType,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
        aggregation: Optional[str] = None,  # "avg", "sum", "min", "max", "count"
    ) -> List[MetricRecord]:
        """Query metrics with optional filtering and aggregation."""
        records = self._metrics.get(metric_type.value, [])
        
        # Time filter
        if start_time:
            records = [r for r in records if r.timestamp >= start_time]
        if end_time:
            records = [r for r in records if r.timestamp <= end_time]
        
        # Tag filter
        if tags:
            records = [
                r for r in records
                if all(r.tags.get(k) == v for k, v in tags.items())
            ]
        
        return sorted(records, key=lambda r: r.timestamp)
    
    @timed
    async def get_metric_summary(
        self,
        metric_type: MetricType,
        window_seconds: int = 3600,
    ) -> Dict[str, Any]:
        """Get summary statistics for a metric over a time window."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
        records = [
            r for r in self._metrics.get(metric_type.value, [])
            if r.timestamp >= cutoff
        ]
        
        if not records:
            return {"count": 0, "window_seconds": window_seconds}
        
        values = [r.value for r in records]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1],
            "window_seconds": window_seconds,
        }
    
    @timed
    async def create_alert_rule(self, rule: AlertRule) -> AlertRule:
        """Create an alert rule."""
        self._alert_rules[str(rule.id)] = rule
        logger.info(f"Alert rule created: {rule.name} ({rule.metric_type.value} {rule.condition})")
        return rule
    
    async def get_alert_rules(self) -> List[AlertRule]:
        """Get all alert rules."""
        return list(self._alert_rules.values())
    
    async def get_alerts(
        self, acknowledged: Optional[bool] = None, severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """Get triggered alerts with filtering."""
        alerts = list(self._alerts.values())
        
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda a: a.triggered_at, reverse=True)
    
    @timed
    async def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert."""
        alert = self._alerts.get(alert_id)
        if alert:
            alert.acknowledged = True
            alert.acknowledged_by = user
            logger.info(f"Alert acknowledged: {alert.rule_name} by {user}")
            return True
        return False
    
    async def _evaluate_alerts_loop(self, interval: int = 30):
        """Background loop to evaluate alert rules."""
        while True:
            try:
                await self._evaluate_all_rules()
            except Exception as e:
                logger.exception(f"Alert evaluation error: {e}")
            await asyncio.sleep(interval)
    
    async def _evaluate_all_rules(self):
        """Evaluate all enabled alert rules."""
        now = datetime.now(timezone.utc)
        
        for rule_id, rule in self._alert_rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            last_eval = self._last_evaluation.get(rule_id)
            if last_eval and (now - last_eval).total_seconds() < rule.cooldown_seconds:
                continue
            
            # Get recent metrics
            cutoff = now - timedelta(seconds=rule.window_seconds)
            records = [
                r for r in self._metrics.get(rule.metric_type.value, [])
                if r.timestamp >= cutoff
            ]
            
            if not records:
                continue
            
            # Evaluate condition on latest value
            latest_value = records[-1].value
            triggered = self._evaluate_condition(latest_value, rule.condition)
            
            if triggered:
                await self._trigger_alert(rule, latest_value)
            
            self._last_evaluation[rule_id] = now
    
    def _evaluate_condition(self, value: float, condition: str) -> bool:
        """Evaluate a simple condition string against a value."""
        import operator
        import re
        
        ops = {
            ">": operator.gt,
            "<": operator.lt,
            ">=": operator.ge,
            "<=": operator.le,
            "==": operator.eq,
            "!=": operator.ne,
        }
        
        match = re.match(rf"([{''.join(ops.keys())}]+)\s*([\d.]+)", condition)
        if not match:
            return False
        
        op_str, threshold_str = match.groups()
        threshold = float(threshold_str)
        op_func = ops.get(op_str)
        
        if op_func:
            return op_func(value, threshold)
        return False
    
    async def _trigger_alert(self, rule: AlertRule, value: float):
        """Trigger an alert for a rule."""
        alert = Alert(
            id=uuid4(),
            rule_id=rule.id,
            rule_name=rule.name,
            severity=rule.severity,
            metric_type=rule.metric_type,
            current_value=value,
            threshold_condition=rule.condition,
            message=f"{rule.name}: {rule.metric_type.value} = {value} (condition: {rule.condition})",
        )
        
        self._alerts[str(alert.id)] = alert
        self._alert_history.append(alert)
        
        log_level = "error" if rule.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY] else "warning"
        getattr(logger, log_level)(f"ALERT: {alert.message}")
        
        # Keep only last 100 active alerts
        if len(self._alerts) > 100:
            oldest = min(self._alerts.values(), key=lambda a: a.triggered_at)
            del self._alerts[str(oldest.id)]
    
    @timed
    async def get_system_metrics(self) -> SystemMetrics:
        """Get current system-wide metrics snapshot."""
        import psutil
        
        # System resources
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory().percent
        disk = psutil.disk_usage("/").percent
        
        # API metrics
        api_summary = await self.get_metric_summary(MetricType.API_REQUEST_COUNT, 60)
        latency_summary = await self.get_metric_summary(MetricType.API_LATENCY, 60)
        
        # Pipeline metrics
        pipeline_summary = await self.get_metric_summary(MetricType.PIPELINE_RUN_COUNT, 3600)
        
        # ML metrics
        pred_summary = await self.get_metric_summary(MetricType.MODEL_PREDICTION_COUNT, 86400)
        
        return SystemMetrics(
            cpu_percent=cpu,
            memory_percent=memory,
            disk_percent=disk,
            api_requests_per_minute=api_summary.get("count", 0),
            avg_latency_ms=latency_summary.get("avg", 0),
            active_pipelines=int(pipeline_summary.get("latest", 0)),
            models_deployed=0,  # Would query ML service
            feature_groups=0,  # Would query feature store
            total_predictions_today=int(pred_summary.get("count", 0)),
        )