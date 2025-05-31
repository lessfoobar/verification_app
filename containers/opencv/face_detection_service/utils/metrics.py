#!/usr/bin/env python3
"""
Metrics Manager Utility
========================

Extracted from face_detection.py - handles performance metrics collection and management.
"""

import threading
import time
import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MetricsManager:
    """Thread-safe metrics collection and management"""
    
    def __init__(self, max_stored_times: int = 1000):
        """
        Initialize metrics manager
        
        Args:
            max_stored_times: Maximum number of processing times to store
        """
        self.max_stored_times = max_stored_times
        self._lock = threading.RLock()
        
        # Core metrics
        self._metrics = {
            'requests_processed': 0,
            'faces_detected': 0,
            'spoofs_detected': 0,
            'error_count': 0,
            'video_processing_count': 0,
            'frame_analysis_count': 0
        }
        
        # Processing times
        self._processing_times = deque(maxlen=max_stored_times)
        self._video_processing_times = deque(maxlen=max_stored_times)
        self._frame_processing_times = deque(maxlen=max_stored_times)
        
        # Error tracking
        self._error_types = defaultdict(int)
        self._error_history = deque(maxlen=100)
        
        # Performance tracking
        self._verdict_counts = defaultdict(int)
        self._analysis_status_counts = defaultdict(int)
        
        # Time-based metrics
        self._hourly_metrics = defaultdict(lambda: defaultdict(int))
        self._daily_metrics = defaultdict(lambda: defaultdict(int))
        
        # Service start time
        self._start_time = datetime.now()
        
        logger.info("MetricsManager initialized")
    
    def update_video_metrics(self, processing_time_ms: float, faces_detected: int,
                            spoofs_detected: int, verdict: str) -> None:
        """
        Update metrics for video processing
        
        Args:
            processing_time_ms: Processing time in milliseconds
            faces_detected: Number of faces detected
            spoofs_detected: Number of spoofs detected
            verdict: Processing verdict (PASS, FAIL, RETRY_NEEDED)
        """
        with self._lock:
            self._metrics['requests_processed'] += 1
            self._metrics['video_processing_count'] += 1
            self._metrics['faces_detected'] += faces_detected
            self._metrics['spoofs_detected'] += spoofs_detected
            
            # Store processing time
            self._processing_times.append(processing_time_ms)
            self._video_processing_times.append(processing_time_ms)
            
            # Update verdict counts
            self._verdict_counts[verdict] += 1
            
            # Time-based metrics
            now = datetime.now()
            hour_key = now.strftime('%Y-%m-%d-%H')
            day_key = now.strftime('%Y-%m-%d')
            
            self._hourly_metrics[hour_key]['video_requests'] += 1
            self._hourly_metrics[hour_key]['faces_detected'] += faces_detected
            self._daily_metrics[day_key]['video_requests'] += 1
            self._daily_metrics[day_key]['faces_detected'] += faces_detected
            
            logger.debug(f"Video metrics updated: {verdict} verdict, {processing_time_ms:.0f}ms")
    
    def update_frame_metrics(self, processing_time_ms: Optional[float] = None,
                           faces_detected: int = 0, analysis_status: str = 'unknown') -> None:
        """
        Update metrics for frame analysis
        
        Args:
            processing_time_ms: Processing time in milliseconds
            faces_detected: Number of faces detected
            analysis_status: Analysis status (good, warning, error)
        """
        with self._lock:
            self._metrics['requests_processed'] += 1
            self._metrics['frame_analysis_count'] += 1
            self._metrics['faces_detected'] += faces_detected
            
            if processing_time_ms is not None:
                self._processing_times.append(processing_time_ms)
                self._frame_processing_times.append(processing_time_ms)
            
            # Update analysis status counts
            self._analysis_status_counts[analysis_status] += 1
            
            # Time-based metrics
            now = datetime.now()
            hour_key = now.strftime('%Y-%m-%d-%H')
            day_key = now.strftime('%Y-%m-%d')
            
            self._hourly_metrics[hour_key]['frame_requests'] += 1
            self._daily_metrics[day_key]['frame_requests'] += 1
    
    def update_error_metrics(self, error_type: str, error_details: Optional[str] = None) -> None:
        """
        Update error metrics
        
        Args:
            error_type: Type of error
            error_details: Additional error details
        """
        with self._lock:
            self._metrics['error_count'] += 1
            self._error_types[error_type] += 1
            
            # Store error in history
            error_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': error_type,
                'details': error_details
            }
            self._error_history.append(error_entry)
            
            # Time-based error tracking
            now = datetime.now()
            hour_key = now.strftime('%Y-%m-%d-%H')
            day_key = now.strftime('%Y-%m-%d')
            
            self._hourly_metrics[hour_key]['errors'] += 1
            self._daily_metrics[day_key]['errors'] += 1
            
            logger.warning(f"Error metric updated: {error_type}")
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics report"""
        with self._lock:
            # Calculate processing time statistics
            processing_stats = self._calculate_processing_stats()
            
            # Calculate rates
            uptime_hours = (datetime.now() - self._start_time).total_seconds() / 3600
            request_rate_per_hour = self._metrics['requests_processed'] / max(uptime_hours, 0.001)
            
            # Success rate
            total_requests = self._metrics['requests_processed']
            success_rate = (total_requests - self._metrics['error_count']) / max(total_requests, 1)
            
            return {
                'overview': {
                    'service_uptime_hours': uptime_hours,
                    'total_requests': total_requests,
                    'request_rate_per_hour': request_rate_per_hour,
                    'success_rate': success_rate,
                    'error_rate': self._metrics['error_count'] / max(total_requests, 1)
                },
                'processing': {
                    'video_processing_count': self._metrics['video_processing_count'],
                    'frame_analysis_count': self._metrics['frame_analysis_count'],
                    'faces_detected': self._metrics['faces_detected'],
                    'spoofs_detected': self._metrics['spoofs_detected']
                },
                'performance': processing_stats,
                'verdicts': dict(self._verdict_counts),
                'analysis_status': dict(self._analysis_status_counts),
                'errors': {
                    'total_errors': self._metrics['error_count'],
                    'error_types': dict(self._error_types),
                    'recent_errors': list(self._error_history)[-10:]  # Last 10 errors
                },
                'time_based': {
                    'last_24_hours': self._get_last_24_hours_metrics(),
                    'current_hour': self._get_current_hour_metrics()
                }
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for health checks"""
        with self._lock:
            processing_stats = self._calculate_processing_stats()
            
            return {
                'requests_processed': self._metrics['requests_processed'],
                'avg_processing_time_ms': processing_stats['avg_processing_time_ms'],
                'success_rate': self._calculate_success_rate(),
                'error_count': self._metrics['error_count'],
                'faces_detected': self._metrics['faces_detected'],
                'spoofs_detected': self._metrics['spoofs_detected']
            }
    
    def get_total_requests(self) -> int:
        """Get total number of requests processed"""
        with self._lock:
            return self._metrics['requests_processed']
    
    def get_error_rate(self) -> float:
        """Get current error rate"""
        with self._lock:
            total = self._metrics['requests_processed']
            if total == 0:
                return 0.0
            return self._metrics['error_count'] / total
    
    def get_recent_performance(self, minutes: int = 60) -> Dict[str, Any]:
        """
        Get performance metrics for recent time period
        
        Args:
            minutes: Number of minutes to look back
            
        Returns:
            Performance metrics for the time period
        """
        with self._lock:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            
            # This is a simplified implementation
            # In a real system, you'd store timestamps with each metric
            return {
                'time_period_minutes': minutes,
                'estimated_requests': self._metrics['requests_processed'],  # Placeholder
                'error_rate': self.get_error_rate(),
                'avg_processing_time_ms': self._calculate_avg_processing_time()
            }
    
    def _calculate_processing_stats(self) -> Dict[str, float]:
        """Calculate processing time statistics"""
        if not self._processing_times:
            return {
                'avg_processing_time_ms': 0.0,
                'min_processing_time_ms': 0.0,
                'max_processing_time_ms': 0.0,
                'p50_processing_time_ms': 0.0,
                'p95_processing_time_ms': 0.0,
                'p99_processing_time_ms': 0.0
            }
        
        times = list(self._processing_times)
        return {
            'avg_processing_time_ms': np.mean(times),
            'min_processing_time_ms': np.min(times),
            'max_processing_time_ms': np.max(times),
            'p50_processing_time_ms': np.percentile(times, 50),
            'p95_processing_time_ms': np.percentile(times, 95),
            'p99_processing_time_ms': np.percentile(times, 99)
        }
    
    def _calculate_avg_processing_time(self) -> float:
        """Calculate average processing time"""
        if not self._processing_times:
            return 0.0
        return np.mean(list(self._processing_times))
    
    def _calculate_success_rate(self) -> float:
        """Calculate success rate"""
        total = self._metrics['requests_processed']
        if total == 0:
            return 1.0
        return (total - self._metrics['error_count']) / total
    
    def _get_last_24_hours_metrics(self) -> Dict[str, Any]:
        """Get metrics for last 24 hours"""
        now = datetime.now()
        last_24_hours = []
        
        for i in range(24):
            hour = now - timedelta(hours=i)
            hour_key = hour.strftime('%Y-%m-%d-%H')
            hour_metrics = self._hourly_metrics.get(hour_key, {})
            
            last_24_hours.append({
                'hour': hour.strftime('%H:00'),
                'requests': hour_metrics.get('video_requests', 0) + hour_metrics.get('frame_requests', 0),
                'errors': hour_metrics.get('errors', 0),
                'faces_detected': hour_metrics.get('faces_detected', 0)
            })
        
        return {
            'hourly_breakdown': list(reversed(last_24_hours)),
            'total_requests_24h': sum(h['requests'] for h in last_24_hours),
            'total_errors_24h': sum(h['errors'] for h in last_24_hours)
        }
    
    def _get_current_hour_metrics(self) -> Dict[str, Any]:
        """Get metrics for current hour"""
        now = datetime.now()
        hour_key = now.strftime('%Y-%m-%d-%H')
        hour_metrics = self._hourly_metrics.get(hour_key, {})
        
        return {
            'hour': now.strftime('%Y-%m-%d %H:00'),
            'video_requests': hour_metrics.get('video_requests', 0),
            'frame_requests': hour_metrics.get('frame_requests', 0),
            'errors': hour_metrics.get('errors', 0),
            'faces_detected': hour_metrics.get('faces_detected', 0)
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics (useful for testing)"""
        with self._lock:
            self._metrics = {
                'requests_processed': 0,
                'faces_detected': 0,
                'spoofs_detected': 0,
                'error_count': 0,
                'video_processing_count': 0,
                'frame_analysis_count': 0
            }
            
            self._processing_times.clear()
            self._video_processing_times.clear()
            self._frame_processing_times.clear()
            self._error_types.clear()
            self._error_history.clear()
            self._verdict_counts.clear()
            self._analysis_status_counts.clear()
            self._hourly_metrics.clear()
            self._daily_metrics.clear()
            
            self._start_time = datetime.now()
            
            logger.info("All metrics reset")
    
    def export_metrics(self, format_type: str = 'json') -> str:
        """
        Export metrics in specified format
        
        Args:
            format_type: Export format ('json', 'prometheus', 'csv')
            
        Returns:
            Formatted metrics string
        """
        metrics_data = self.get_all_metrics()
        
        if format_type == 'json':
            import json
            return json.dumps(metrics_data, indent=2, default=str)
        
        elif format_type == 'prometheus':
            return self._export_prometheus_format(metrics_data)
        
        elif format_type == 'csv':
            return self._export_csv_format(metrics_data)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _export_prometheus_format(self, metrics_data: Dict[str, Any]) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        # Basic counters
        lines.append(f"# HELP face_detection_requests_total Total number of requests processed")
        lines.append(f"# TYPE face_detection_requests_total counter")
        lines.append(f"face_detection_requests_total {metrics_data['overview']['total_requests']}")
        
        lines.append(f"# HELP face_detection_errors_total Total number of errors")
        lines.append(f"# TYPE face_detection_errors_total counter")
        lines.append(f"face_detection_errors_total {metrics_data['errors']['total_errors']}")
        
        lines.append(f"# HELP face_detection_faces_detected_total Total faces detected")
        lines.append(f"# TYPE face_detection_faces_detected_total counter")
        lines.append(f"face_detection_faces_detected_total {metrics_data['processing']['faces_detected']}")
        
        # Gauges
        lines.append(f"# HELP face_detection_success_rate Current success rate")
        lines.append(f"# TYPE face_detection_success_rate gauge")
        lines.append(f"face_detection_success_rate {metrics_data['overview']['success_rate']}")
        
        lines.append(f"# HELP face_detection_avg_processing_time_ms Average processing time in milliseconds")
        lines.append(f"# TYPE face_detection_avg_processing_time_ms gauge")
        lines.append(f"face_detection_avg_processing_time_ms {metrics_data['performance']['avg_processing_time_ms']}")
        
        # Histograms for processing times would go here in a full implementation
        
        return '\n'.join(lines)
    
    def _export_csv_format(self, metrics_data: Dict[str, Any]) -> str:
        """Export metrics in CSV format"""
        lines = []
        
        # Header
        lines.append("metric_name,value,timestamp")
        
        timestamp = datetime.now().isoformat()
        
        # Add basic metrics
        for key, value in metrics_data['overview'].items():
            lines.append(f"overview_{key},{value},{timestamp}")
        
        for key, value in metrics_data['processing'].items():
            lines.append(f"processing_{key},{value},{timestamp}")
        
        for key, value in metrics_data['performance'].items():
            lines.append(f"performance_{key},{value},{timestamp}")
        
        return '\n'.join(lines)
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get metrics relevant for health checks"""
        with self._lock:
            return {
                'total_requests': self._metrics['requests_processed'],
                'error_rate': self.get_error_rate(),
                'avg_processing_time_ms': self._calculate_avg_processing_time(),
                'recent_errors': len([e for e in self._error_history 
                                    if datetime.fromisoformat(e['timestamp']) > 
                                    datetime.now() - timedelta(minutes=5)])
            }
    
    def is_healthy(self, max_error_rate: float = 0.1, 
                  max_avg_processing_time: float = 5000) -> bool:
        """
        Check if service is healthy based on metrics
        
        Args:
            max_error_rate: Maximum acceptable error rate
            max_avg_processing_time: Maximum acceptable average processing time
            
        Returns:
            True if service is healthy
        """
        with self._lock:
            error_rate = self.get_error_rate()
            avg_processing_time = self._calculate_avg_processing_time()
            
            return (error_rate <= max_error_rate and 
                   avg_processing_time <= max_avg_processing_time)
    
    def get_alert_conditions(self) -> List[Dict[str, Any]]:
        """Get current alert conditions"""
        alerts = []
        
        with self._lock:
            error_rate = self.get_error_rate()
            avg_processing_time = self._calculate_avg_processing_time()
            
            # High error rate alert
            if error_rate > 0.1:
                alerts.append({
                    'type': 'HIGH_ERROR_RATE',
                    'severity': 'critical' if error_rate > 0.2 else 'warning',
                    'message': f'Error rate is {error_rate:.2%}',
                    'value': error_rate,
                    'threshold': 0.1
                })
            
            # Slow processing alert
            if avg_processing_time > 5000:
                alerts.append({
                    'type': 'SLOW_PROCESSING',
                    'severity': 'critical' if avg_processing_time > 10000 else 'warning',
                    'message': f'Average processing time is {avg_processing_time:.0f}ms',
                    'value': avg_processing_time,
                    'threshold': 5000
                })
            
            # Recent errors spike
            recent_errors = len([e for e in self._error_history 
                               if datetime.fromisoformat(e['timestamp']) > 
                               datetime.now() - timedelta(minutes=5)])
            
            if recent_errors > 10:
                alerts.append({
                    'type': 'ERROR_SPIKE',
                    'severity': 'warning',
                    'message': f'{recent_errors} errors in last 5 minutes',
                    'value': recent_errors,
                    'threshold': 10
                })
        
        return alerts
    
    def record_custom_metric(self, metric_name: str, value: float, 
                           tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a custom metric
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            tags: Optional tags for the metric
        """
        with self._lock:
            # Store custom metrics (simplified implementation)
            if not hasattr(self, '_custom_metrics'):
                self._custom_metrics = defaultdict(list)
            
            metric_entry = {
                'timestamp': datetime.now().isoformat(),
                'value': value,
                'tags': tags or {}
            }
            
            self._custom_metrics[metric_name].append(metric_entry)
            
            # Keep only last 1000 entries per metric
            if len(self._custom_metrics[metric_name]) > 1000:
                self._custom_metrics[metric_name] = self._custom_metrics[metric_name][-1000:]
            
            logger.debug(f"Custom metric recorded: {metric_name} = {value}")
    
    def get_custom_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all custom metrics"""
        with self._lock:
            if hasattr(self, '_custom_metrics'):
                return dict(self._custom_metrics)
            return {}
    
    def cleanup_old_metrics(self, days_to_keep: int = 7) -> None:
        """
        Clean up old time-based metrics
        
        Args:
            days_to_keep: Number of days of metrics to keep
        """
        with self._lock:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean hourly metrics
            keys_to_remove = []
            for hour_key in self._hourly_metrics:
                try:
                    hour_date = datetime.strptime(hour_key, '%Y-%m-%d-%H')
                    if hour_date < cutoff_date:
                        keys_to_remove.append(hour_key)
                except ValueError:
                    # Invalid key format, remove it
                    keys_to_remove.append(hour_key)
            
            for key in keys_to_remove:
                del self._hourly_metrics[key]
            
            # Clean daily metrics
            keys_to_remove = []
            for day_key in self._daily_metrics:
                try:
                    day_date = datetime.strptime(day_key, '%Y-%m-%d')
                    if day_date < cutoff_date:
                        keys_to_remove.append(day_key)
                except ValueError:
                    keys_to_remove.append(day_key)
            
            for key in keys_to_remove:
                del self._daily_metrics[key]
            
            # Clean error history
            cutoff_timestamp = cutoff_date.isoformat()
            self._error_history = deque([
                error for error in self._error_history 
                if error['timestamp'] > cutoff_timestamp
            ], maxlen=100)
            
            logger.info(f"Cleaned up metrics older than {days_to_keep} days")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup if needed"""
        pass