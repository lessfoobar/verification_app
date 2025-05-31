#!/usr/bin/env python3
"""
Recording Session Manager
=========================

Manages recording sessions for real-time video verification feedback.
Handles session lifecycle, feedback storage, and quality assessment.
"""

import threading
import time
import uuid
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from ..core.data_classes import LiveFeedback, RecordingQuality

logger = logging.getLogger(__name__)

@dataclass
class RecordingSession:
    """Recording session data structure"""
    session_id: str
    created_at: datetime
    status: str  # 'active', 'completed', 'expired', 'cancelled'
    config: Dict[str, Any]
    feedback_history: List[Dict[str, Any]]
    quality_analysis: Optional[Dict[str, Any]]
    expires_at: datetime
    last_activity: datetime
    metadata: Dict[str, Any]

class RecordingSessionManager:
    """Thread-safe recording session management"""
    
    def __init__(self, max_sessions: int = 1000, session_timeout_minutes: int = 30):
        """
        Initialize session manager
        
        Args:
            max_sessions: Maximum number of concurrent sessions
            session_timeout_minutes: Session timeout in minutes
        """
        self.max_sessions = max_sessions
        self.session_timeout_minutes = session_timeout_minutes
        self._lock = threading.RLock()
        
        # Session storage
        self._sessions: Dict[str, RecordingSession] = {}
        self._session_feedback: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Metrics
        self._session_metrics = {
            'total_sessions_created': 0,
            'active_sessions': 0,
            'completed_sessions': 0,
            'expired_sessions': 0,
            'cancelled_sessions': 0
        }
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        
        logger.info(f"SessionManager initialized: max_sessions={max_sessions}, timeout={session_timeout_minutes}min")
    
    def create_session(self, config: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new recording session
        
        Args:
            config: Session configuration
            metadata: Optional session metadata
            
        Returns:
            Session ID
        """
        with self._lock:
            # Check session limit
            if len(self._sessions) >= self.max_sessions:
                self._cleanup_expired_sessions()
                
                if len(self._sessions) >= self.max_sessions:
                    raise RuntimeError(f"Maximum sessions limit reached: {self.max_sessions}")
            
            # Generate unique session ID
            session_id = self._generate_session_id()
            
            # Create session
            now = datetime.now()
            expires_at = now + timedelta(minutes=self.session_timeout_minutes)
            
            session = RecordingSession(
                session_id=session_id,
                created_at=now,
                status='active',
                config=config.copy(),
                feedback_history=[],
                quality_analysis=None,
                expires_at=expires_at,
                last_activity=now,
                metadata=metadata or {}
            )
            
            self._sessions[session_id] = session
            self._session_metrics['total_sessions_created'] += 1
            self._session_metrics['active_sessions'] += 1
            
            logger.info(f"Created recording session: {session_id}")
            return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data
        
        Args:
            session_id: Session ID
            
        Returns:
            Session data dictionary or None if not found
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            
            # Check if session is expired
            if self._is_session_expired(session):
                self._expire_session(session_id)
                return None
            
            # Update last activity
            session.last_activity = datetime.now()
            
            return asdict(session)
    
    def session_exists(self, session_id: str) -> bool:
        """
        Check if session exists and is active
        
        Args:
            session_id: Session ID
            
        Returns:
            True if session exists and is active
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            
            if self._is_session_expired(session):
                self._expire_session(session_id)
                return False
            
            return session.status == 'active'
    
    def add_feedback(self, session_id: str, feedback_data: Dict[str, Any]) -> bool:
        """
        Add feedback to a session
        
        Args:
            session_id: Session ID
            feedback_data: Feedback data
            
        Returns:
            True if feedback was added successfully
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.status != 'active':
                logger.warning(f"Cannot add feedback to session {session_id}: session not found or not active")
                return False
            
            # Check if session is expired
            if self._is_session_expired(session):
                self._expire_session(session_id)
                return False
            
            # Add timestamp if not present
            if 'timestamp' not in feedback_data:
                feedback_data['timestamp'] = datetime.now().isoformat()
            
            # Add feedback to session history
            session.feedback_history.append(feedback_data)
            self._session_feedback[session_id].append(feedback_data)
            
            # Update last activity
            session.last_activity = datetime.now()
            
            logger.debug(f"Added feedback to session {session_id}: {len(session.feedback_history)} total")
            return True
    
    def get_session_feedback(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all feedback for a session
        
        Args:
            session_id: Session ID
            
        Returns:
            List of feedback data
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return []
            
            return session.feedback_history.copy()
    
    def finalize_session(self, session_id: str, quality_analysis: Dict[str, Any]) -> bool:
        """
        Finalize a session with quality analysis
        
        Args:
            session_id: Session ID
            quality_analysis: Quality analysis results
            
        Returns:
            True if session was finalized successfully
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                logger.warning(f"Cannot finalize session {session_id}: session not found")
                return False
            
            # Update session
            session.status = 'completed'
            session.quality_analysis = quality_analysis
            session.last_activity = datetime.now()
            
            # Update metrics
            self._session_metrics['active_sessions'] -= 1
            self._session_metrics['completed_sessions'] += 1
            
            logger.info(f"Finalized session {session_id} with quality score: {quality_analysis.get('quality_score', 'unknown')}")
            return True
    
    def cancel_session(self, session_id: str, reason: str = 'User cancelled') -> bool:
        """
        Cancel a session
        
        Args:
            session_id: Session ID
            reason: Cancellation reason
            
        Returns:
            True if session was cancelled successfully
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                logger.warning(f"Cannot cancel session {session_id}: session not found")
                return False
            
            if session.status != 'active':
                logger.warning(f"Cannot cancel session {session_id}: session not active (status: {session.status})")
                return False
            
            # Update session
            session.status = 'cancelled'
            session.metadata['cancellation_reason'] = reason
            session.last_activity = datetime.now()
            
            # Update metrics
            self._session_metrics['active_sessions'] -= 1
            self._session_metrics['cancelled_sessions'] += 1
            
            logger.info(f"Cancelled session {session_id}: {reason}")
            return True
    
    def extend_session(self, session_id: str, additional_minutes: int = 30) -> bool:
        """
        Extend session timeout
        
        Args:
            session_id: Session ID
            additional_minutes: Additional minutes to extend
            
        Returns:
            True if session was extended successfully
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.status != 'active':
                return False
            
            session.expires_at += timedelta(minutes=additional_minutes)
            session.last_activity = datetime.now()
            
            logger.debug(f"Extended session {session_id} by {additional_minutes} minutes")
            return True
    
    def get_active_sessions(self) -> List[str]:
        """
        Get list of active session IDs
        
        Returns:
            List of active session IDs
        """
        with self._lock:
            active_sessions = []
            for session_id, session in self._sessions.items():
                if session.status == 'active' and not self._is_session_expired(session):
                    active_sessions.append(session_id)
            return active_sessions
    
    def get_session_metrics(self) -> Dict[str, Any]:
        """
        Get session metrics
        
        Returns:
            Dictionary with session metrics
        """
        with self._lock:
            # Update active sessions count
            active_count = 0
            for session in self._sessions.values():
                if session.status == 'active' and not self._is_session_expired(session):
                    active_count += 1
            
            self._session_metrics['active_sessions'] = active_count
            
            return {
                'current_metrics': self._session_metrics.copy(),
                'session_breakdown': {
                    'total_sessions': len(self._sessions),
                    'active_sessions': active_count,
                    'completed_sessions': sum(1 for s in self._sessions.values() if s.status == 'completed'),
                    'expired_sessions': sum(1 for s in self._sessions.values() if s.status == 'expired'),
                    'cancelled_sessions': sum(1 for s in self._sessions.values() if s.status == 'cancelled')
                },
                'recent_activity': self._get_recent_activity_stats()
            }
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session summary for reporting
        
        Args:
            session_id: Session ID
            
        Returns:
            Session summary dictionary
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            
            feedback_count = len(session.feedback_history)
            
            # Calculate session duration
            if session.status == 'completed':
                duration = session.last_activity - session.created_at
            else:
                duration = datetime.now() - session.created_at
            
            # Analyze feedback quality if available
            feedback_analysis = self._analyze_session_feedback(session.feedback_history)
            
            return {
                'session_id': session_id,
                'status': session.status,
                'created_at': session.created_at.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'feedback_count': feedback_count,
                'config': session.config,
                'quality_analysis': session.quality_analysis,
                'feedback_analysis': feedback_analysis,
                'metadata': session.metadata
            }
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up old sessions
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of sessions cleaned up
        """
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            sessions_to_remove = []
            
            for session_id, session in self._sessions.items():
                if session.last_activity < cutoff_time and session.status in ['completed', 'expired', 'cancelled']:
                    sessions_to_remove.append(session_id)
            
            # Remove old sessions
            for session_id in sessions_to_remove:
                del self._sessions[session_id]
                if session_id in self._session_feedback:
                    del self._session_feedback[session_id]
            
            logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
            return len(sessions_to_remove)
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = int(time.time() * 1000)
        random_part = str(uuid.uuid4()).replace('-', '')[:8]
        return f"session_{timestamp}_{random_part}"
    
    def _is_session_expired(self, session: RecordingSession) -> bool:
        """Check if session is expired"""
        return datetime.now() > session.expires_at
    
    def _expire_session(self, session_id: str) -> None:
        """Mark session as expired"""
        session = self._sessions.get(session_id)
        if session and session.status == 'active':
            session.status = 'expired'
            session.last_activity = datetime.now()
            
            # Update metrics
            self._session_metrics['active_sessions'] -= 1
            self._session_metrics['expired_sessions'] += 1
            
            logger.debug(f"Session {session_id} expired")
    
    def _cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions"""
        now = datetime.now()
        expired_sessions = []
        
        for session_id, session in self._sessions.items():
            if session.status == 'active' and now > session.expires_at:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self._expire_session(session_id)
    
    def _cleanup_worker(self) -> None:
        """Background worker for session cleanup"""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                
                with self._lock:
                    # Clean up expired sessions
                    self._cleanup_expired_sessions()
                    
                    # Clean up old completed sessions (older than 1 hour)
                    cutoff_time = datetime.now() - timedelta(hours=1)
                    sessions_to_remove = []
                    
                    for session_id, session in self._sessions.items():
                        if (session.status in ['completed', 'expired', 'cancelled'] and 
                            session.last_activity < cutoff_time):
                            sessions_to_remove.append(session_id)
                    
                    for session_id in sessions_to_remove:
                        del self._sessions[session_id]
                        if session_id in self._session_feedback:
                            del self._session_feedback[session_id]
                    
                    if sessions_to_remove:
                        logger.debug(f"Cleaned up {len(sessions_to_remove)} old sessions")
                        
            except Exception as e:
                logger.error(f"Session cleanup worker error: {e}")
    
    def _analyze_session_feedback(self, feedback_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze session feedback for quality metrics"""
        if not feedback_history:
            return {
                'total_feedback': 0,
                'quality_distribution': {},
                'common_issues': [],
                'session_quality': 'unknown'
            }
        
        try:
            # Count feedback by status
            status_counts = defaultdict(int)
            issue_counts = defaultdict(int)
            
            for feedback in feedback_history:
                status = feedback.get('overall_status', 'unknown')
                status_counts[status] += 1
                
                # Extract issues from user messages
                message = feedback.get('user_message', '').lower()
                if 'blurry' in message or 'blur' in message:
                    issue_counts['blur'] += 1
                if 'lighting' in message or 'dark' in message or 'bright' in message:
                    issue_counts['lighting'] += 1
                if 'position' in message or 'center' in message:
                    issue_counts['positioning'] += 1
                if 'face not' in message or 'no face' in message:
                    issue_counts['face_detection'] += 1
            
            total_feedback = len(feedback_history)
            good_ratio = status_counts['good'] / total_feedback if total_feedback > 0 else 0
            
            # Determine session quality
            if good_ratio >= 0.8:
                session_quality = 'excellent'
            elif good_ratio >= 0.6:
                session_quality = 'good'
            elif good_ratio >= 0.4:
                session_quality = 'fair'
            else:
                session_quality = 'poor'
            
            # Get most common issues
            common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            common_issues = [issue for issue, count in common_issues if count > 0]
            
            return {
                'total_feedback': total_feedback,
                'quality_distribution': dict(status_counts),
                'good_ratio': good_ratio,
                'common_issues': common_issues,
                'session_quality': session_quality
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze session feedback: {e}")
            return {
                'total_feedback': len(feedback_history),
                'analysis_error': str(e)
            }
    
    def _get_recent_activity_stats(self) -> Dict[str, Any]:
        """Get recent activity statistics"""
        try:
            now = datetime.now()
            last_hour = now - timedelta(hours=1)
            last_day = now - timedelta(days=1)
            
            recent_sessions = {
                'last_hour': 0,
                'last_day': 0,
                'active_in_last_hour': 0
            }
            
            for session in self._sessions.values():
                if session.created_at > last_hour:
                    recent_sessions['last_hour'] += 1
                if session.created_at > last_day:
                    recent_sessions['last_day'] += 1
                if session.last_activity > last_hour and session.status == 'active':
                    recent_sessions['active_in_last_hour'] += 1
            
            return recent_sessions
            
        except Exception as e:
            logger.error(f"Failed to get recent activity stats: {e}")
            return {}
    
    def export_session_data(self, session_id: str, include_feedback: bool = True) -> Optional[Dict[str, Any]]:
        """
        Export session data for analysis or GDPR compliance
        
        Args:
            session_id: Session ID
            include_feedback: Whether to include feedback data
            
        Returns:
            Complete session data or None if not found
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            
            export_data = {
                'session_info': asdict(session),
                'export_timestamp': datetime.now().isoformat(),
                'export_type': 'complete_session_data'
            }
            
            if include_feedback:
                export_data['feedback_data'] = session.feedback_history.copy()
                export_data['feedback_analysis'] = self._analyze_session_feedback(session.feedback_history)
            
            return export_data
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get comprehensive session statistics"""
        with self._lock:
            stats = {
                'session_counts': self.get_session_metrics(),
                'feedback_stats': self._get_feedback_statistics(),
                'performance_stats': self._get_performance_statistics(),
                'quality_stats': self._get_quality_statistics()
            }
            
            return stats
    
    def _get_feedback_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics across all sessions"""
        try:
            total_feedback = 0
            status_distribution = defaultdict(int)
            
            for session in self._sessions.values():
                total_feedback += len(session.feedback_history)
                
                for feedback in session.feedback_history:
                    status = feedback.get('overall_status', 'unknown')
                    status_distribution[status] += 1
            
            return {
                'total_feedback_items': total_feedback,
                'status_distribution': dict(status_distribution),
                'avg_feedback_per_session': total_feedback / max(len(self._sessions), 1)
            }
            
        except Exception as e:
            logger.error(f"Failed to get feedback statistics: {e}")
            return {}
    
    def _get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        try:
            session_durations = []
            
            for session in self._sessions.values():
                if session.status == 'completed':
                    duration = session.last_activity - session.created_at
                    session_durations.append(duration.total_seconds())
            
            if session_durations:
                import numpy as np
                avg_duration = np.mean(session_durations)
                median_duration = np.median(session_durations)
                min_duration = np.min(session_durations)
                max_duration = np.max(session_durations)
            else:
                avg_duration = median_duration = min_duration = max_duration = 0
            
            return {
                'completed_sessions': len(session_durations),
                'avg_session_duration_seconds': avg_duration,
                'median_session_duration_seconds': median_duration,
                'min_session_duration_seconds': min_duration,
                'max_session_duration_seconds': max_duration
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance statistics: {e}")
            return {}
    
    def _get_quality_statistics(self) -> Dict[str, Any]:
        """Get quality statistics across all sessions"""
        try:
            quality_scores = []
            session_qualities = defaultdict(int)
            
            for session in self._sessions.values():
                if session.quality_analysis:
                    score = session.quality_analysis.get('quality_score', 0)
                    quality_scores.append(score)
                
                # Analyze session feedback quality
                feedback_analysis = self._analyze_session_feedback(session.feedback_history)
                session_quality = feedback_analysis.get('session_quality', 'unknown')
                session_qualities[session_quality] += 1
            
            if quality_scores:
                import numpy as np
                avg_quality = np.mean(quality_scores)
                min_quality = np.min(quality_scores)
                max_quality = np.max(quality_scores)
            else:
                avg_quality = min_quality = max_quality = 0
            
            return {
                'quality_score_stats': {
                    'sessions_with_scores': len(quality_scores),
                    'avg_quality_score': avg_quality,
                    'min_quality_score': min_quality,
                    'max_quality_score': max_quality
                },
                'session_quality_distribution': dict(session_qualities)
            }
            
        except Exception as e:
            logger.error(f"Failed to get quality statistics: {e}")
            return {}
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        # Optionally save session data or perform cleanup
        pass