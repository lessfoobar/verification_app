#!/usr/bin/env python3
"""
External Application Integration Module
======================================

This module provides a complete integration layer for external applications
to interact with the video verification service. It handles:

1. Authentication and API key management
2. Verification session lifecycle
3. Webhook handling for callbacks
4. Error handling and retry logic
5. Real-time status updates
6. GDPR compliance helpers

Usage Example:
    from api_integration import VerificationAPI
    
    api = VerificationAPI(api_key="your_api_key", base_url="https://verification-service.com")
    
    # Create verification session
    session = api.create_verification_session(
        user_id="user123",
        callback_url="https://yourapp.com/webhook"
    )
    
    # Redirect user to verification URL
    redirect_to(session.verification_url)
    
    # Handle webhook callback
    @app.route('/webhook', methods=['POST'])
    def handle_verification_callback():
        return api.handle_webhook(request)
"""

import requests
import json
import time
import hmac
import hashlib
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
from urllib.parse import urljoin
import uuid

logger = logging.getLogger(__name__)

@dataclass
class VerificationSession:
    """Represents a verification session"""
    verification_id: str
    verification_url: str
    status: str
    expires_at: datetime
    user_metadata: Dict
    created_at: datetime
    callback_url: str
    required_document_types: List[str]
    language: str = "en"

@dataclass
class VerificationResult:
    """Represents the final verification result"""
    verification_id: str
    status: str  # 'approved', 'denied', 'expired'
    confidence_score: float
    verification_timestamp: datetime
    document_type: str
    reviewer_notes: Optional[str]
    face_detection_result: Dict
    liveness_result: Dict
    quality_metrics: Dict

@dataclass
class WebhookPayload:
    """Webhook payload structure"""
    verification_id: str
    external_app_id: str
    user_metadata: Dict
    result: VerificationResult
    signature: str

class VerificationAPIError(Exception):
    """Base exception for API errors"""
    pass

class AuthenticationError(VerificationAPIError):
    """Authentication related errors"""
    pass

class ValidationError(VerificationAPIError):
    """Validation related errors"""
    pass

class RateLimitError(VerificationAPIError):
    """Rate limiting errors"""
    pass

class VerificationAPI:
    """Main API client for verification service integration"""
    
    def __init__(self, api_key: str, base_url: str, 
                 webhook_secret: Optional[str] = None,
                 timeout: int = 30, max_retries: int = 3):
        """
        Initialize the verification API client
        
        Args:
            api_key: Your API key for authentication
            base_url: Base URL of the verification service
            webhook_secret: Secret for webhook signature validation
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.webhook_secret = webhook_secret
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'VerificationAPI-Python/1.0'
        })
        
        # Webhook callbacks
        self._webhook_handlers: Dict[str, Callable] = {}
        
        # Cache for session data
        self._session_cache: Dict[str, VerificationSession] = {}
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with retry logic"""
        url = urljoin(self.base_url, endpoint)
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(
                    method, url, timeout=self.timeout, **kwargs
                )
                
                # Handle specific error codes
                if response.status_code == 401:
                    raise AuthenticationError("Invalid API key or expired token")
                elif response.status_code == 400:
                    error_data = response.json() if response.content else {}
                    raise ValidationError(f"Validation error: {error_data.get('error', 'Unknown')}")
                elif response.status_code == 429:
                    raise RateLimitError("Rate limit exceeded")
                elif response.status_code >= 500:
                    if attempt < self.max_retries:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"Server error {response.status_code}, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        response.raise_for_status()
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(f"Request failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                    continue
                else:
                    raise VerificationAPIError(f"Request failed after {self.max_retries} retries: {e}")
    
    def create_verification_session(self, 
                                  user_id: str,
                                  callback_url: str,
                                  required_document_types: Optional[List[str]] = None,
                                  language: str = "en",
                                  session_metadata: Optional[Dict] = None,
                                  external_app_id: Optional[str] = None) -> VerificationSession:
        """
        Create a new verification session
        
        Args:
            user_id: Unique user identifier
            callback_url: URL to receive webhook notifications
            required_document_types: List of accepted document types
            language: Language code for the verification interface
            session_metadata: Additional metadata to store with the session
            external_app_id: Your application identifier
            
        Returns:
            VerificationSession object with verification URL and details
        """
        try:
            payload = {
                "external_app_id": external_app_id or f"app_{int(time.time())}",
                "callback_url": callback_url,
                "user_metadata": {
                    "user_id": user_id,
                    "session_id": str(uuid.uuid4()),
                    **(session_metadata or {})
                },
                "required_document_types": required_document_types or ["passport", "national_id", "drivers_license"],
                "language": language
            }
            
            response = self._make_request('POST', '/api/v1/verification/initiate', json=payload)
            result = response.json()
            
            # Parse response into VerificationSession
            session = VerificationSession(
                verification_id=result['verification_id'],
                verification_url=result['verification_url'],
                status=result['status'],
                expires_at=datetime.fromisoformat(result['expires_at'].replace('Z', '+00:00')),
                user_metadata=payload['user_metadata'],
                created_at=datetime.now(),
                callback_url=callback_url,
                required_document_types=payload['required_document_types'],
                language=language
            )
            
            # Cache session
            self._session_cache[session.verification_id] = session
            
            logger.info(f"✅ Created verification session: {session.verification_id}")
            return session
            
        except Exception as e:
            logger.error(f"❌ Failed to create verification session: {e}")
            raise
    
    def get_verification_status(self, verification_id: str) -> Dict:
        """
        Get current status of a verification session
        
        Args:
            verification_id: The verification session ID
            
        Returns:
            Dictionary with current verification status and details
        """
        try:
            response = self._make_request('GET', f'/api/v1/verification/{verification_id}')
            result = response.json()
            
            # Update cached session if exists
            if verification_id in self._session_cache:
                self._session_cache[verification_id].status = result['status']
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to get verification status: {e}")
            raise
    
    def cancel_verification(self, verification_id: str, reason: str = "Cancelled by user") -> bool:
        """
        Cancel a verification session
        
        Args:
            verification_id: The verification session ID
            reason: Reason for cancellation
            
        Returns:
            True if successfully cancelled
        """
        try:
            payload = {"reason": reason}
            response = self._make_request('POST', f'/api/v1/verification/{verification_id}/cancel', 
                                        json=payload)
            
            # Update cached session
            if verification_id in self._session_cache:
                self._session_cache[verification_id].status = "cancelled"
            
            logger.info(f"✅ Cancelled verification: {verification_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to cancel verification: {e}")
            raise
    
    def list_verifications(self, 
                          user_id: Optional[str] = None,
                          status: Optional[str] = None,
                          limit: int = 50,
                          offset: int = 0) -> Dict:
        """
        List verification sessions with optional filtering
        
        Args:
            user_id: Filter by user ID
            status: Filter by status
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            Dictionary with verification list and pagination info
        """
        try:
            params = {
                'limit': limit,
                'offset': offset
            }
            
            if user_id:
                params['user_id'] = user_id
            if status:
                params['status'] = status
            
            response = self._make_request('GET', '/api/v1/verifications', params=params)
            return response.json()
            
        except Exception as e:
            logger.error(f"❌ Failed to list verifications: {e}")
            raise
    
    def handle_webhook(self, request_data: bytes, 
                      request_headers: Dict[str, str],
                      verify_signature: bool = True) -> Dict:
        """
        Handle incoming webhook from verification service
        
        Args:
            request_data: Raw request body bytes
            request_headers: HTTP headers from the request
            verify_signature: Whether to verify the webhook signature
            
        Returns:
            Dictionary with processed webhook data
        """
        try:
            # Verify signature if enabled
            if verify_signature and self.webhook_secret:
                if not self._verify_webhook_signature(request_data, request_headers):
                    raise ValidationError("Invalid webhook signature")
            
            # Parse webhook payload
            payload_data = json.loads(request_data.decode('utf-8'))
            
            verification_id = payload_data['verification_id']
            result_data = payload_data['result']
            
            # Create VerificationResult object
            result = VerificationResult(
                verification_id=verification_id,
                status=result_data['status'],
                confidence_score=result_data['confidence_score'],
                verification_timestamp=datetime.fromisoformat(result_data['verification_timestamp'].replace('Z', '+00:00')),
                document_type=result_data.get('document_type', 'unknown'),
                reviewer_notes=result_data.get('reviewer_notes'),
                face_detection_result=result_data.get('face_detection_result', {}),
                liveness_result=result_data.get('liveness_result', {}),
                quality_metrics=result_data.get('quality_metrics', {})
            )
            
            # Create webhook payload object
            webhook_payload = WebhookPayload(
                verification_id=verification_id,
                external_app_id=payload_data['external_app_id'],
                user_metadata=payload_data['user_metadata'],
                result=result,
                signature=payload_data.get('signature', '')
            )
            
            # Call registered handlers
            self._call_webhook_handlers(webhook_payload)
            
            # Update cached session
            if verification_id in self._session_cache:
                self._session_cache[verification_id].status = result.status
            
            logger.info(f"✅ Processed webhook for verification: {verification_id}")
            return asdict(webhook_payload)
            
        except Exception as e:
            logger.error(f"❌ Failed to handle webhook: {e}")
            raise
    
    def register_webhook_handler(self, event_type: str, handler: Callable[[WebhookPayload], None]):
        """
        Register a webhook event handler
        
        Args:
            event_type: Type of event ('approved', 'denied', 'expired', 'all')
            handler: Function to call when event occurs
        """
        self._webhook_handlers[event_type] = handler
        logger.info(f"✅ Registered webhook handler for: {event_type}")
    
    def _verify_webhook_signature(self, payload: bytes, headers: Dict[str, str]) -> bool:
        """Verify webhook signature"""
        if not self.webhook_secret:
            return True
        
        signature_header = headers.get('X-Verification-Signature', '')
        if not signature_header:
            return False
        
        expected_signature = hmac.new(
            self.webhook_secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        provided_signature = signature_header.replace('sha256=', '')
        
        return hmac.compare_digest(expected_signature, provided_signature)
    
    def _call_webhook_handlers(self, payload: WebhookPayload):
        """Call registered webhook handlers"""
        # Call specific status handler
        if payload.result.status in self._webhook_handlers:
            try:
                self._webhook_handlers[payload.result.status](payload)
            except Exception as e:
                logger.error(f"Error in webhook handler for {payload.result.status}: {e}")
        
        # Call 'all' handler
        if 'all' in self._webhook_handlers:
            try:
                self._webhook_handlers['all'](payload)
            except Exception as e:
                logger.error(f"Error in 'all' webhook handler: {e}")
    
    def get_verification_metrics(self, 
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> Dict:
        """
        Get verification metrics for analytics
        
        Args:
            start_date: Start date for metrics (default: 30 days ago)
            end_date: End date for metrics (default: now)
            
        Returns:
            Dictionary with verification metrics
        """
        try:
            params = {}
            
            if start_date:
                params['start_date'] = start_date.isoformat()
            if end_date:
                params['end_date'] = end_date.isoformat()
            
            response = self._make_request('GET', '/api/v1/verification/metrics', params=params)
            return response.json()
            
        except Exception as e:
            logger.error(f"❌ Failed to get verification metrics: {e}")
            raise
    
    def export_verification_data(self, verification_id: str, 
                                include_video: bool = False) -> Dict:
        """
        Export verification data for GDPR compliance
        
        Args:
            verification_id: The verification session ID
            include_video: Whether to include video data in export
            
        Returns:
            Dictionary with all verification data
        """
        try:
            params = {'include_video': include_video}
            response = self._make_request('GET', f'/api/v1/verification/{verification_id}/export',
                                        params=params)
            return response.json()
            
        except Exception as e:
            logger.error(f"❌ Failed to export verification data: {e}")
            raise
    
    def delete_verification_data(self, verification_id: str, 
                               reason: str = "GDPR deletion request") -> bool:
        """
        Delete verification data (GDPR right to erasure)
        
        Args:
            verification_id: The verification session ID
            reason: Reason for deletion
            
        Returns:
            True if successfully deleted
        """
        try:
            payload = {"reason": reason}
            response = self._make_request('DELETE', f'/api/v1/verification/{verification_id}',
                                        json=payload)
            
            # Remove from cache
            if verification_id in self._session_cache:
                del self._session_cache[verification_id]
            
            logger.info(f"✅ Deleted verification data: {verification_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete verification data: {e}")
            raise

# Helper functions for common integration patterns

def create_flask_webhook_handler(api: VerificationAPI):
    """
    Create a Flask route handler for webhooks
    
    Usage:
        from flask import Flask, request
        
        app = Flask(__name__)
        webhook_handler = create_flask_webhook_handler(verification_api)
        
        @app.route('/verification/webhook', methods=['POST'])
        def handle_verification_webhook():
            return webhook_handler()
    """
    def webhook_handler():
        try:
            from flask import request
            
            # Process webhook
            result = api.handle_webhook(
                request.get_data(),
                dict(request.headers)
            )
            
            return {'status': 'ok', 'processed': True}, 200
            
        except ValidationError as e:
            return {'error': 'Invalid signature'}, 401
        except Exception as e:
            logger.error(f"Webhook processing error: {e}")
            return {'error': 'Processing failed'}, 500
    
    return webhook_handler

def create_django_webhook_handler(api: VerificationAPI):
    """
    Create a Django view for webhooks
    
    Usage:
        from django.http import JsonResponse
        from django.views.decorators.csrf import csrf_exempt
        from django.views.decorators.http import require_http_methods
        
        webhook_handler = create_django_webhook_handler(verification_api)
        
        @csrf_exempt
        @require_http_methods(["POST"])
        def verification_webhook(request):
            return webhook_handler(request)
    """
    def webhook_handler(request):
        try:
            from django.http import JsonResponse
            
            # Process webhook
            result = api.handle_webhook(
                request.body,
                dict(request.META)
            )
            
            return JsonResponse({'status': 'ok', 'processed': True})
            
        except ValidationError as e:
            return JsonResponse({'error': 'Invalid signature'}, status=401)
        except Exception as e:
            logger.error(f"Webhook processing error: {e}")
            return JsonResponse({'error': 'Processing failed'}, status=500)
    
    return webhook_handler

# Example usage and integration patterns
if __name__ == '__main__':
    # Example integration
    print("🚀 Verification API Integration Example")
    print("=" * 40)
    
    # Initialize API client
    api = VerificationAPI(
        api_key="your_api_key_here",
        base_url="http://localhost:8001",
        webhook_secret="your_webhook_secret"
    )
    
    # Register webhook handlers
    def handle_approved_verification(payload: WebhookPayload):
        print(f"✅ Verification approved: {payload.verification_id}")
        print(f"   User: {payload.user_metadata['user_id']}")
        print(f"   Confidence: {payload.result.confidence_score:.2f}")
    
    def handle_denied_verification(payload: WebhookPayload):
        print(f"❌ Verification denied: {payload.verification_id}")
        print(f"   User: {payload.user_metadata['user_id']}")
        print(f"   Reason: {payload.result.reviewer_notes or 'Not specified'}")
    
    api.register_webhook_handler('approved', handle_approved_verification)
    api.register_webhook_handler('denied', handle_denied_verification)
    
    # Example: Create verification session
    try:
        session = api.create_verification_session(
            user_id="example_user_123",
            callback_url="https://yourapp.com/verification/webhook",
            required_document_types=["passport", "drivers_license"],
            language="en",
            session_metadata={
                "ip_address": "192.168.1.1",
                "user_agent": "Mozilla/5.0...",
                "source": "web_registration"
            }
        )
        
        print(f"✅ Created verification session")
        print(f"   ID: {session.verification_id}")
        print(f"   URL: {session.verification_url}")
        print(f"   Expires: {session.expires_at}")
        
        # Example: Check status
        status = api.get_verification_status(session.verification_id)
        print(f"📊 Current status: {status['status']}")
        
    except VerificationAPIError as e:
        print(f"❌ API Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print("\n📖 Integration complete. See documentation for more examples.")