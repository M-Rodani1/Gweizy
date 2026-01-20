"""
RPC Provider Manager with Rotation and Rate Limit Detection

Manages multiple RPC endpoints with automatic rotation on failures or rate limits.
Optimized for high-frequency collection (5-second intervals).
"""

import threading
import time
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from config import Config
from utils.logger import logger


class RPCManager:
    """
    Manages RPC endpoint rotation and health monitoring.
    
    Features:
    - Automatic rotation on failures
    - Rate limit detection (429 errors)
    - Health tracking per endpoint
    - Priority-based endpoint selection
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        
        # Priority-ordered RPC endpoints
        # Primary should be paid provider (Alchemy/Infura)
        # Falls back to free providers
        primary_rpc = Config.BASE_RPC_URL
        
        self.endpoints: List[Dict[str, any]] = [
            {
                'url': primary_rpc,
                'priority': 1,
                'is_paid': 'alchemy' in primary_rpc.lower() or 'infura' in primary_rpc.lower() or 'quicknode' in primary_rpc.lower(),
                'failures': 0,
                'rate_limited_until': None,
                'last_success': datetime.now(),
                'consecutive_failures': 0,
                'total_requests': 0,
                'total_failures': 0
            },
            # Free fallback endpoints
            {
                'url': 'https://base.llamarpc.com',
                'priority': 2,
                'is_paid': False,
                'failures': 0,
                'rate_limited_until': None,
                'last_success': datetime.now(),
                'consecutive_failures': 0,
                'total_requests': 0,
                'total_failures': 0
            },
            {
                'url': 'https://base-rpc.publicnode.com',
                'priority': 3,
                'is_paid': False,
                'failures': 0,
                'rate_limited_until': None,
                'last_success': datetime.now(),
                'consecutive_failures': 0,
                'total_requests': 0,
                'total_failures': 0
            },
            {
                'url': 'https://base.drpc.org',
                'priority': 4,
                'is_paid': False,
                'failures': 0,
                'rate_limited_until': None,
                'last_success': datetime.now(),
                'consecutive_failures': 0,
                'total_requests': 0,
                'total_failures': 0
            }
        ]
        
        # Remove duplicate URLs
        seen_urls = set()
        unique_endpoints = []
        for endpoint in self.endpoints:
            if endpoint['url'] not in seen_urls:
                seen_urls.add(endpoint['url'])
                unique_endpoints.append(endpoint)
        self.endpoints = unique_endpoints
        
        self.current_index = 0
        self.rate_limit_cooldown = 60  # 1 minute cooldown after rate limit
        
        logger.info(f"RPC Manager initialized with {len(self.endpoints)} endpoints")
        for i, ep in enumerate(self.endpoints):
            logger.info(f"  {i+1}. {ep['url']} (priority: {ep['priority']}, paid: {ep['is_paid']})")
    
    def get_current_rpc(self) -> str:
        """Get the current active RPC endpoint URL."""
        with self._lock:
            endpoint = self._get_healthy_endpoint()
            return endpoint['url']
    
    def _get_healthy_endpoint(self) -> Dict:
        """Get the best available endpoint based on health and priority."""
        now = datetime.now()
        
        # Sort by priority, then by health
        available = []
        for endpoint in self.endpoints:
            # Skip if rate limited
            if endpoint['rate_limited_until'] and endpoint['rate_limited_until'] > now:
                continue
            
            # Prefer endpoints with fewer recent failures
            available.append(endpoint)
        
        if not available:
            # All endpoints rate limited, return least recently rate limited
            logger.warning("All RPC endpoints rate limited, using least recent")
            return min(self.endpoints, key=lambda e: e['rate_limited_until'] or datetime.min)
        
        # Sort by priority (lower is better), then by consecutive failures
        available.sort(key=lambda e: (e['priority'], e['consecutive_failures']))
        return available[0]
    
    def record_success(self, url: str) -> None:
        """Record a successful request."""
        with self._lock:
            endpoint = self._find_endpoint(url)
            if endpoint:
                endpoint['last_success'] = datetime.now()
                endpoint['consecutive_failures'] = 0
                endpoint['total_requests'] += 1
    
    def record_failure(self, url: str, is_rate_limit: bool = False) -> None:
        """Record a failed request."""
        with self._lock:
            endpoint = self._find_endpoint(url)
            if endpoint:
                endpoint['failures'] += 1
                endpoint['consecutive_failures'] += 1
                endpoint['total_requests'] += 1
                endpoint['total_failures'] += 1
                
                if is_rate_limit:
                    # Rate limited - mark as unavailable for cooldown period
                    endpoint['rate_limited_until'] = datetime.now() + timedelta(seconds=self.rate_limit_cooldown)
                    logger.warning(
                        f"RPC endpoint rate limited: {url}. "
                        f"Will retry after {self.rate_limit_cooldown}s"
                    )
                    self._rotate_to_next()
                elif endpoint['consecutive_failures'] >= 3:
                    # Too many consecutive failures, rotate
                    logger.warning(
                        f"RPC endpoint has {endpoint['consecutive_failures']} consecutive failures: {url}. Rotating."
                    )
                    self._rotate_to_next()
    
    def _find_endpoint(self, url: str) -> Optional[Dict]:
        """Find endpoint by URL."""
        for endpoint in self.endpoints:
            if endpoint['url'] == url:
                return endpoint
        return None
    
    def _rotate_to_next(self) -> None:
        """Rotate to the next available endpoint."""
        current_url = self._get_healthy_endpoint()['url']
        
        # Find current index
        for i, endpoint in enumerate(self.endpoints):
            if endpoint['url'] == current_url:
                self.current_index = (i + 1) % len(self.endpoints)
                break
        
        next_endpoint = self._get_healthy_endpoint()
        logger.info(f"Rotated to RPC: {next_endpoint['url']}")
    
    def get_stats(self) -> Dict:
        """Get statistics about RPC endpoint usage."""
        with self._lock:
            now = datetime.now()
            stats = {
                'current_rpc': self._get_healthy_endpoint()['url'],
                'endpoints': []
            }
            
            for endpoint in self.endpoints:
                success_rate = (
                    (endpoint['total_requests'] - endpoint['total_failures']) / endpoint['total_requests']
                    if endpoint['total_requests'] > 0 else 0
                )
                
                stats['endpoints'].append({
                    'url': endpoint['url'],
                    'priority': endpoint['priority'],
                    'is_paid': endpoint['is_paid'],
                    'total_requests': endpoint['total_requests'],
                    'total_failures': endpoint['total_failures'],
                    'success_rate': round(success_rate * 100, 2),
                    'consecutive_failures': endpoint['consecutive_failures'],
                    'is_rate_limited': endpoint['rate_limited_until'] and endpoint['rate_limited_until'] > now,
                    'rate_limited_until': endpoint['rate_limited_until'].isoformat() if endpoint['rate_limited_until'] else None,
                    'last_success': endpoint['last_success'].isoformat()
                })
            
            return stats
    
    def reset_endpoint(self, url: str) -> None:
        """Manually reset an endpoint (e.g., after rate limit expires)."""
        with self._lock:
            endpoint = self._find_endpoint(url)
            if endpoint:
                endpoint['consecutive_failures'] = 0
                endpoint['rate_limited_until'] = None
                logger.info(f"Reset RPC endpoint: {url}")


# Global RPC manager instance
_rpc_manager: Optional[RPCManager] = None
_manager_lock = threading.Lock()


def get_rpc_manager() -> RPCManager:
    """Get or create the global RPC manager instance."""
    global _rpc_manager
    if _rpc_manager is None:
        with _manager_lock:
            if _rpc_manager is None:
                _rpc_manager = RPCManager()
    return _rpc_manager
