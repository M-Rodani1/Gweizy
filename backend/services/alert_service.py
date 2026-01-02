"""
Gas Price Alert Service
Manages user alert subscriptions and notifications
"""

from datetime import datetime
from typing import List, Dict, Optional
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean
from data.database import Base, DatabaseManager
from utils.logger import logger


class GasAlert(Base):
    """User gas price alert subscription"""
    __tablename__ = 'gas_alerts'

    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True)  # Wallet address or session ID
    alert_type = Column(String)  # 'below', 'above'
    threshold_gwei = Column(Float)
    notification_method = Column(String)  # 'email', 'webhook', 'browser'
    notification_target = Column(String)  # Email address, webhook URL, or browser token
    is_active = Column(Boolean, default=True)
    last_triggered = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.now, index=True)
    cooldown_minutes = Column(Integer, default=30)  # Prevent spam


class AlertService:
    """Service for managing gas price alerts"""

    def __init__(self):
        self.db = DatabaseManager()

    def create_alert(
        self,
        user_id: str,
        alert_type: str,
        threshold_gwei: float,
        notification_method: str = 'browser',
        notification_target: str = '',
        cooldown_minutes: int = 30
    ) -> Dict:
        """Create a new gas price alert"""
        session = self.db._get_session()
        try:
            # Validate inputs
            if alert_type not in ['below', 'above']:
                raise ValueError("alert_type must be 'below' or 'above'")

            if threshold_gwei <= 0:
                raise ValueError("threshold_gwei must be positive")

            if notification_method not in ['email', 'webhook', 'browser', 'discord', 'telegram']:
                raise ValueError("notification_method must be 'email', 'webhook', 'browser', 'discord', or 'telegram'")

            # Create alert
            alert = GasAlert(
                user_id=user_id,
                alert_type=alert_type,
                threshold_gwei=threshold_gwei,
                notification_method=notification_method,
                notification_target=notification_target,
                cooldown_minutes=cooldown_minutes
            )

            session.add(alert)
            session.commit()

            logger.info(f"Created alert for user {user_id}: {alert_type} {threshold_gwei} gwei")

            return {
                'id': alert.id,
                'user_id': alert.user_id,
                'alert_type': alert.alert_type,
                'threshold_gwei': alert.threshold_gwei,
                'notification_method': alert.notification_method,
                'is_active': alert.is_active,
                'created_at': alert.created_at.isoformat()
            }
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to create alert: {e}")
            raise
        finally:
            session.close()

    def get_user_alerts(self, user_id: str) -> List[Dict]:
        """Get all alerts for a user"""
        session = self.db._get_session()
        try:
            alerts = session.query(GasAlert).filter(
                GasAlert.user_id == user_id
            ).order_by(GasAlert.created_at.desc()).all()

            return [{
                'id': a.id,
                'alert_type': a.alert_type,
                'threshold_gwei': a.threshold_gwei,
                'notification_method': a.notification_method,
                'is_active': a.is_active,
                'last_triggered': a.last_triggered.isoformat() if a.last_triggered else None,
                'created_at': a.created_at.isoformat()
            } for a in alerts]
        finally:
            session.close()

    def update_alert(self, alert_id: int, is_active: Optional[bool] = None) -> Dict:
        """Update an alert's active status"""
        session = self.db._get_session()
        try:
            alert = session.query(GasAlert).filter(GasAlert.id == alert_id).first()

            if not alert:
                raise ValueError(f"Alert {alert_id} not found")

            if is_active is not None:
                alert.is_active = is_active

            session.commit()

            return {
                'id': alert.id,
                'is_active': alert.is_active
            }
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update alert: {e}")
            raise
        finally:
            session.close()

    def delete_alert(self, alert_id: int, user_id: str) -> bool:
        """Delete an alert (with user verification)"""
        session = self.db._get_session()
        try:
            alert = session.query(GasAlert).filter(
                GasAlert.id == alert_id,
                GasAlert.user_id == user_id
            ).first()

            if not alert:
                return False

            session.delete(alert)
            session.commit()

            logger.info(f"Deleted alert {alert_id} for user {user_id}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete alert: {e}")
            raise
        finally:
            session.close()

    def check_alerts(self, current_gas_gwei: float, chain_id: int = 8453) -> List[Dict]:
        """
        Check if any alerts should be triggered and send notifications.
        
        Args:
            current_gas_gwei: Current gas price in gwei
            chain_id: Chain ID for chain-specific alerts
        
        Returns:
            List of triggered alerts
        """
        from services.notification_service import get_notification_service
        from data.multichain_collector import CHAINS
        
        session = self.db._get_session()
        triggered_alerts = []
        notification_service = get_notification_service()
        chain_name = CHAINS.get(chain_id, {}).get('name', f'Chain {chain_id}')

        try:
            # Get all active alerts
            alerts = session.query(GasAlert).filter(
                GasAlert.is_active == True
            ).all()

            for alert in alerts:
                # Check if alert should trigger
                should_trigger = False

                if alert.alert_type == 'below' and current_gas_gwei <= alert.threshold_gwei:
                    should_trigger = True
                elif alert.alert_type == 'above' and current_gas_gwei >= alert.threshold_gwei:
                    should_trigger = True

                # Check cooldown
                if should_trigger and alert.last_triggered:
                    from datetime import timedelta
                    cooldown_expires = alert.last_triggered + timedelta(minutes=alert.cooldown_minutes)
                    if datetime.now() < cooldown_expires:
                        should_trigger = False

                if should_trigger:
                    # Update last triggered time
                    alert.last_triggered = datetime.now()
                    session.commit()

                    alert_dict = {
                        'alert_id': alert.id,
                        'user_id': alert.user_id,
                        'alert_type': alert.alert_type,
                        'threshold_gwei': alert.threshold_gwei,
                        'current_gwei': current_gas_gwei,
                        'notification_method': alert.notification_method,
                        'notification_target': alert.notification_target
                    }
                    
                    triggered_alerts.append(alert_dict)
                    
                    # Send notification if not browser-only
                    if alert.notification_method != 'browser':
                        notification_service.send_gas_alert(
                            alert=alert_dict,
                            current_gas=current_gas_gwei,
                            chain_name=chain_name
                        )

            return triggered_alerts

        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
            return []
        finally:
            session.close()
