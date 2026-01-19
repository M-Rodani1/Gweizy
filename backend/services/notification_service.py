"""
Enhanced notification service for gas price alerts.
Supports Email, Discord, Telegram, and Browser notifications.
"""
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Optional
from datetime import datetime
from config import Config
from utils.logger import logger


class NotificationService:
    """Service for sending notifications via multiple channels."""
    
    def __init__(self):
        self.smtp_server = getattr(Config, 'SMTP_SERVER', None)
        self.smtp_port = getattr(Config, 'SMTP_PORT', 587)
        self.smtp_user = getattr(Config, 'SMTP_USER', None)
        self.smtp_password = getattr(Config, 'SMTP_PASSWORD', None)
        self.from_email = getattr(Config, 'FROM_EMAIL', 'noreply@gweizy.com')
        self.discord_webhook_url = getattr(Config, 'DISCORD_WEBHOOK_URL', None)
        self.telegram_bot_token = getattr(Config, 'TELEGRAM_BOT_TOKEN', None)
        self._session = requests.Session()
    
    def send_email(self, to_email: str, subject: str, body: str, html_body: Optional[str] = None) -> bool:
        """
        Send email notification.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            body: Plain text body
            html_body: Optional HTML body
        
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.smtp_server or not self.smtp_user:
            logger.warning("SMTP not configured, cannot send email")
            return False
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = to_email
            
            # Add plain text part
            text_part = MIMEText(body, 'plain')
            msg.attach(text_part)
            
            # Add HTML part if provided
            if html_body:
                html_part = MIMEText(html_body, 'html')
                msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email sent to {to_email}: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False
    
    def send_discord_webhook(self, webhook_url: str, title: str, description: str, 
                            color: int = 0x00FF00, fields: Optional[list] = None) -> bool:
        """
        Send Discord webhook notification.
        
        Args:
            webhook_url: Discord webhook URL
            title: Embed title
            description: Embed description
            color: Embed color (hex as int)
            fields: Optional list of field dicts with 'name' and 'value'
        
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            embed = {
                'title': title,
                'description': description,
                'color': color,
                'timestamp': datetime.utcnow().isoformat(),
                'footer': {'text': 'Gweizy Gas Optimizer'}
            }
            
            if fields:
                embed['fields'] = fields
            
            response = self._session.post(
                webhook_url,
                json={'embeds': [embed]},
                timeout=10
            )
            
            if response.status_code == 204:
                logger.info(f"Discord webhook sent: {title}")
                return True
            else:
                logger.warning(f"Discord webhook failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Discord webhook: {e}")
            return False
    
    def send_telegram_message(self, chat_id: str, message: str, parse_mode: str = 'HTML') -> bool:
        """
        Send Telegram message via bot.
        
        Args:
            chat_id: Telegram chat ID
            message: Message text
            parse_mode: 'HTML' or 'Markdown'
        
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.telegram_bot_token:
            logger.warning("Telegram bot token not configured")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            
            response = self._session.post(
                url,
                json={
                    'chat_id': chat_id,
                    'text': message,
                    'parse_mode': parse_mode
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('ok'):
                    logger.info(f"Telegram message sent to {chat_id}")
                    return True
            
            logger.warning(f"Telegram message failed: {response.status_code}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def send_gas_alert(self, alert: Dict, current_gas: float, chain_name: str = "Base") -> bool:
        """
        Send gas price alert notification via configured method.
        
        Args:
            alert: Alert dictionary with notification_method and notification_target
            current_gas: Current gas price in gwei
            chain_name: Name of the chain
        
        Returns:
            True if sent successfully, False otherwise
        """
        method = alert.get('notification_method', 'browser')
        target = alert.get('notification_target', '')
        alert_type = alert.get('alert_type', 'below')
        threshold = alert.get('threshold_gwei', 0)
        
        direction = "dropped below" if alert_type == 'below' else "rose above"
        
        # Format message
        title = f"Gas Price Alert - {chain_name}"
        message = (
            f"🚨 Gas price {direction} your threshold!\n\n"
            f"Chain: {chain_name}\n"
            f"Threshold: {threshold:.6f} gwei\n"
            f"Current: {current_gas:.6f} gwei\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        
        html_message = f"""
        <h2>🚨 Gas Price Alert</h2>
        <p>Gas price on <strong>{chain_name}</strong> {direction} your threshold!</p>
        <ul>
            <li><strong>Threshold:</strong> {threshold:.6f} gwei</li>
            <li><strong>Current:</strong> {current_gas:.6f} gwei</li>
            <li><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</li>
        </ul>
        <p><a href="https://basegasfeesml.pages.dev">View on Gweizy</a></p>
        """
        
        # Send based on method
        if method == 'email' and target:
            return self.send_email(
                to_email=target,
                subject=title,
                body=message,
                html_body=html_message
            )
        
        elif method == 'discord' and target:
            color = 0x00FF00 if alert_type == 'below' else 0xFF0000
            fields = [
                {'name': 'Chain', 'value': chain_name, 'inline': True},
                {'name': 'Threshold', 'value': f'{threshold:.6f} gwei', 'inline': True},
                {'name': 'Current', 'value': f'{current_gas:.6f} gwei', 'inline': True}
            ]
            return self.send_discord_webhook(
                webhook_url=target,
                title=title,
                description=f"Gas price {direction} threshold!",
                color=color,
                fields=fields
            )
        
        elif method == 'telegram' and target:
            telegram_message = (
                f"<b>🚨 Gas Price Alert</b>\n\n"
                f"Chain: <b>{chain_name}</b>\n"
                f"Gas price {direction} your threshold!\n\n"
                f"Threshold: <code>{threshold:.6f}</code> gwei\n"
                f"Current: <code>{current_gas:.6f}</code> gwei\n\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
            )
            return self.send_telegram_message(chat_id=target, message=telegram_message)
        
        elif method == 'webhook' and target:
            # Generic webhook (POST JSON)
            try:
                response = requests.post(
                    target,
                    json={
                        'alert_type': alert_type,
                        'threshold_gwei': threshold,
                        'current_gwei': current_gas,
                        'chain_name': chain_name,
                        'timestamp': datetime.now().isoformat()
                    },
                    timeout=10
                )
                return response.status_code == 200
            except Exception as e:
                logger.error(f"Failed to send webhook: {e}")
                return False
        
        else:
            logger.warning(f"Unknown notification method or missing target: {method}")
            return False


# Global instance
_notification_service = None

def get_notification_service() -> NotificationService:
    """Get or create notification service instance."""
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service
