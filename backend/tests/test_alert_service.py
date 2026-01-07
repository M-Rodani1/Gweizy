#!/usr/bin/env python3
"""
Test Alert Service functionality.
Run: python -m pytest tests/test_alert_service.py -v
"""

import sys
sys.path.append('.')

import os
import tempfile
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Create a test database before importing AlertService
@pytest.fixture(scope='module')
def test_db():
    """Create a temporary test database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test_alerts.db')
        os.environ['TEST_DATABASE_URL'] = f'sqlite:///{db_path}'
        yield db_path


class TestAlertServiceCreate:
    """Test alert creation functionality."""

    @pytest.fixture
    def alert_service(self):
        """Create AlertService with mocked database."""
        with patch('services.alert_service.DatabaseManager') as mock_db:
            mock_session = MagicMock()
            mock_db.return_value._get_session.return_value = mock_session

            from services.alert_service import AlertService, GasAlert
            service = AlertService()
            service._mock_session = mock_session

            yield service, mock_session

    def test_create_alert_valid(self, alert_service):
        """Test creating a valid alert."""
        service, mock_session = alert_service

        # Mock the alert object
        mock_alert = MagicMock()
        mock_alert.id = 1
        mock_alert.user_id = '0x123'
        mock_alert.alert_type = 'below'
        mock_alert.threshold_gwei = 0.01
        mock_alert.notification_method = 'browser'
        mock_alert.is_active = True
        mock_alert.created_at = datetime.now()

        with patch('services.alert_service.GasAlert', return_value=mock_alert):
            result = service.create_alert(
                user_id='0x123',
                alert_type='below',
                threshold_gwei=0.01,
                notification_method='browser'
            )

        assert result['user_id'] == '0x123'
        assert result['alert_type'] == 'below'
        assert result['threshold_gwei'] == 0.01
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    def test_create_alert_invalid_type(self, alert_service):
        """Test that invalid alert_type raises ValueError."""
        service, _ = alert_service

        with pytest.raises(ValueError, match="alert_type must be"):
            service.create_alert(
                user_id='0x123',
                alert_type='invalid',
                threshold_gwei=0.01
            )

    def test_create_alert_negative_threshold(self, alert_service):
        """Test that negative threshold raises ValueError."""
        service, _ = alert_service

        with pytest.raises(ValueError, match="threshold_gwei must be positive"):
            service.create_alert(
                user_id='0x123',
                alert_type='below',
                threshold_gwei=-0.01
            )

    def test_create_alert_invalid_notification_method(self, alert_service):
        """Test that invalid notification method raises ValueError."""
        service, _ = alert_service

        with pytest.raises(ValueError, match="notification_method must be"):
            service.create_alert(
                user_id='0x123',
                alert_type='below',
                threshold_gwei=0.01,
                notification_method='invalid'
            )

    def test_create_alert_all_notification_methods(self, alert_service):
        """Test that all valid notification methods are accepted."""
        service, mock_session = alert_service

        valid_methods = ['email', 'webhook', 'browser', 'discord', 'telegram']

        for method in valid_methods:
            mock_alert = MagicMock()
            mock_alert.id = 1
            mock_alert.user_id = '0x123'
            mock_alert.alert_type = 'below'
            mock_alert.threshold_gwei = 0.01
            mock_alert.notification_method = method
            mock_alert.is_active = True
            mock_alert.created_at = datetime.now()

            with patch('services.alert_service.GasAlert', return_value=mock_alert):
                result = service.create_alert(
                    user_id='0x123',
                    alert_type='below',
                    threshold_gwei=0.01,
                    notification_method=method
                )

            assert result['notification_method'] == method


class TestAlertServiceQuery:
    """Test alert query functionality."""

    @pytest.fixture
    def alert_service(self):
        """Create AlertService with mocked database."""
        with patch('services.alert_service.DatabaseManager') as mock_db:
            mock_session = MagicMock()
            mock_db.return_value._get_session.return_value = mock_session

            from services.alert_service import AlertService
            service = AlertService()
            service._mock_session = mock_session

            yield service, mock_session

    def test_get_user_alerts(self, alert_service):
        """Test getting alerts for a user."""
        service, mock_session = alert_service

        # Mock alert objects
        mock_alert1 = MagicMock()
        mock_alert1.id = 1
        mock_alert1.alert_type = 'below'
        mock_alert1.threshold_gwei = 0.01
        mock_alert1.notification_method = 'browser'
        mock_alert1.is_active = True
        mock_alert1.last_triggered = None
        mock_alert1.created_at = datetime.now()

        mock_alert2 = MagicMock()
        mock_alert2.id = 2
        mock_alert2.alert_type = 'above'
        mock_alert2.threshold_gwei = 0.05
        mock_alert2.notification_method = 'email'
        mock_alert2.is_active = True
        mock_alert2.last_triggered = datetime.now()
        mock_alert2.created_at = datetime.now()

        mock_query = MagicMock()
        mock_query.filter.return_value.order_by.return_value.all.return_value = [mock_alert1, mock_alert2]
        mock_session.query.return_value = mock_query

        result = service.get_user_alerts('0x123')

        assert len(result) == 2
        assert result[0]['id'] == 1
        assert result[1]['id'] == 2

    def test_get_user_alerts_empty(self, alert_service):
        """Test getting alerts when user has none."""
        service, mock_session = alert_service

        mock_query = MagicMock()
        mock_query.filter.return_value.order_by.return_value.all.return_value = []
        mock_session.query.return_value = mock_query

        result = service.get_user_alerts('0x123')

        assert result == []


class TestAlertServiceUpdate:
    """Test alert update functionality."""

    @pytest.fixture
    def alert_service(self):
        """Create AlertService with mocked database."""
        with patch('services.alert_service.DatabaseManager') as mock_db:
            mock_session = MagicMock()
            mock_db.return_value._get_session.return_value = mock_session

            from services.alert_service import AlertService
            service = AlertService()
            service._mock_session = mock_session

            yield service, mock_session

    def test_update_alert_active_status(self, alert_service):
        """Test updating alert active status."""
        service, mock_session = alert_service

        mock_alert = MagicMock()
        mock_alert.id = 1
        mock_alert.is_active = True

        mock_query = MagicMock()
        mock_query.filter.return_value.first.return_value = mock_alert
        mock_session.query.return_value = mock_query

        result = service.update_alert(alert_id=1, is_active=False)

        assert result['id'] == 1
        assert mock_alert.is_active == False
        mock_session.commit.assert_called_once()

    def test_update_alert_not_found(self, alert_service):
        """Test updating non-existent alert."""
        service, mock_session = alert_service

        mock_query = MagicMock()
        mock_query.filter.return_value.first.return_value = None
        mock_session.query.return_value = mock_query

        with pytest.raises(ValueError, match="not found"):
            service.update_alert(alert_id=999, is_active=False)


class TestAlertServiceDelete:
    """Test alert deletion functionality."""

    @pytest.fixture
    def alert_service(self):
        """Create AlertService with mocked database."""
        with patch('services.alert_service.DatabaseManager') as mock_db:
            mock_session = MagicMock()
            mock_db.return_value._get_session.return_value = mock_session

            from services.alert_service import AlertService
            service = AlertService()
            service._mock_session = mock_session

            yield service, mock_session

    def test_delete_alert_success(self, alert_service):
        """Test successful alert deletion."""
        service, mock_session = alert_service

        mock_alert = MagicMock()
        mock_alert.id = 1

        mock_query = MagicMock()
        mock_query.filter.return_value.first.return_value = mock_alert
        mock_session.query.return_value = mock_query

        result = service.delete_alert(alert_id=1, user_id='0x123')

        assert result == True
        mock_session.delete.assert_called_once_with(mock_alert)
        mock_session.commit.assert_called_once()

    def test_delete_alert_not_found(self, alert_service):
        """Test deleting non-existent alert returns False."""
        service, mock_session = alert_service

        mock_query = MagicMock()
        mock_query.filter.return_value.first.return_value = None
        mock_session.query.return_value = mock_query

        result = service.delete_alert(alert_id=999, user_id='0x123')

        assert result == False
        mock_session.delete.assert_not_called()

    def test_delete_alert_wrong_user(self, alert_service):
        """Test that deleting alert of another user returns False."""
        service, mock_session = alert_service

        # The filter includes user_id, so wrong user will return None
        mock_query = MagicMock()
        mock_query.filter.return_value.first.return_value = None
        mock_session.query.return_value = mock_query

        result = service.delete_alert(alert_id=1, user_id='0xWRONG')

        assert result == False


class TestAlertServiceTrigger:
    """Test alert triggering logic."""

    @pytest.fixture
    def alert_service(self):
        """Create AlertService with mocked database."""
        with patch('services.alert_service.DatabaseManager') as mock_db:
            mock_session = MagicMock()
            mock_db.return_value._get_session.return_value = mock_session

            from services.alert_service import AlertService
            service = AlertService()
            service._mock_session = mock_session

            yield service, mock_session

    def test_check_alerts_below_threshold(self, alert_service):
        """Test alert triggers when price is below threshold."""
        service, mock_session = alert_service

        mock_alert = MagicMock()
        mock_alert.id = 1
        mock_alert.user_id = '0x123'
        mock_alert.alert_type = 'below'
        mock_alert.threshold_gwei = 0.01
        mock_alert.notification_method = 'browser'
        mock_alert.notification_target = ''
        mock_alert.is_active = True
        mock_alert.last_triggered = None
        mock_alert.cooldown_minutes = 30

        mock_query = MagicMock()
        mock_query.filter.return_value.all.return_value = [mock_alert]
        mock_session.query.return_value = mock_query

        with patch('services.alert_service.get_notification_service') as mock_notif:
            mock_notif.return_value = MagicMock()
            with patch('services.alert_service.CHAINS', {8453: {'name': 'Base'}}):
                result = service.check_alerts(current_gas_gwei=0.005)

        assert len(result) == 1
        assert result[0]['alert_id'] == 1
        assert result[0]['current_gwei'] == 0.005

    def test_check_alerts_above_threshold(self, alert_service):
        """Test alert triggers when price is above threshold."""
        service, mock_session = alert_service

        mock_alert = MagicMock()
        mock_alert.id = 1
        mock_alert.user_id = '0x123'
        mock_alert.alert_type = 'above'
        mock_alert.threshold_gwei = 0.05
        mock_alert.notification_method = 'browser'
        mock_alert.notification_target = ''
        mock_alert.is_active = True
        mock_alert.last_triggered = None
        mock_alert.cooldown_minutes = 30

        mock_query = MagicMock()
        mock_query.filter.return_value.all.return_value = [mock_alert]
        mock_session.query.return_value = mock_query

        with patch('services.alert_service.get_notification_service') as mock_notif:
            mock_notif.return_value = MagicMock()
            with patch('services.alert_service.CHAINS', {8453: {'name': 'Base'}}):
                result = service.check_alerts(current_gas_gwei=0.1)

        assert len(result) == 1
        assert result[0]['alert_type'] == 'above'

    def test_check_alerts_cooldown_respected(self, alert_service):
        """Test that cooldown prevents re-triggering."""
        service, mock_session = alert_service

        mock_alert = MagicMock()
        mock_alert.id = 1
        mock_alert.user_id = '0x123'
        mock_alert.alert_type = 'below'
        mock_alert.threshold_gwei = 0.01
        mock_alert.notification_method = 'browser'
        mock_alert.notification_target = ''
        mock_alert.is_active = True
        mock_alert.last_triggered = datetime.now() - timedelta(minutes=5)  # Triggered 5 min ago
        mock_alert.cooldown_minutes = 30  # 30 min cooldown

        mock_query = MagicMock()
        mock_query.filter.return_value.all.return_value = [mock_alert]
        mock_session.query.return_value = mock_query

        with patch('services.alert_service.get_notification_service') as mock_notif:
            mock_notif.return_value = MagicMock()
            with patch('services.alert_service.CHAINS', {8453: {'name': 'Base'}}):
                result = service.check_alerts(current_gas_gwei=0.005)

        assert len(result) == 0  # Should not trigger due to cooldown

    def test_check_alerts_cooldown_expired(self, alert_service):
        """Test that alert triggers after cooldown expires."""
        service, mock_session = alert_service

        mock_alert = MagicMock()
        mock_alert.id = 1
        mock_alert.user_id = '0x123'
        mock_alert.alert_type = 'below'
        mock_alert.threshold_gwei = 0.01
        mock_alert.notification_method = 'browser'
        mock_alert.notification_target = ''
        mock_alert.is_active = True
        mock_alert.last_triggered = datetime.now() - timedelta(minutes=60)  # Triggered 60 min ago
        mock_alert.cooldown_minutes = 30  # 30 min cooldown (expired)

        mock_query = MagicMock()
        mock_query.filter.return_value.all.return_value = [mock_alert]
        mock_session.query.return_value = mock_query

        with patch('services.alert_service.get_notification_service') as mock_notif:
            mock_notif.return_value = MagicMock()
            with patch('services.alert_service.CHAINS', {8453: {'name': 'Base'}}):
                result = service.check_alerts(current_gas_gwei=0.005)

        assert len(result) == 1  # Should trigger as cooldown expired

    def test_check_alerts_no_trigger_when_not_met(self, alert_service):
        """Test that alert doesn't trigger when condition not met."""
        service, mock_session = alert_service

        mock_alert = MagicMock()
        mock_alert.id = 1
        mock_alert.alert_type = 'below'
        mock_alert.threshold_gwei = 0.01
        mock_alert.is_active = True
        mock_alert.last_triggered = None
        mock_alert.cooldown_minutes = 30

        mock_query = MagicMock()
        mock_query.filter.return_value.all.return_value = [mock_alert]
        mock_session.query.return_value = mock_query

        with patch('services.alert_service.get_notification_service') as mock_notif:
            mock_notif.return_value = MagicMock()
            with patch('services.alert_service.CHAINS', {8453: {'name': 'Base'}}):
                result = service.check_alerts(current_gas_gwei=0.05)  # Above threshold

        assert len(result) == 0  # Should not trigger


# Manual test runner for backwards compatibility
if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'pytest', __file__, '-v', '--tb=short'],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    sys.exit(result.returncode)
