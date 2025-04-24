"""
Alerting system for the inventory system.
Monitors system health and sends alerts when issues are detected.
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional
import os
from datetime import datetime, timedelta
from ..config.settings import MODEL_DIR
import json

class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self, email_config: Optional[Dict[str, str]] = None):
        """
        Initialize the alert manager.
        
        Args:
            email_config: Email configuration dictionary
        """
        self.email_config = email_config or {}
        self.alert_history: List[Dict[str, Any]] = []
        self.alert_thresholds = {
            "api_error_rate": 0.1,  # 10% error rate
            "processing_speed": 10,  # items per second
            "memory_usage": 0.9,    # 90% memory usage
            "disk_usage": 0.9,      # 90% disk usage
            "model_accuracy": 0.7    # 70% accuracy
        }
        self.alert_history_file = os.path.join(MODEL_DIR, "alert_history.json")
        self.cooldown_period = timedelta(hours=1)  # Minimum time between alerts

    def check_metrics(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check metrics against thresholds and generate alerts.
        
        Args:
            metrics: Current system metrics
            
        Returns:
            List of generated alerts
        """
        alerts = []
        
        # Check API error rate
        if metrics["api_calls"]["total"] > 0:
            error_rate = metrics["api_calls"]["failed"] / metrics["api_calls"]["total"]
            if error_rate > self.alert_thresholds["api_error_rate"]:
                alerts.append({
                    "type": "api_error_rate",
                    "message": f"High API error rate: {error_rate:.2%}",
                    "severity": "high",
                    "value": error_rate,
                    "threshold": self.alert_thresholds["api_error_rate"]
                })
        
        # Check processing speed
        if "items_per_second" in metrics["processing"]:
            if metrics["processing"]["items_per_second"] < self.alert_thresholds["processing_speed"]:
                alerts.append({
                    "type": "processing_speed",
                    "message": f"Low processing speed: {metrics['processing']['items_per_second']:.2f} items/second",
                    "severity": "medium",
                    "value": metrics["processing"]["items_per_second"],
                    "threshold": self.alert_thresholds["processing_speed"]
                })
        
        # Check memory usage
        if metrics["system"]["memory_usage"] > self.alert_thresholds["memory_usage"]:
            alerts.append({
                "type": "memory_usage",
                "message": f"High memory usage: {metrics['system']['memory_usage']:.2%}",
                "severity": "high",
                "value": metrics["system"]["memory_usage"],
                "threshold": self.alert_thresholds["memory_usage"]
            })
        
        # Check disk usage
        if metrics["system"]["disk_usage"] > self.alert_thresholds["disk_usage"]:
            alerts.append({
                "type": "disk_usage",
                "message": f"High disk usage: {metrics['system']['disk_usage']:.2%}",
                "severity": "high",
                "value": metrics["system"]["disk_usage"],
                "threshold": self.alert_thresholds["disk_usage"]
            })
        
        # Check model accuracy
        if metrics["model"]["predictions"] > 0:
            if metrics["model"]["accuracy"] < self.alert_thresholds["model_accuracy"]:
                alerts.append({
                    "type": "model_accuracy",
                    "message": f"Low model accuracy: {metrics['model']['accuracy']:.2%}",
                    "severity": "medium",
                    "value": metrics["model"]["accuracy"],
                    "threshold": self.alert_thresholds["model_accuracy"]
                })
        
        return alerts

    def process_alerts(self, alerts: List[Dict[str, Any]]) -> None:
        """
        Process generated alerts.
        
        Args:
            alerts: List of alerts to process
        """
        for alert in alerts:
            # Check if similar alert was sent recently
            if not self._should_send_alert(alert):
                continue
            
            # Log alert
            logging.warning(f"Alert: {alert['message']}")
            
            # Send email if configured
            if self.email_config:
                self._send_email_alert(alert)
            
            # Record alert in history
            self._record_alert(alert)

    def _should_send_alert(self, alert: Dict[str, Any]) -> bool:
        """
        Check if alert should be sent based on cooldown period.
        
        Args:
            alert: Alert to check
            
        Returns:
            bool: Whether alert should be sent
        """
        if not self.alert_history:
            return True
        
        last_alert = self.alert_history[-1]
        if last_alert["type"] != alert["type"]:
            return True
        
        last_time = datetime.fromisoformat(last_alert["timestamp"])
        return datetime.now() - last_time > self.cooldown_period

    def _send_email_alert(self, alert: Dict[str, Any]) -> None:
        """
        Send email alert.
        
        Args:
            alert: Alert to send
        """
        try:
            msg = MIMEMultipart()
            msg["From"] = self.email_config["from_email"]
            msg["To"] = self.email_config["to_email"]
            msg["Subject"] = f"Inventory System Alert: {alert['type']}"
            
            body = f"""
            Alert Type: {alert['type']}
            Severity: {alert['severity']}
            Message: {alert['message']}
            Value: {alert['value']}
            Threshold: {alert['threshold']}
            Time: {datetime.now().isoformat()}
            """
            
            msg.attach(MIMEText(body, "plain"))
            
            with smtplib.SMTP(self.email_config["smtp_server"], self.email_config["smtp_port"]) as server:
                if self.email_config.get("use_tls"):
                    server.starttls()
                if self.email_config.get("username") and self.email_config.get("password"):
                    server.login(self.email_config["username"], self.email_config["password"])
                server.send_message(msg)
            
            logging.info(f"Email alert sent for {alert['type']}")
        except Exception as e:
            logging.error(f"Error sending email alert: {e}")

    def _record_alert(self, alert: Dict[str, Any]) -> None:
        """
        Record alert in history.
        
        Args:
            alert: Alert to record
        """
        alert_record = {
            "timestamp": datetime.now().isoformat(),
            **alert
        }
        self.alert_history.append(alert_record)
        self._save_alert_history()

    def _save_alert_history(self) -> None:
        """Save alert history to file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.alert_history_file), exist_ok=True)
            
            # Save alert history
            with open(self.alert_history_file, 'w') as f:
                json.dump(self.alert_history, f, indent=2)
            
            logging.info(f"Alert history saved to {self.alert_history_file}")
        except Exception as e:
            logging.error(f"Error saving alert history: {e}")

    def load_alert_history(self) -> None:
        """Load alert history from file."""
        try:
            if os.path.exists(self.alert_history_file):
                with open(self.alert_history_file, 'r') as f:
                    self.alert_history = json.load(f)
                logging.info(f"Alert history loaded from {self.alert_history_file}")
        except Exception as e:
            logging.error(f"Error loading alert history: {e}")

    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent alert history.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert["timestamp"]) > cutoff_time
        ] 