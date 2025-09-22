"""
System notification module for ErgoSense.
Provides native OS notifications across platforms.
"""

import platform
import os
import subprocess
from typing import Optional

class SystemNotifier:
    def __init__(self):
        self.system = platform.system()
        self._check_requirements()
        
    def _check_requirements(self):
        """Check if system has required notification tools"""
        if self.system == "Darwin":  # macOS
            self.has_osascript = subprocess.run(['which', 'osascript'], 
                                              capture_output=True).returncode == 0
        elif self.system == "Linux":
            self.has_notify_send = subprocess.run(['which', 'notify-send'], 
                                                capture_output=True).returncode == 0
            
    def notify(self, title: str, message: str, urgency: str = "normal") -> bool:
        """
        Send system notification
        Args:
            title: Notification title
            message: Notification message
            urgency: Priority level ("low", "normal", "critical")
        Returns:
            bool: True if notification was sent successfully
        """
        try:
            if self.system == "Darwin":  # macOS
                if not self.has_osascript:
                    return False
                cmd = f'''osascript -e 'display notification "{message}" with title "{title}"' '''
                subprocess.run(cmd, shell=True, check=True)
                return True
                
            elif self.system == "Linux":
                if not self.has_notify_send:
                    return False
                urgency_flag = f"--urgency={urgency}"
                subprocess.run(['notify-send', urgency_flag, title, message], check=True)
                return True
                
            elif self.system == "Windows":
                # Using PowerShell for Windows notifications
                ps_cmd = f'New-BurntToastNotification -Text "{title}","{message}"'
                subprocess.run(['powershell', '-command', ps_cmd], check=True)
                return True
                
        except subprocess.SubprocessError:
            return False
        
        return False  # Unsupported platform
