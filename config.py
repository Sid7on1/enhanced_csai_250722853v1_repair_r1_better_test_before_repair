"""
Agent and environment configuration file.

This module provides configuration management for the agent and environment.
It includes settings, parameters, and customization options.
"""

import logging
import os
import json
import yaml
from typing import Dict, List, Optional
from enum import Enum
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
CONFIG_FILE_NAME = 'config.yaml'
DEFAULT_CONFIG = {
    'agent': {
        'name': 'default_agent',
        'version': '1.0.0',
        'log_level': 'INFO'
    },
    'environment': {
        'name': 'default_environment',
        'version': '1.0.0',
        'log_level': 'INFO'
    }
}

class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'
    CRITICAL = 'CRITICAL'

class Config:
    """Configuration class."""
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or CONFIG_FILE_NAME
        self.config = self.load_config()

    def load_config(self) -> Dict:
        """Load configuration from file."""
        try:
            with open(self.config_file, 'r') as file:
                config = yaml.safe_load(file)
                return config or DEFAULT_CONFIG
        except FileNotFoundError:
            logger.warning(f'Config file not found: {self.config_file}')
            return DEFAULT_CONFIG

    def save_config(self) -> None:
        """Save configuration to file."""
        with open(self.config_file, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)

    def get_agent_config(self) -> Dict:
        """Get agent configuration."""
        return self.config['agent']

    def get_environment_config(self) -> Dict:
        """Get environment configuration."""
        return self.config['environment']

    def update_config(self, config: Dict) -> None:
        """Update configuration."""
        self.config.update(config)
        self.save_config()

    def get_log_level(self) -> LogLevel:
        """Get log level."""
        return LogLevel(self.config['agent']['log_level'])

def get_config() -> Config:
    """Get configuration instance."""
    return Config()

def get_agent_config() -> Dict:
    """Get agent configuration."""
    config = get_config()
    return config.get_agent_config()

def get_environment_config() -> Dict:
    """Get environment configuration."""
    config = get_config()
    return config.get_environment_config()

def update_config(config: Dict) -> None:
    """Update configuration."""
    get_config().update_config(config)

def get_log_level() -> LogLevel:
    """Get log level."""
    config = get_config()
    return config.get_log_level()

if __name__ == '__main__':
    # Example usage
    config = get_config()
    agent_config = get_agent_config()
    environment_config = get_environment_config()
    log_level = get_log_level()

    print('Agent Config:')
    print(json.dumps(agent_config, indent=4))
    print('Environment Config:')
    print(json.dumps(environment_config, indent=4))
    print('Log Level:')
    print(log_level.value)