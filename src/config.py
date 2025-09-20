"""
Configuration management for Real-time Vision Analysis Application
"""

import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import json

# Configure logging
logger = logging.getLogger(__name__)

class Config:
    """
    Configuration manager for the application
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_file: Optional path to JSON config file
        """
        # Load environment variables
        load_dotenv()
        
        # Default configuration
        self._config = {
            # OpenAI API settings
            'api_key': os.getenv('OPENAI_API_KEY', ''),
            'use_realtime_api': os.getenv('USE_REALTIME_API', 'false').lower() == 'true',
            'model': os.getenv('OPENAI_MODEL', 'gpt-4o'),
            'realtime_model': os.getenv('OPENAI_REALTIME_MODEL', 'gpt-4o-realtime-preview-2024-10-01'),
            
            # Camera settings
            'camera_index': int(os.getenv('CAMERA_INDEX', '0')),
            'frame_width': int(os.getenv('FRAME_WIDTH', '640')),
            'frame_height': int(os.getenv('FRAME_HEIGHT', '480')),
            'fps': int(os.getenv('FPS', '30')),
            
            # Processing settings
            'frame_interval': int(os.getenv('FRAME_INTERVAL', '2')),
            'max_retries': int(os.getenv('MAX_RETRIES', '3')),
            'processing_timeout': float(os.getenv('PROCESSING_TIMEOUT', '30.0')),
            'jpeg_quality': int(os.getenv('JPEG_QUALITY', '80')),
            
            # Audio settings
            'audio_sample_rate': int(os.getenv('AUDIO_SAMPLE_RATE', '24000')),
            'audio_channels': int(os.getenv('AUDIO_CHANNELS', '1')),
            'audio_buffer_size': int(os.getenv('AUDIO_BUFFER_SIZE', '1024')),
            'tts_voice': os.getenv('TTS_VOICE', 'alloy'),
            'use_tts_fallback': os.getenv('USE_TTS_FALLBACK', 'true').lower() == 'true',
            
            # Microphone settings
            'microphone_index': self._parse_optional_int(os.getenv('MICROPHONE_INDEX')),
            'vad_threshold': int(os.getenv('VAD_THRESHOLD', '500')),
            'silence_duration': float(os.getenv('SILENCE_DURATION', '2.0')),
            
            # Application settings
            'show_video': os.getenv('SHOW_VIDEO', 'true').lower() == 'true',
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'log_file': os.getenv('LOG_FILE', 'realtime_vision.log'),
            'max_response_history': int(os.getenv('MAX_RESPONSE_HISTORY', '50')),
            
            # System prompts
            'system_prompt': os.getenv('SYSTEM_PROMPT', 
                'You are an expert in computer vision that you want to understand the meaning of human gestures. '
                'Analyze the image and describe what you see in front of you with particular attention to human '
                'gestures, body language, and their potential meanings. Be concise but informative in your response.'),
            'user_prompt': os.getenv('USER_PROMPT', 'describe what is in front of you'),
            
            # Performance settings
            'max_concurrent_requests': int(os.getenv('MAX_CONCURRENT_REQUESTS', '3')),
            'request_timeout': float(os.getenv('REQUEST_TIMEOUT', '30.0')),
            'rate_limit_delay': float(os.getenv('RATE_LIMIT_DELAY', '1.0')),
        }
        
        # Load config file if provided
        if config_file and os.path.exists(config_file):
            self._load_config_file(config_file)
        
        # Validate configuration
        self._validate_config()
    
    def _parse_optional_int(self, value: str) -> Optional[int]:
        """
        Parse an optional integer from environment variable
        
        Args:
            value: String value from environment variable
            
        Returns:
            int or None: Parsed integer or None if empty/invalid
        """
        if not value or not value.strip():
            return None
        
        # Remove any comments
        value = value.split('#')[0].strip()
        
        if not value:
            return None
        
        try:
            return int(value)
        except ValueError:
            return None
    
    def _load_config_file(self, config_file: str):
        """
        Load configuration from JSON file
        
        Args:
            config_file: Path to JSON config file
        """
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            
            # Update config with file values
            self._config.update(file_config)
            logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            logger.warning(f"Failed to load config file {config_file}: {e}")
    
    def _validate_config(self):
        """
        Validate configuration values
        """
        errors = []
        
        # Validate API key
        if not self._config['api_key']:
            errors.append("OpenAI API key is required")
        
        # Validate camera settings
        if self._config['camera_index'] < 0:
            errors.append("Camera index must be >= 0")
        
        if self._config['frame_width'] <= 0 or self._config['frame_height'] <= 0:
            errors.append("Frame dimensions must be > 0")
        
        if self._config['fps'] <= 0:
            errors.append("FPS must be > 0")
        
        # Validate processing settings
        if self._config['frame_interval'] <= 0:
            errors.append("Frame interval must be > 0")
        
        if self._config['max_retries'] < 0:
            errors.append("Max retries must be >= 0")
        
        if self._config['jpeg_quality'] < 1 or self._config['jpeg_quality'] > 100:
            errors.append("JPEG quality must be between 1 and 100")
        
        # Validate audio settings
        if self._config['audio_sample_rate'] <= 0:
            errors.append("Audio sample rate must be > 0")
        
        if self._config['audio_channels'] not in [1, 2]:
            errors.append("Audio channels must be 1 or 2")
        
        # Validate TTS voice
        valid_voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
        if self._config['tts_voice'] not in valid_voices:
            errors.append(f"TTS voice must be one of: {valid_voices}")
        
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self._config['log_level'].upper() not in valid_log_levels:
            errors.append(f"Log level must be one of: {valid_log_levels}")
        
        if errors:
            error_msg = "Configuration validation errors:\n" + "\n".join(f"- {error}" for error in errors)
            raise ValueError(error_msg)
        
        logger.info("Configuration validation passed")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set configuration value
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self._config[key] = value
    
    def get_camera_config(self) -> Dict[str, Any]:
        """
        Get camera configuration
        
        Returns:
            Dictionary with camera configuration
        """
        return {
            'camera_index': self._config['camera_index'],
            'width': self._config['frame_width'],
            'height': self._config['frame_height'],
            'fps': self._config['fps']
        }
    
    def get_audio_config(self) -> Dict[str, Any]:
        """
        Get audio configuration
        
        Returns:
            Dictionary with audio configuration
        """
        return {
            'sample_rate': self._config['audio_sample_rate'],
            'channels': self._config['audio_channels'],
            'buffer_size': self._config['audio_buffer_size'],
            'voice': self._config['tts_voice'],
            'use_tts_fallback': self._config['use_tts_fallback']
        }
    
    def get_processing_config(self) -> Dict[str, Any]:
        """
        Get processing configuration
        
        Returns:
            Dictionary with processing configuration
        """
        return {
            'frame_interval': self._config['frame_interval'],
            'max_retries': self._config['max_retries'],
            'timeout': self._config['processing_timeout'],
            'jpeg_quality': self._config['jpeg_quality'],
            'max_concurrent_requests': self._config['max_concurrent_requests'],
            'request_timeout': self._config['request_timeout'],
            'rate_limit_delay': self._config['rate_limit_delay']
        }
    
    def get_openai_config(self) -> Dict[str, Any]:
        """
        Get OpenAI configuration
        
        Returns:
            Dictionary with OpenAI configuration
        """
        return {
            'api_key': self._config['api_key'],
            'use_realtime_api': self._config['use_realtime_api'],
            'model': self._config['model'],
            'realtime_model': self._config['realtime_model'],
            'system_prompt': self._config['system_prompt'],
            'user_prompt': self._config['user_prompt']
        }
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values
        
        Returns:
            Dictionary with all configuration
        """
        return self._config.copy()
    
    def save_to_file(self, filename: str):
        """
        Save current configuration to JSON file
        
        Args:
            filename: Output filename
        """
        try:
            # Remove sensitive data before saving
            safe_config = self._config.copy()
            safe_config['api_key'] = '[REDACTED]' if safe_config['api_key'] else ''
            
            with open(filename, 'w') as f:
                json.dump(safe_config, f, indent=2)
            
            logger.info(f"Configuration saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {filename}: {e}")
            raise


class ErrorHandler:
    """
    Centralized error handling for the application
    """
    
    def __init__(self, config: Config):
        """
        Initialize error handler
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.error_counts = {}
        self.max_error_count = 10
        self.error_reset_time = 300  # 5 minutes
        self.last_error_reset = {}
        
    def handle_error(self, error_type: str, error: Exception, context: str = "") -> bool:
        """
        Handle an error with retry logic and circuit breaker pattern
        
        Args:
            error_type: Type of error (e.g., 'camera', 'api', 'audio')
            error: The exception that occurred
            context: Additional context about the error
            
        Returns:
            bool: True if operation should be retried, False if it should be stopped
        """
        import time
        
        # Log the error
        logger.error(f"{error_type} error in {context}: {error}")
        
        # Update error count
        current_time = time.time()
        
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
            self.last_error_reset[error_type] = current_time
        
        # Reset error count if enough time has passed
        if current_time - self.last_error_reset[error_type] > self.error_reset_time:
            self.error_counts[error_type] = 0
            self.last_error_reset[error_type] = current_time
        
        self.error_counts[error_type] += 1
        
        # Check if we've exceeded the error threshold
        if self.error_counts[error_type] > self.max_error_count:
            logger.critical(f"Too many {error_type} errors ({self.error_counts[error_type]}). "
                          f"Stopping operation to prevent damage.")
            return False
        
        # Determine if we should retry based on error type and count
        should_retry = self._should_retry(error_type, error, self.error_counts[error_type])
        
        if should_retry:
            retry_delay = min(2 ** (self.error_counts[error_type] - 1), 30)  # Exponential backoff, max 30s
            logger.info(f"Will retry {error_type} operation in {retry_delay} seconds...")
            time.sleep(retry_delay)
        
        return should_retry
    
    def _should_retry(self, error_type: str, error: Exception, error_count: int) -> bool:
        """
        Determine if an operation should be retried based on error type and count
        
        Args:
            error_type: Type of error
            error: The exception
            error_count: Number of consecutive errors
            
        Returns:
            bool: True if should retry, False otherwise
        """
        max_retries = self.config.get('max_retries', 3)
        
        # Don't retry if we've exceeded max retries
        if error_count > max_retries:
            return False
        
        # Handle specific error types
        if error_type == 'camera':
            # Retry camera errors (device might be temporarily unavailable)
            return True
        
        elif error_type == 'api':
            # Check for specific API errors
            error_str = str(error).lower()
            
            # Don't retry authentication errors
            if 'authentication' in error_str or 'invalid api key' in error_str:
                return False
            
            # Don't retry quota exceeded errors
            if 'quota' in error_str or 'rate limit' in error_str:
                return error_count <= 2  # Only retry a few times for rate limits
            
            # Retry other API errors (network issues, temporary outages)
            return True
        
        elif error_type == 'audio':
            # Retry audio errors (device might be busy)
            return True
        
        elif error_type == 'processing':
            # Retry processing errors
            return True
        
        # Default: retry for a few attempts
        return error_count <= max_retries
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics
        
        Returns:
            Dictionary with error statistics
        """
        return {
            'error_counts': self.error_counts.copy(),
            'last_error_reset': self.last_error_reset.copy(),
            'max_error_count': self.max_error_count,
            'error_reset_time': self.error_reset_time
        }
    
    def reset_error_count(self, error_type: str = None):
        """
        Reset error count for a specific error type or all types
        
        Args:
            error_type: Error type to reset, or None to reset all
        """
        import time
        
        if error_type:
            self.error_counts[error_type] = 0
            self.last_error_reset[error_type] = time.time()
            logger.info(f"Reset error count for {error_type}")
        else:
            self.error_counts.clear()
            self.last_error_reset.clear()
            logger.info("Reset all error counts")


# Global configuration instance
_config = None

def get_config(config_file: Optional[str] = None) -> Config:
    """
    Get the global configuration instance
    
    Args:
        config_file: Optional path to config file
        
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_file)
    return _config


# Example usage
if __name__ == "__main__":
    # Test configuration
    try:
        config = Config()
        print("Configuration loaded successfully!")
        print(f"API Key configured: {'Yes' if config.get('api_key') else 'No'}")
        print(f"Camera index: {config.get('camera_index')}")
        print(f"Frame interval: {config.get('frame_interval')}")
        print(f"Use realtime API: {config.get('use_realtime_api')}")
        
        # Test error handler
        error_handler = ErrorHandler(config)
        
        # Simulate some errors
        try:
            raise Exception("Test API error")
        except Exception as e:
            should_retry = error_handler.handle_error('api', e, 'test context')
            print(f"Should retry API error: {should_retry}")
        
    except Exception as e:
        print(f"Configuration error: {e}")