#!/usr/bin/env python3
"""
Real-time Vision Analysis Application

A real-time computer vision application that uses your camera to analyze what's in front of you
and provides audio and text responses using OpenAI's Real-time API or Vision API.

Features:
- Real-time video capture from computer camera
- AI-powered visual analysis with gesture understanding
- Audio and text responses from OpenAI
- Continuous monitoring and description of environment

Usage:
    python main.py [--realtime] [--camera-index 0] [--frame-interval 2]
"""

import asyncio
import sys
import os
import argparse
import signal
import time
import threading
from datetime import datetime
import cv2
import logging
from dotenv import load_dotenv

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.camera_capture import CameraCapture
from src.frame_processor import RealtimeVisionPipeline
from src.audio_handler import AudioResponseHandler
from src.openai_realtime import OpenAIRealtimeClient
from src.multimodal_processor import RealtimeMultimodalPipeline
from src.microphone_capture import MicrophoneCapture
from src.config import Config, ErrorHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('realtime_vision.log')
    ]
)
logger = logging.getLogger(__name__)

class RealtimeVisionApp:
    """
    Main application class for real-time vision analysis
    """
    
    def __init__(self, config: Config):
        """
        Initialize the real-time vision application
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.error_handler = ErrorHandler(config)
        
        # Extract commonly used config values
        self.api_key = config.get('api_key')
        self.camera_index = config.get('camera_index')
        self.frame_interval = config.get('frame_interval')
        self.use_realtime_api = config.get('use_realtime_api')
        self.show_video = config.get('show_video')
        
        # Initialize components
        self.multimodal_pipeline = None  # For realtime API with voice
        self.vision_pipeline = None      # For vision-only mode
        self.audio_handler = None
        self.camera = None
        
        # Application state
        self.is_running = False
        self.start_time = None
        
        # Response storage for display
        self.recent_responses = []
        self.transcripts = []
        self.max_recent_responses = 10
        
        # Statistics
        self.stats = {
            'total_responses': 0,
            'total_errors': 0,
            'voice_interactions': 0,
            'uptime_seconds': 0
        }
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def initialize(self) -> bool:
        """
        Initialize all application components
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing Real-time Vision Application...")
            
            # Initialize audio handler
            logger.info("Setting up audio response handler...")
            audio_config = self.config.get_audio_config()
            self.audio_handler = AudioResponseHandler(
                api_key=self.api_key, 
                use_tts_fallback=audio_config['use_tts_fallback']
            )
            
            if not self.audio_handler.start():
                logger.error("Failed to start audio handler")
                return False
            
            # Initialize vision pipeline
            logger.info("Setting up vision processing pipeline...")
            camera_config = self.config.get_camera_config()
            self.vision_pipeline = RealtimeVisionPipeline(
                api_key=self.api_key,
                camera_index=camera_config['camera_index'],
                frame_interval=self.frame_interval,
                use_realtime_api=self.use_realtime_api
            )
            
            # Set up callbacks for responses
            self.vision_pipeline.set_callbacks(
                text_callback=self._handle_text_response,
                audio_callback=self._handle_audio_response,
                error_callback=self._handle_error_response
            )
            
            # Start vision pipeline
            if not await self.vision_pipeline.start_pipeline():
                logger.error("Failed to start vision pipeline")
                return False
            
            # Setup camera for display if needed
            if self.show_video:
                self.camera = self.vision_pipeline.camera
            
            logger.info("Application initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            return False
    
    def _handle_text_response(self, text: str):
        """Handle text response from AI"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            response_data = {
                'timestamp': timestamp,
                'type': 'text',
                'content': text
            }
            
            # Add to recent responses
            self.recent_responses.append(response_data)
            if len(self.recent_responses) > self.max_recent_responses:
                self.recent_responses.pop(0)
            
            # Update statistics
            self.stats['total_responses'] += 1
            
            # Log the response
            logger.info(f"AI Response: {text}")
            
            # Print to console with formatting
            print(f"\n[{timestamp}] 🤖 AI: {text}\n")
            
            # Handle audio (TTS if not using realtime API)
            self.audio_handler.handle_text_response(text)
            
        except Exception as e:
            logger.error(f"Error handling text response: {e}")
            self._handle_error_response(str(e))
    
    def _handle_audio_response(self, audio_data: bytes):
        """Handle audio response from AI"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Handle real-time audio playback
            self.audio_handler.handle_audio_response(audio_data)
            
            # Log audio reception
            logger.debug(f"Received audio chunk: {len(audio_data)} bytes")
            
            # Update statistics
            self.stats['total_responses'] += 1
            
        except Exception as e:
            logger.error(f"Error handling audio response: {e}")
            self._handle_error_response(str(e))
    
    def _handle_error_response(self, error_msg: str):
        """Handle error responses"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Update statistics
            self.stats['total_errors'] += 1
            
            # Use error handler for structured error handling
            error_exception = Exception(error_msg)
            should_retry = self.error_handler.handle_error('api', error_exception, 'AI response')
            
            # Log the error
            logger.error(f"AI Error: {error_msg}")
            
            # Print to console
            print(f"\n[{timestamp}] ❌ Error: {error_msg}\n")
            
            # If too many errors, consider stopping
            if not should_retry:
                logger.critical("Too many API errors, stopping application")
                self.stop()
            
        except Exception as e:
            logger.error(f"Error handling error response: {e}")
    
    def _display_status(self):
        """Display current application status"""
        try:
            # Calculate uptime
            if self.start_time:
                self.stats['uptime_seconds'] = int(time.time() - self.start_time)
            
            # Get pipeline status
            pipeline_status = self.vision_pipeline.get_status() if self.vision_pipeline else {}
            
            # Get audio status
            audio_stats = self.audio_handler.get_statistics() if self.audio_handler else {}
            
            # Clear screen and display status
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("=" * 60)
            print("         REAL-TIME VISION ANALYSIS")
            print("=" * 60)
            print(f"Status: {'🟢 RUNNING' if self.is_running else '🔴 STOPPED'}")
            print(f"API Mode: {'Real-time API' if self.use_realtime_api else 'Vision API + TTS'}")
            print(f"Camera: {'🟢 Active' if pipeline_status.get('camera_available', False) else '🔴 Inactive'}")
            print(f"Uptime: {self.stats['uptime_seconds']}s")
            print(f"Responses: {self.stats['total_responses']}")
            print(f"Errors: {self.stats['total_errors']}")
            
            if pipeline_status.get('processing_stats'):
                ps = pipeline_status['processing_stats']
                print(f"Frames Processed: {ps.get('total_frames_processed', 0)}")
                print(f"Processing Rate: {ps.get('processing_rate', 0):.1f}%")
            
            print("-" * 60)
            print("Recent AI Responses:")
            print("-" * 60)
            
            if self.recent_responses:
                for response in self.recent_responses[-3:]:  # Show last 3 responses
                    print(f"[{response['timestamp']}] {response['content'][:80]}...")
            else:
                print("No responses yet...")
            
            print("-" * 60)
            print("Press 'q' in video window or Ctrl+C to quit")
            print("=" * 60)
            
        except Exception as e:
            logger.error(f"Error displaying status: {e}")
    
    async def run(self):
        """
        Main application run loop
        """
        try:
            self.is_running = True
            self.start_time = time.time()
            
            logger.info("Starting Real-time Vision Analysis...")
            print("\n🚀 Real-time Vision Analysis Started!")
            print("Looking for gestures and analyzing what's in front of the camera...")
            
            # Main application loop
            while self.is_running:
                try:
                    # Display video if enabled
                    if self.show_video and self.camera:
                        self.camera.display_frame("Real-time Vision Analysis")
                        
                        # Check for quit command
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            logger.info("Quit command received from video window")
                            break
                    
                    # Update status display every 5 seconds
                    if int(time.time()) % 5 == 0:
                        self._display_status()
                    
                    # Small delay to prevent excessive CPU usage
                    await asyncio.sleep(0.1)
                    
                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt received")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(1)  # Prevent rapid error loops
            
        except Exception as e:
            logger.error(f"Error in application run loop: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """
        Cleanup application resources
        """
        try:
            logger.info("Cleaning up application resources...")
            
            self.is_running = False
            
            # Stop vision pipeline
            if self.vision_pipeline:
                self.vision_pipeline.stop_pipeline()
            
            # Stop audio handler
            if self.audio_handler:
                self.audio_handler.stop()
            
            # Close OpenCV windows
            cv2.destroyAllWindows()
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def stop(self):
        """
        Stop the application
        """
        self.is_running = False


async def main():
    """
    Main entry point for the application
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Real-time Vision Analysis with OpenAI")
    parser.add_argument("--config", type=str, 
                       help="Path to configuration file")
    parser.add_argument("--realtime", action="store_true", 
                       help="Use OpenAI Realtime API with voice interaction (recommended)")
    parser.add_argument("--voice", action="store_true", 
                       help="Enable voice interaction (same as --realtime)")
    parser.add_argument("--vision-only", action="store_true",
                       help="Use vision-only mode without voice")
    parser.add_argument("--camera-index", type=int,
                       help="Camera device index")
    parser.add_argument("--frame-interval", type=int,
                       help="Send camera frame every N seconds")
    parser.add_argument("--no-video", action="store_true",
                       help="Don't show video window")
    parser.add_argument("--log-level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help="Set logging level")
    parser.add_argument("--list-devices", action="store_true",
                       help="List available audio/video devices and exit")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = Config(args.config)
        
        # Override config with command line arguments
        if args.realtime:
            config.set('use_realtime_api', True)
        if args.camera_index is not None:
            config.set('camera_index', args.camera_index)
        if args.frame_interval is not None:
            config.set('frame_interval', args.frame_interval)
        if args.no_video:
            config.set('show_video', False)
        if args.log_level:
            config.set('log_level', args.log_level)
        
        # Configure logging
        log_level = getattr(logging, config.get('log_level', 'INFO').upper())
        logging.getLogger().setLevel(log_level)
        
        # Validate API key
        if not config.get('api_key'):
            print("❌ Error: OPENAI_API_KEY not found in environment variables")
            print("Please set your API key in the .env file")
            return 1
            
        # Create and initialize application
        app = RealtimeVisionApp(config)
        
        # Setup signal handlers for graceful shutdown
        app.setup_signal_handlers()
        
        # Initialize application
        if not await app.initialize():
            print("❌ Failed to initialize application")
            return 1
        
        # Run application
        await app.run()
        
        print("\n✅ Application terminated successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    try:
        # Run the async main function
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)