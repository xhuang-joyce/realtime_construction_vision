import asyncio
import time
import threading
import logging
from typing import Optional, Callable
from datetime import datetime
import queue
import cv2

from .camera_capture import CameraCapture
from .microphone_capture import MicrophoneCapture
from .openai_realtime import OpenAIRealtimeClient
from .audio_handler import AudioResponseHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalProcessor:
    """
    Multimodal processor that handles both video (camera) and audio (microphone) input
    for real-time interaction with OpenAI's Realtime API
    """
    
    def __init__(self, 
                 api_key: str,
                 frame_interval: int = 3,
                 image_on_speech: bool = True):
        """
        Initialize multimodal processor
        
        Args:
            api_key: OpenAI API key
            frame_interval: Send camera image every N seconds
            image_on_speech: Send current camera frame when user speaks
        """
        self.api_key = api_key
        self.frame_interval = frame_interval
        self.image_on_speech = image_on_speech
        
        # Initialize OpenAI Realtime client
        self.ai_client = OpenAIRealtimeClient(api_key)
        
        # Audio buffer for collecting speech
        self.audio_buffer = []
        self.is_collecting_audio = False
        self.last_image_time = 0
        
        # Callbacks for responses
        self.on_text_response: Optional[Callable[[str], None]] = None
        self.on_audio_response: Optional[Callable[[bytes], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_transcript: Optional[Callable[[str], None]] = None
        
        # Statistics
        self.total_audio_sent = 0
        self.total_images_sent = 0
        self.conversation_turns = 0
        
    def set_callbacks(self,
                     text_callback: Optional[Callable[[str], None]] = None,
                     audio_callback: Optional[Callable[[bytes], None]] = None,
                     error_callback: Optional[Callable[[str], None]] = None,
                     transcript_callback: Optional[Callable[[str], None]] = None):
        """Set callbacks for different types of responses"""
        if text_callback:
            self.on_text_response = text_callback
            self.ai_client.set_text_callback(text_callback)
        
        if audio_callback:
            self.on_audio_response = audio_callback
            self.ai_client.set_audio_callback(audio_callback)
        
        if error_callback:
            self.on_error = error_callback
            self.ai_client.set_error_callback(error_callback)
        
        if transcript_callback:
            self.on_transcript = transcript_callback
    
    async def start_processing(self) -> bool:
        """
        Start the multimodal processing
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        try:
            # Connect to OpenAI Realtime API
            if not await self.ai_client.connect():
                logger.error("Failed to connect to OpenAI Realtime API")
                return False
            
            logger.info("Multimodal processing started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting multimodal processing: {e}")
            if self.on_error:
                self.on_error(str(e))
            return False
    
    def handle_audio_data(self, audio_data: bytes):
        """
        Handle incoming audio data from microphone
        
        Args:
            audio_data: Raw audio data from microphone
        """
        try:
            if self.is_collecting_audio:
                self.audio_buffer.append(audio_data)
            
            # Send audio data in real-time to OpenAI
            asyncio.create_task(self.ai_client.send_audio_data(audio_data))
            
        except Exception as e:
            logger.error(f"Error handling audio data: {e}")
            if self.on_error:
                self.on_error(str(e))
    
    def handle_speech_start(self):
        """Handle when user starts speaking"""
        try:
            logger.info("👄 User started speaking")
            self.is_collecting_audio = True
            self.audio_buffer = []
            
            # Optional: Show some UI feedback that user is speaking
            
        except Exception as e:
            logger.error(f"Error handling speech start: {e}")
    
    def handle_speech_end(self, current_frame=None):
        """
        Handle when user stops speaking
        
        Args:
            current_frame: Current camera frame to send with speech
        """
        try:
            logger.info("🔇 User stopped speaking")
            self.is_collecting_audio = False
            
            # Commit audio input to trigger processing
            asyncio.create_task(self.ai_client.commit_audio_input())
            
            # Send current camera frame if enabled and available
            if self.image_on_speech and current_frame is not None:
                asyncio.create_task(self._send_current_frame(current_frame))
            
            self.conversation_turns += 1
            
        except Exception as e:
            logger.error(f"Error handling speech end: {e}")
            if self.on_error:
                self.on_error(str(e))
    
    async def _send_current_frame(self, frame):
        """
        Send current camera frame to AI
        
        Args:
            frame: OpenCV frame to send
        """
        try:
            from camera_capture import CameraCapture
            
            # Convert frame to base64
            camera = CameraCapture()  # Temporary instance for conversion
            base64_frame = camera.frame_to_base64(frame)
            
            if base64_frame:
                # Send image with context about the conversation
                await self.ai_client.send_combined_input(
                    base64_image=base64_frame,
                    text_prompt="Here's what I can see right now while we're talking."
                )
                
                self.total_images_sent += 1
                self.last_image_time = time.time()
                
                logger.info("📸 Sent camera frame with voice interaction")
            
        except Exception as e:
            logger.error(f"Error sending current frame: {e}")
    
    async def send_periodic_image(self, frame):
        """
        Send periodic camera updates even without speech
        
        Args:
            frame: Current camera frame
        """
        try:
            current_time = time.time()
            
            # Send image at regular intervals
            if current_time - self.last_image_time >= self.frame_interval:
                from camera_capture import CameraCapture
                
                camera = CameraCapture()
                base64_frame = camera.frame_to_base64(frame)
                
                if base64_frame:
                    await self.ai_client.send_combined_input(
                        base64_image=base64_frame,
                        text_prompt="This is what I can see right now. Let me know if you notice anything interesting or if you have questions about what's in the scene."
                    )
                    
                    self.total_images_sent += 1
                    self.last_image_time = current_time
                    
                    logger.info("📸 Sent periodic camera update")
        
        except Exception as e:
            logger.error(f"Error sending periodic image: {e}")
    
    def stop_processing(self):
        """
        Stop multimodal processing
        """
        try:
            asyncio.create_task(self.ai_client.disconnect())
            logger.info("Multimodal processing stopped")
            
        except Exception as e:
            logger.error(f"Error stopping multimodal processing: {e}")
    
    def get_statistics(self) -> dict:
        """
        Get processing statistics
        
        Returns:
            dict: Statistics about multimodal processing
        """
        return {
            'total_audio_sent': self.total_audio_sent,
            'total_images_sent': self.total_images_sent,
            'conversation_turns': self.conversation_turns,
            'is_collecting_audio': self.is_collecting_audio,
            'last_image_time': datetime.fromtimestamp(self.last_image_time).isoformat() if self.last_image_time else None
        }


class RealtimeMultimodalPipeline:
    """
    Complete pipeline that combines camera, microphone, and AI processing
    """
    
    def __init__(self, 
                 api_key: str,
                 camera_index: int = 0,
                 microphone_index: int = None,
                 frame_interval: int = 3,
                 vad_threshold: int = 500):
        """
        Initialize complete multimodal pipeline
        
        Args:
            api_key: OpenAI API key
            camera_index: Camera device index
            microphone_index: Microphone device index (None for default)
            frame_interval: Send camera frame every N seconds
            vad_threshold: Voice Activity Detection threshold
        """
        self.api_key = api_key
        self.camera_index = camera_index
        self.microphone_index = microphone_index
        self.frame_interval = frame_interval
        self.vad_threshold = vad_threshold
        
        # Initialize components
        self.camera = CameraCapture(camera_index=camera_index)
        self.microphone = MicrophoneCapture()
        self.processor = MultimodalProcessor(api_key, frame_interval)
        self.audio_handler = AudioResponseHandler(api_key)
        
        self.is_running = False
        self.pipeline_thread = None
        
        # Set up microphone callbacks
        self.microphone.set_audio_callback(self.processor.handle_audio_data)
        self.microphone.set_speech_callbacks(
            on_start=self.processor.handle_speech_start,
            on_end=self._handle_speech_end_with_frame
        )
        self.microphone.set_vad_threshold(vad_threshold)
        
    def _handle_speech_end_with_frame(self):
        """Handle speech end with current camera frame"""
        frame = self.camera.get_current_frame()
        self.processor.handle_speech_end(frame)
    
    def set_callbacks(self,
                     text_callback: Optional[Callable[[str], None]] = None,
                     audio_callback: Optional[Callable[[bytes], None]] = None,
                     error_callback: Optional[Callable[[str], None]] = None,
                     transcript_callback: Optional[Callable[[str], None]] = None):
        """Set callbacks for handling responses"""
        
        # Set up audio handler callbacks
        if audio_callback:
            self.audio_handler.handle_audio_response = audio_callback
        
        # Set up processor callbacks
        self.processor.set_callbacks(text_callback, audio_callback, error_callback, transcript_callback)
    
    async def start_pipeline(self) -> bool:
        """
        Start the complete multimodal pipeline
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        try:
            # Start camera
            if not self.camera.start_capture():
                logger.error("Failed to start camera")
                return False
            
            # Start microphone
            if not self.microphone.start_capture(self.microphone_index):
                logger.error("Failed to start microphone")
                self.camera.stop_capture()
                return False
            
            # Start audio handler
            if not self.audio_handler.start():
                logger.warning("Failed to start audio handler, continuing without audio output")
            
            # Start multimodal processor
            if not await self.processor.start_processing():
                logger.error("Failed to start multimodal processor")
                self.microphone.stop_capture()
                self.camera.stop_capture()
                return False
            
            # Connect audio handler to processor
            self.processor.set_callbacks(
                audio_callback=self.audio_handler.handle_audio_response
            )
            
            # Start pipeline thread for periodic tasks
            self.is_running = True
            self.pipeline_thread = threading.Thread(target=self._pipeline_loop, daemon=True)
            self.pipeline_thread.start()
            
            logger.info("🚀 Complete multimodal pipeline started successfully!")
            logger.info("🎤 You can now speak to the AI while it watches through your camera")
            return True
            
        except Exception as e:
            logger.error(f"Error starting multimodal pipeline: {e}")
            return False
    
    def _pipeline_loop(self):
        """
        Main pipeline loop for periodic tasks
        """
        while self.is_running:
            try:
                # Get current frame
                frame = self.camera.get_current_frame()
                
                if frame is not None:
                    # Send periodic image updates
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(self.processor.send_periodic_image(frame))
                    finally:
                        loop.close()
                
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in pipeline loop: {e}")
    
    def stop_pipeline(self):
        """
        Stop the complete multimodal pipeline
        """
        logger.info("Stopping multimodal pipeline...")
        
        self.is_running = False
        
        # Wait for pipeline thread
        if self.pipeline_thread and self.pipeline_thread.is_alive():
            self.pipeline_thread.join(timeout=3.0)
        
        # Stop components
        self.processor.stop_processing()
        self.microphone.stop_capture()
        self.microphone.cleanup()
        self.camera.stop_capture()
        self.audio_handler.stop()
        
        logger.info("Multimodal pipeline stopped")
    
    def get_status(self) -> dict:
        """
        Get complete pipeline status
        
        Returns:
            dict: Status information
        """
        return {
            'is_running': self.is_running,
            'camera_available': self.camera.is_camera_available(),
            'microphone_stats': self.microphone.get_statistics(),
            'processor_stats': self.processor.get_statistics(),
            'audio_handler_stats': self.audio_handler.get_statistics()
        }