import asyncio
import time
import threading
import logging
from typing import Optional, Callable
from datetime import datetime
import queue
import cv2

from .camera_capture import CameraCapture
from .openai_realtime import OpenAIRealtimeClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameProcessor:
    """
    Pipeline for processing video frames and sending them to OpenAI for analysis
    """
    
    def __init__(self, 
                 api_key: str,
                 frame_interval: int = 2,
                 use_realtime_api: bool = False,
                 prompt: str = "describe what is in front of you"):
        """
        Initialize frame processor
        
        Args:
            api_key: OpenAI API key
            frame_interval: Process every N frames to reduce API calls
            use_realtime_api: Whether to use realtime API or standard vision API
            prompt: Prompt to send with each frame
        """
        self.api_key = api_key
        self.frame_interval = frame_interval
        self.use_realtime_api = use_realtime_api
        self.prompt = prompt
        
        # Initialize OpenAI client
        if use_realtime_api:
            self.ai_client = OpenAIRealtimeClient(api_key)
        else:
            self.ai_client = VisionAnalysisClient(api_key)
        
        # Store reference to the main event loop for cross-thread async calls
        try:
            self.main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self.main_loop = None
        
        # Frame processing state
        self.frame_count = 0
        self.last_processed_time = 0
        self.processing_queue = queue.Queue(maxsize=5)  # Limit queue size to prevent memory issues
        self.is_processing = False
        self.processing_thread = None
        
        # Callbacks for responses
        self.on_text_response: Optional[Callable[[str], None]] = None
        self.on_audio_response: Optional[Callable[[bytes], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        
        # Statistics
        self.total_frames_processed = 0
        self.total_api_calls = 0
        self.last_analysis_time = None
        
    def set_text_callback(self, callback: Callable[[str], None]):
        """Set callback for text responses"""
        self.on_text_response = callback
        if self.use_realtime_api:
            self.ai_client.set_text_callback(callback)
    
    def set_audio_callback(self, callback: Callable[[bytes], None]):
        """Set callback for audio responses"""
        self.on_audio_response = callback
        if self.use_realtime_api:
            self.ai_client.set_audio_callback(callback)
    
    def set_error_callback(self, callback: Callable[[str], None]):
        """Set callback for error handling"""
        self.on_error = callback
        if self.use_realtime_api:
            self.ai_client.set_error_callback(callback)
    
    async def start_processing(self) -> bool:
        """
        Start the frame processing pipeline
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        try:
            # Connect to realtime API if using it
            if self.use_realtime_api:
                if not await self.ai_client.connect():
                    logger.error("Failed to connect to OpenAI Realtime API")
                    return False
            
            # Start processing thread
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            logger.info("Frame processing pipeline started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting frame processing: {e}")
            if self.on_error:
                self.on_error(str(e))
            return False
    
    def process_frame(self, frame, force_process: bool = False):
        """
        Queue a frame for processing
        
        Args:
            frame: OpenCV frame (numpy array)
            force_process: Force processing regardless of frame interval
        """
        try:
            self.frame_count += 1
            
            # Check if we should process this frame
            should_process = (
                force_process or 
                self.frame_count % self.frame_interval == 0 or
                time.time() - self.last_processed_time > 3.0  # Process at least every 3 seconds
            )
            
            if should_process and not self.processing_queue.full():
                # Convert frame to base64
                from camera_capture import CameraCapture
                camera = CameraCapture()
                base64_frame = camera.frame_to_base64(frame)
                
                if base64_frame:
                    timestamp = datetime.now()
                    self.processing_queue.put({
                        'base64_frame': base64_frame,
                        'timestamp': timestamp,
                        'frame_number': self.frame_count
                    })
                    self.last_processed_time = time.time()
                    
        except Exception as e:
            logger.error(f"Error queuing frame for processing: {e}")
            if self.on_error:
                self.on_error(str(e))
    
    def _processing_loop(self):
        """
        Main processing loop running in separate thread
        """
        while self.is_processing:
            try:
                # Get frame from queue (blocking with timeout)
                try:
                    frame_data = self.processing_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the frame
                if self.use_realtime_api:
                    # For realtime API, we need to schedule this in the main event loop
                    # Convert to sync processing with the realtime client
                    self._process_frame_realtime_sync(frame_data)
                else:
                    self._process_frame_standard(frame_data)
                
                self.total_frames_processed += 1
                self.total_api_calls += 1
                self.last_analysis_time = frame_data['timestamp']
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                if self.on_error:
                    self.on_error(str(e))
    
    def _process_frame_realtime_sync(self, frame_data):
        """
        Process frame using OpenAI Realtime API (synchronous wrapper)
        
        Args:
            frame_data: Dictionary containing frame data and metadata
        """
        try:
            base64_frame = frame_data['base64_frame']
            logger.info(f"Processing frame {frame_data['frame_number']} with Realtime API")
            
            # Schedule the coroutine in the main event loop
            if self.main_loop and self.main_loop.is_running():
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self.ai_client.send_image_for_analysis(base64_frame, self.prompt),
                        self.main_loop
                    )
                    logger.debug("Scheduled image analysis in main event loop")
                except Exception as e:
                    logger.error(f"Error scheduling async work: {e}")
            else:
                logger.warning("Main event loop not available for realtime processing")
            
        except Exception as e:
            logger.error(f"Error processing frame with Realtime API: {e}")
            if self.on_error:
                self.on_error(str(e))

    async def _process_frame_realtime(self, frame_data):
        """
        Process frame using OpenAI Realtime API
        
        Args:
            frame_data: Dictionary containing frame data and metadata
        """
        try:
            base64_frame = frame_data['base64_frame']
            logger.info(f"Processing frame {frame_data['frame_number']} with Realtime API")
            
            await self.ai_client.send_image_for_analysis(base64_frame, self.prompt)
            
        except Exception as e:
            logger.error(f"Error processing frame with Realtime API: {e}")
            if self.on_error:
                self.on_error(str(e))
    
    def _process_frame_standard(self, frame_data):
        """
        Process frame using standard OpenAI Vision API
        
        Args:
            frame_data: Dictionary containing frame data and metadata
        """
        try:
            base64_frame = frame_data['base64_frame']
            logger.info(f"Processing frame {frame_data['frame_number']} with Vision API")
            
            # Analyze image
            result = self.ai_client.analyze_image(base64_frame, self.prompt)
            
            # Call text callback
            if result and self.on_text_response:
                self.on_text_response(result)
                
        except Exception as e:
            logger.error(f"Error processing frame with Vision API: {e}")
            if self.on_error:
                self.on_error(str(e))
    
    def stop_processing(self):
        """
        Stop the frame processing pipeline
        """
        self.is_processing = False
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        # Disconnect from realtime API if connected
        if self.use_realtime_api:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.ai_client.disconnect())
            except Exception as e:
                logger.error(f"Error disconnecting from realtime API: {e}")
            finally:
                loop.close()
        
        logger.info("Frame processing pipeline stopped")
    
    def get_statistics(self) -> dict:
        """
        Get processing statistics
        
        Returns:
            dict: Statistics about frame processing
        """
        return {
            'total_frames_seen': self.frame_count,
            'total_frames_processed': self.total_frames_processed,
            'total_api_calls': self.total_api_calls,
            'queue_size': self.processing_queue.qsize(),
            'last_analysis_time': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            'processing_rate': self.total_frames_processed / max(self.frame_count, 1) * 100
        }


class RealtimeVisionPipeline:
    """
    Complete pipeline combining camera capture and frame processing
    """
    
    def __init__(self, 
                 api_key: str,
                 camera_index: int = 0,
                 frame_interval: int = 2,
                 use_realtime_api: bool = False):
        """
        Initialize complete vision pipeline
        
        Args:
            api_key: OpenAI API key
            camera_index: Camera device index
            frame_interval: Process every N frames
            use_realtime_api: Whether to use realtime API
        """
        self.camera = CameraCapture(camera_index=camera_index)
        self.processor = FrameProcessor(
            api_key=api_key,
            frame_interval=frame_interval,
            use_realtime_api=use_realtime_api
        )
        
        self.is_running = False
        self.pipeline_thread = None
        
        # Response handlers
        self.text_responses = []
        self.audio_responses = []
        
    def set_callbacks(self, 
                     text_callback: Optional[Callable[[str], None]] = None,
                     audio_callback: Optional[Callable[[bytes], None]] = None,
                     error_callback: Optional[Callable[[str], None]] = None):
        """
        Set callbacks for handling responses
        """
        if text_callback:
            self.processor.set_text_callback(text_callback)
        if audio_callback:
            self.processor.set_audio_callback(audio_callback)
        if error_callback:
            self.processor.set_error_callback(error_callback)
    
    async def start_pipeline(self) -> bool:
        """
        Start the complete vision pipeline
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        try:
            # Start camera capture
            if not self.camera.start_capture():
                return False
            
            # Start frame processing
            if not await self.processor.start_processing():
                self.camera.stop_capture()
                return False
            
            # Start pipeline thread
            self.is_running = True
            self.pipeline_thread = threading.Thread(target=self._pipeline_loop, daemon=True)
            self.pipeline_thread.start()
            
            logger.info("Complete vision pipeline started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting pipeline: {e}")
            return False
    
    def _pipeline_loop(self):
        """
        Main pipeline loop - captures frames and sends them for processing
        """
        while self.is_running:
            try:
                # Get current frame
                frame = self.camera.get_current_frame()
                if frame is not None:
                    # Send frame for processing
                    self.processor.process_frame(frame)
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                logger.error(f"Error in pipeline loop: {e}")
    
    def stop_pipeline(self):
        """
        Stop the complete vision pipeline
        """
        self.is_running = False
        
        # Wait for pipeline thread
        if self.pipeline_thread and self.pipeline_thread.is_alive():
            self.pipeline_thread.join(timeout=3.0)
        
        # Stop components
        self.processor.stop_processing()
        self.camera.stop_capture()
        
        logger.info("Complete vision pipeline stopped")
    
    def get_status(self) -> dict:
        """
        Get pipeline status and statistics
        
        Returns:
            dict: Status information
        """
        return {
            'is_running': self.is_running,
            'camera_available': self.camera.is_camera_available(),
            'processing_stats': self.processor.get_statistics()
        }