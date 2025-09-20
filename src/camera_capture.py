import cv2
import numpy as np
import base64
import threading
import time
from typing import Optional, Callable
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraCapture:
    """
    Real-time camera capture class for processing video frames
    """
    
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize camera capture
        
        Args:
            camera_index: Camera device index (usually 0 for default camera)
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.capture_thread = None
        
    def start_capture(self) -> bool:
        """
        Start camera capture in a separate thread
        
        Returns:
            bool: True if capture started successfully, False otherwise
        """
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Start capture thread
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            logger.info(f"Camera capture started successfully on device {self.camera_index}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera capture: {e}")
            return False
    
    def _capture_loop(self):
        """
        Main capture loop running in separate thread
        """
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                else:
                    logger.warning("Failed to read frame from camera")
                    
                # Control frame rate
                time.sleep(1.0 / self.fps)
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                break
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Get the most recent frame
        
        Returns:
            numpy.ndarray: Current frame or None if no frame available
        """
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def frame_to_base64(self, frame: np.ndarray, quality: int = 80) -> str:
        """
        Convert frame to base64 encoded JPEG
        
        Args:
            frame: OpenCV frame (numpy array)
            quality: JPEG quality (1-100)
            
        Returns:
            str: Base64 encoded image string
        """
        try:
            # Encode frame as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            
            # Convert to base64
            base64_str = base64.b64encode(buffer).decode('utf-8')
            return base64_str
            
        except Exception as e:
            logger.error(f"Error converting frame to base64: {e}")
            return ""
    
    def get_frame_base64(self, quality: int = 80) -> Optional[str]:
        """
        Get current frame as base64 encoded string
        
        Args:
            quality: JPEG quality (1-100)
            
        Returns:
            str: Base64 encoded frame or None if no frame available
        """
        frame = self.get_current_frame()
        if frame is not None:
            return self.frame_to_base64(frame, quality)
        return None
    
    def display_frame(self, window_name: str = "Camera Feed"):
        """
        Display current frame in OpenCV window
        
        Args:
            window_name: Name of the display window
        """
        frame = self.get_current_frame()
        if frame is not None:
            cv2.imshow(window_name, frame)
    
    def stop_capture(self):
        """
        Stop camera capture and cleanup resources
        """
        self.is_running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            
        cv2.destroyAllWindows()
        logger.info("Camera capture stopped")
    
    def is_camera_available(self) -> bool:
        """
        Check if camera is available and working
        
        Returns:
            bool: True if camera is available, False otherwise
        """
        return self.cap is not None and self.cap.isOpened() and self.is_running


# Example usage and testing
if __name__ == "__main__":
    # Test camera capture
    camera = CameraCapture(camera_index=0, width=640, height=480)
    
    if camera.start_capture():
        print("Camera capture started. Press 'q' to quit.")
        
        try:
            while True:
                # Display frame
                camera.display_frame("Test Camera Feed")
                
                # Check for quit command
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                # Test base64 conversion every 2 seconds
                if int(time.time()) % 2 == 0:
                    base64_frame = camera.get_frame_base64()
                    if base64_frame:
                        print(f"Frame converted to base64 (length: {len(base64_frame)})")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nReceived interrupt signal")
        
        finally:
            camera.stop_capture()
    else:
        print("Failed to start camera capture")