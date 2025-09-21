"""
Screen Capture Module

Captures the computer screen in real-time, similar to camera_capture.py but for desktop screen sharing.
Supports full screen capture and specific window capture.
"""

import threading
import time
import base64
import cv2
import numpy as np
import logging
from typing import Optional, Tuple

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    logging.warning("mss not available, screen capture will use basic methods")

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    logging.warning("pyautogui not available, using alternative screen capture")


class ScreenCapture:
    """Screen capture class for real-time desktop screen sharing."""
    
    def __init__(self, capture_region: Optional[Tuple[int, int, int, int]] = None, fps: int = 15):
        """
        Initialize screen capture.
        
        Args:
            capture_region: (x, y, width, height) to capture specific region, None for full screen
            fps: Target frames per second for capture
        """
        self.capture_region = capture_region
        self.fps = fps
        self.frame_interval = 1.0 / fps
        
        self.current_frame = None
        self.is_capturing = False
        self.capture_thread = None
        self.frame_lock = threading.Lock()
        
        # Initialize screen capture method
        self.sct = None
        self._thread_sct = None
        self._mss_failed = False
        
        # Try mss first, but be prepared to fall back to pyautogui
        if MSS_AVAILABLE:
            try:
                self.sct = mss.mss()
                self.capture_method = "mss"
                # Test if mss works with a quick capture
                test_monitor = self.sct.monitors[1]
                test_shot = self.sct.grab(test_monitor)
                logging.info("MSS initialization successful")
            except Exception as e:
                logging.warning(f"MSS initialization failed: {e}, falling back to pyautogui")
                self.sct = None
                if PYAUTOGUI_AVAILABLE:
                    self.capture_method = "pyautogui"
                else:
                    raise RuntimeError("MSS failed and pyautogui not available")
        elif PYAUTOGUI_AVAILABLE:
            self.capture_method = "pyautogui"
        else:
            raise RuntimeError("No screen capture library available. Install mss or pyautogui.")
            
        self.screen_size = self._get_screen_size()
        
    def _get_screen_size(self) -> Tuple[int, int]:
        """Get screen dimensions."""
        if self.capture_method == "mss" and self.sct:
            monitor = self.sct.monitors[0]  # Primary monitor
            return monitor["width"], monitor["height"]
        elif self.capture_method == "pyautogui":
            return pyautogui.size()
        else:
            return (1920, 1080)  # Default fallback
    
    def _capture_screen_mss(self) -> Optional[np.ndarray]:
        """Capture screen using mss library."""
        try:
            # Create a new mss instance for this thread if needed
            if not hasattr(self, '_thread_sct') or self._thread_sct is None:
                import mss
                self._thread_sct = mss.mss()
            
            if self.capture_region:
                x, y, w, h = self.capture_region
                monitor = {"top": y, "left": x, "width": w, "height": h}
            else:
                monitor = self._thread_sct.monitors[1]  # Primary monitor (index 1, not 0)
            
            screenshot = self._thread_sct.grab(monitor)
            # Convert to numpy array and BGR format (OpenCV format)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        except Exception as e:
            logging.error(f"Error capturing screen with mss: {e}")
            # If mss fails, try to fall back to pyautogui
            if hasattr(self, '_mss_failed'):
                return None
            self._mss_failed = True
            logging.warning("MSS failed, attempting to switch to pyautogui for this capture")
            return self._capture_screen_pyautogui()
    
    def _capture_screen_pyautogui(self) -> Optional[np.ndarray]:
        """Capture screen using pyautogui library."""
        try:
            if self.capture_region:
                x, y, w, h = self.capture_region
                screenshot = pyautogui.screenshot(region=(x, y, w, h))
            else:
                screenshot = pyautogui.screenshot()
            
            # Convert PIL image to OpenCV format
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            return frame
        except Exception as e:
            logging.error(f"Error capturing screen with pyautogui: {e}")
            return None
    
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        consecutive_failures = 0
        max_failures = 5
        
        while self.is_capturing:
            start_time = time.time()
            
            # Capture screen based on available method
            if self.capture_method == "mss" and not self._mss_failed:
                frame = self._capture_screen_mss()
                # If mss fails multiple times, switch to pyautogui permanently
                if frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures and PYAUTOGUI_AVAILABLE:
                        logging.warning(f"MSS failed {max_failures} times, switching to pyautogui permanently")
                        self.capture_method = "pyautogui"
                        consecutive_failures = 0
                else:
                    consecutive_failures = 0
            else:
                frame = self._capture_screen_pyautogui()
            
            if frame is not None:
                with self.frame_lock:
                    self.current_frame = frame.copy()
            
            # Control frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, self.frame_interval - elapsed)
            time.sleep(sleep_time)
    
    def start_capture(self):
        """Start screen capture in background thread."""
        if self.is_capturing:
            return
        
        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logging.info(f"Screen capture started using {self.capture_method}")
    
    def stop_capture(self):
        """Stop screen capture and cleanup."""
        self.is_capturing = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        # Clean up mss instances
        if self.sct:
            try:
                self.sct.close()
            except Exception:
                pass
        
        if hasattr(self, '_thread_sct') and self._thread_sct:
            try:
                self._thread_sct.close()
            except Exception:
                pass
        
        cv2.destroyAllWindows()
        logging.info("Screen capture stopped")
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame."""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def frame_to_base64(self, frame: np.ndarray, quality: int = 80) -> str:
        """Convert frame to base64 encoded JPEG."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        return base64.b64encode(buffer).decode('utf-8')
    
    def get_frame_base64(self, quality: int = 80) -> Optional[str]:
        """Get current frame as base64 encoded JPEG."""
        frame = self.get_current_frame()
        if frame is not None:
            return self.frame_to_base64(frame, quality)
        return None
    
    def display_frame(self, window_name: str = "Screen Capture"):
        """Display current frame in OpenCV window."""
        frame = self.get_current_frame()
        if frame is not None:
            # Resize if frame is too large for display
            height, width = frame.shape[:2]
            if width > 1280 or height > 720:
                scale = min(1280/width, 720/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            cv2.imshow(window_name, frame)
    
    def is_screen_available(self) -> bool:
        """Check if screen capture is available."""
        return MSS_AVAILABLE or PYAUTOGUI_AVAILABLE
    
    def get_capture_info(self) -> dict:
        """Get information about current capture setup."""
        return {
            "method": self.capture_method,
            "screen_size": self.screen_size,
            "capture_region": self.capture_region,
            "fps": self.fps,
            "is_capturing": self.is_capturing
        }


def main():
    """Test screen capture functionality."""
    print("Testing screen capture...")
    
    screen_cap = ScreenCapture(fps=10)
    
    if not screen_cap.is_screen_available():
        print("Screen capture not available. Install 'mss' or 'pyautogui'.")
        return
    
    print(f"Screen capture info: {screen_cap.get_capture_info()}")
    
    screen_cap.start_capture()
    window_name = "Screen Capture Test - press 'q' to quit"
    
    try:
        print("Showing screen capture. Press 'q' in the window to quit, or Ctrl+C in terminal.")
        while True:
            screen_cap.display_frame(window_name)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        screen_cap.stop_capture()


if __name__ == "__main__":
    main()