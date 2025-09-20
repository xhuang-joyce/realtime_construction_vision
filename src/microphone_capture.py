import pyaudio
import numpy as np
import threading
import queue
import time
import logging
from typing import Optional, Callable
import wave
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MicrophoneCapture:
    """
    Real-time microphone capture for voice input to OpenAI Realtime API
    """
    
    def __init__(self, 
                 sample_rate: int = 24000,
                 channels: int = 1,
                 chunk_size: int = 1024,
                 format_type: int = pyaudio.paInt16):
        """
        Initialize microphone capture
        
        Args:
            sample_rate: Audio sample rate (24000 Hz for OpenAI Realtime API)
            channels: Number of audio channels (1 for mono)
            chunk_size: Size of audio chunks to capture
            format_type: PyAudio format type
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format_type = format_type
        
        # PyAudio setup
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        
        # Capture state
        self.is_recording = False
        self.capture_thread = None
        self.audio_queue = queue.Queue()
        
        # Voice Activity Detection (simple threshold-based)
        self.vad_threshold = 500  # Adjust based on your microphone sensitivity
        self.vad_enabled = True
        self.silence_duration = 2.0  # Seconds of silence before stopping speech
        self.last_speech_time = 0
        
        # Callbacks
        self.on_audio_data: Optional[Callable[[bytes], None]] = None
        self.on_speech_start: Optional[Callable[[], None]] = None
        self.on_speech_end: Optional[Callable[[], None]] = None
        
        # Statistics
        self.total_chunks_captured = 0
        self.total_bytes_captured = 0
        self.speech_segments = 0
        
    def list_audio_devices(self):
        """
        List available audio input devices
        
        Returns:
            List of tuples (device_index, device_name, channels)
        """
        devices = []
        info = self.pyaudio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        for i in range(num_devices):
            device_info = self.pyaudio.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                devices.append((
                    i,
                    device_info.get('name'),
                    device_info.get('maxInputChannels')
                ))
        
        return devices
    
    def start_capture(self, device_index: Optional[int] = None) -> bool:
        """
        Start microphone capture
        
        Args:
            device_index: Audio device index to use (None for default)
            
        Returns:
            bool: True if capture started successfully, False otherwise
        """
        try:
            # Open audio stream
            self.stream = self.pyaudio.open(
                format=self.format_type,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size
            )
            
            # Start capture thread
            self.is_recording = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            logger.info(f"Microphone capture started (Rate: {self.sample_rate}Hz, Device: {device_index or 'default'})")
            return True
            
        except Exception as e:
            logger.error(f"Error starting microphone capture: {e}")
            return False
    
    def _capture_loop(self):
        """
        Main capture loop running in separate thread
        """
        is_speaking = False
        silence_start = None
        
        while self.is_recording:
            try:
                # Read audio data
                audio_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                
                # Convert to numpy array for analysis
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                
                # Simple Voice Activity Detection
                audio_level = np.abs(audio_np).mean()
                current_time = time.time()
                
                if self.vad_enabled:
                    if audio_level > self.vad_threshold:
                        # Speech detected
                        if not is_speaking:
                            is_speaking = True
                            silence_start = None
                            self.speech_segments += 1
                            logger.debug("Speech started")
                            if self.on_speech_start:
                                self.on_speech_start()
                        
                        self.last_speech_time = current_time
                        
                        # Send audio data when speaking
                        if self.on_audio_data:
                            self.on_audio_data(audio_data)
                    
                    else:
                        # Silence detected
                        if is_speaking:
                            if silence_start is None:
                                silence_start = current_time
                            elif current_time - silence_start > self.silence_duration:
                                is_speaking = False
                                silence_start = None
                                logger.debug("Speech ended")
                                if self.on_speech_end:
                                    self.on_speech_end()
                else:
                    # VAD disabled, always send audio
                    if self.on_audio_data:
                        self.on_audio_data(audio_data)
                
                # Update statistics
                self.total_chunks_captured += 1
                self.total_bytes_captured += len(audio_data)
                
            except Exception as e:
                logger.error(f"Error in microphone capture loop: {e}")
                break
    
    def stop_capture(self):
        """
        Stop microphone capture and cleanup resources
        """
        self.is_recording = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        logger.info("Microphone capture stopped")
    
    def set_audio_callback(self, callback: Callable[[bytes], None]):
        """
        Set callback for audio data
        
        Args:
            callback: Function to call when audio data is available
        """
        self.on_audio_data = callback
    
    def set_speech_callbacks(self, 
                           on_start: Optional[Callable[[], None]] = None,
                           on_end: Optional[Callable[[], None]] = None):
        """
        Set callbacks for speech start/end events
        
        Args:
            on_start: Function to call when speech starts
            on_end: Function to call when speech ends
        """
        self.on_speech_start = on_start
        self.on_speech_end = on_end
    
    def set_vad_threshold(self, threshold: int):
        """
        Set Voice Activity Detection threshold
        
        Args:
            threshold: Audio level threshold for speech detection
        """
        self.vad_threshold = threshold
        logger.info(f"VAD threshold set to {threshold}")
    
    def enable_vad(self, enabled: bool = True):
        """
        Enable or disable Voice Activity Detection
        
        Args:
            enabled: Whether to enable VAD
        """
        self.vad_enabled = enabled
        logger.info(f"VAD {'enabled' if enabled else 'disabled'}")
    
    def get_audio_level(self) -> float:
        """
        Get current audio level for monitoring
        
        Returns:
            float: Current audio level
        """
        if not self.is_recording or not self.stream:
            return 0.0
        
        try:
            # Read a small chunk to check level
            audio_data = self.stream.read(256, exception_on_overflow=False)
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            return float(np.abs(audio_np).mean())
        except:
            return 0.0
    
    def get_statistics(self) -> dict:
        """
        Get capture statistics
        
        Returns:
            dict: Statistics about audio capture
        """
        return {
            'is_recording': self.is_recording,
            'total_chunks_captured': self.total_chunks_captured,
            'total_bytes_captured': self.total_bytes_captured,
            'speech_segments': self.speech_segments,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'vad_enabled': self.vad_enabled,
            'vad_threshold': self.vad_threshold
        }
    
    def cleanup(self):
        """
        Cleanup PyAudio resources
        """
        if self.stream:
            self.stream.close()
        self.pyaudio.terminate()


# Example usage and testing
if __name__ == "__main__":
    def test_microphone():
        """Test microphone capture functionality"""
        mic = MicrophoneCapture()
        
        # List available devices
        print("Available audio input devices:")
        devices = mic.list_audio_devices()
        for idx, name, channels in devices:
            print(f"  {idx}: {name} ({channels} channels)")
        
        # Set up callbacks
        def on_audio(audio_data):
            print(f"Audio chunk: {len(audio_data)} bytes")
        
        def on_speech_start():
            print("🎤 Speech started")
        
        def on_speech_end():
            print("🔇 Speech ended")
        
        mic.set_audio_callback(on_audio)
        mic.set_speech_callbacks(on_speech_start, on_speech_end)
        
        # Start capture
        if mic.start_capture():
            print("Microphone capture started. Speak into the microphone...")
            print("Press Ctrl+C to stop")
            
            try:
                while True:
                    time.sleep(1)
                    level = mic.get_audio_level()
                    print(f"Audio level: {level:.1f}")
            except KeyboardInterrupt:
                print("\nStopping...")
            
            mic.stop_capture()
            mic.cleanup()
            
            stats = mic.get_statistics()
            print(f"Capture statistics: {stats}")
        else:
            print("Failed to start microphone capture")
    
    # Uncomment to test
    # test_microphone()