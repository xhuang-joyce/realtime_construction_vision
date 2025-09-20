import pyaudio
import wave
import io
import threading
import queue
import logging
import time
from typing import Optional, Callable
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioPlayer:
    """
    Audio player for real-time playback of OpenAI audio responses
    """
    
    def __init__(self, 
                 sample_rate: int = 24000,
                 channels: int = 1,
                 sample_width: int = 2,
                 buffer_size: int = 1024):
        """
        Initialize audio player
        
        Args:
            sample_rate: Audio sample rate (Hz)
            channels: Number of audio channels (1 for mono, 2 for stereo)
            sample_width: Sample width in bytes (2 for 16-bit)
            buffer_size: Audio buffer size
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.buffer_size = buffer_size
        
        # PyAudio setup
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        
        # Audio queue and playback state
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.playback_thread = None
        
        # Statistics
        self.total_audio_chunks = 0
        self.total_bytes_played = 0
        
    def start_playback(self) -> bool:
        """
        Start audio playback system
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        try:
            # Open audio stream
            self.stream = self.pyaudio.open(
                format=self.pyaudio.get_format_from_width(self.sample_width),
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.buffer_size
            )
            
            # Start playback thread
            self.is_playing = True
            self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
            self.playback_thread.start()
            
            logger.info(f"Audio playback started (Rate: {self.sample_rate}Hz, Channels: {self.channels})")
            return True
            
        except Exception as e:
            logger.error(f"Error starting audio playback: {e}")
            return False
    
    def _playback_loop(self):
        """
        Main audio playback loop
        """
        while self.is_playing:
            try:
                # Get audio data from queue (blocking with timeout)
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Play audio data
                if audio_data and self.stream:
                    self.stream.write(audio_data)
                    self.total_audio_chunks += 1
                    self.total_bytes_played += len(audio_data)
                
                # Mark task as done
                self.audio_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in audio playback loop: {e}")
    
    def play_audio_chunk(self, audio_data: bytes):
        """
        Queue audio chunk for playback
        
        Args:
            audio_data: Raw audio data bytes
        """
        try:
            if self.is_playing and audio_data:
                # Add to playback queue
                self.audio_queue.put(audio_data)
                
        except Exception as e:
            logger.error(f"Error queueing audio chunk: {e}")
    
    def play_audio_stream(self, audio_stream: bytes):
        """
        Play a continuous audio stream by chunking it
        
        Args:
            audio_stream: Complete audio stream as bytes
        """
        try:
            # Chunk the audio stream for smoother playback
            chunk_size = self.buffer_size * self.sample_width * self.channels
            
            for i in range(0, len(audio_stream), chunk_size):
                chunk = audio_stream[i:i + chunk_size]
                self.play_audio_chunk(chunk)
                
        except Exception as e:
            logger.error(f"Error playing audio stream: {e}")
    
    def stop_playback(self):
        """
        Stop audio playback and cleanup resources
        """
        self.is_playing = False
        
        # Wait for playback thread to finish
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=2.0)
        
        # Close audio stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        # Terminate PyAudio
        if self.pyaudio:
            self.pyaudio.terminate()
        
        logger.info("Audio playback stopped")
    
    def get_queue_size(self) -> int:
        """
        Get current audio queue size
        
        Returns:
            int: Number of audio chunks in queue
        """
        return self.audio_queue.qsize()
    
    def get_statistics(self) -> dict:
        """
        Get audio playback statistics
        
        Returns:
            dict: Playback statistics
        """
        return {
            'total_audio_chunks': self.total_audio_chunks,
            'total_bytes_played': self.total_bytes_played,
            'queue_size': self.get_queue_size(),
            'is_playing': self.is_playing,
            'sample_rate': self.sample_rate,
            'channels': self.channels
        }


class TextToSpeechPlayer:
    """
    Text-to-speech player using OpenAI's TTS API for fallback when realtime audio is not available
    """
    
    def __init__(self, api_key: str, voice: str = "alloy"):
        """
        Initialize TTS player
        
        Args:
            api_key: OpenAI API key
            voice: Voice to use for TTS (alloy, echo, fable, onyx, nova, shimmer)
        """
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.voice = voice
        self.audio_player = AudioPlayer()
        
    def speak_text(self, text: str):
        """
        Convert text to speech and play it
        
        Args:
            text: Text to convert to speech
        """
        try:
            logger.info(f"Converting text to speech: {text[:50]}...")
            
            # Generate speech using OpenAI TTS
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=self.voice,
                input=text,
                response_format="pcm"
            )
            
            # Play the audio
            audio_data = response.content
            self.audio_player.play_audio_stream(audio_data)
            
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
    
    def start(self) -> bool:
        """
        Start the TTS player
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        return self.audio_player.start_playback()
    
    def stop(self):
        """
        Stop the TTS player
        """
        self.audio_player.stop_playback()


class AudioResponseHandler:
    """
    Complete audio response handler that manages both real-time audio and TTS fallback
    """
    
    def __init__(self, api_key: str, use_tts_fallback: bool = True):
        """
        Initialize audio response handler
        
        Args:
            api_key: OpenAI API key
            use_tts_fallback: Whether to use TTS for text responses when no audio is available
        """
        self.api_key = api_key
        self.use_tts_fallback = use_tts_fallback
        
        # Initialize audio players
        self.realtime_player = AudioPlayer()
        self.tts_player = TextToSpeechPlayer(api_key) if use_tts_fallback else None
        
        # Response tracking
        self.last_audio_time = None
        self.last_text_time = None
        self.audio_response_count = 0
        self.text_response_count = 0
        
    def start(self) -> bool:
        """
        Start audio response handling
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        try:
            # Start real-time audio player
            if not self.realtime_player.start_playback():
                return False
            
            # Start TTS player if enabled
            if self.tts_player and not self.tts_player.start():
                logger.warning("Failed to start TTS player, continuing without TTS fallback")
                self.tts_player = None
            
            logger.info("Audio response handler started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting audio response handler: {e}")
            return False
    
    def handle_audio_response(self, audio_data: bytes):
        """
        Handle real-time audio response from OpenAI
        
        Args:
            audio_data: Raw audio data bytes
        """
        try:
            self.realtime_player.play_audio_chunk(audio_data)
            self.audio_response_count += 1
            self.last_audio_time = time.time()
            
            logger.debug(f"Played audio chunk ({len(audio_data)} bytes)")
            
        except Exception as e:
            logger.error(f"Error handling audio response: {e}")
    
    def handle_text_response(self, text: str):
        """
        Handle text response - optionally convert to speech if no recent audio
        
        Args:
            text: Text response from OpenAI
        """
        try:
            self.text_response_count += 1
            self.last_text_time = time.time()
            
            # Log the text response
            logger.info(f"Text response: {text}")
            
            # Use TTS if enabled and no recent audio response
            if (self.tts_player and 
                (not self.last_audio_time or 
                 time.time() - self.last_audio_time > 5.0)):
                
                # Convert text to speech in separate thread to avoid blocking
                threading.Thread(
                    target=self.tts_player.speak_text,
                    args=(text,),
                    daemon=True
                ).start()
                
        except Exception as e:
            logger.error(f"Error handling text response: {e}")
    
    def stop(self):
        """
        Stop audio response handling
        """
        try:
            self.realtime_player.stop_playback()
            
            if self.tts_player:
                self.tts_player.stop()
            
            logger.info("Audio response handler stopped")
            
        except Exception as e:
            logger.error(f"Error stopping audio response handler: {e}")
    
    def get_statistics(self) -> dict:
        """
        Get audio response statistics
        
        Returns:
            dict: Statistics about audio responses
        """
        stats = {
            'audio_response_count': self.audio_response_count,
            'text_response_count': self.text_response_count,
            'last_audio_time': self.last_audio_time,
            'last_text_time': self.last_text_time,
            'realtime_player_stats': self.realtime_player.get_statistics()
        }
        
        if self.tts_player:
            stats['tts_player_stats'] = self.tts_player.audio_player.get_statistics()
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    # Test audio playback
    import numpy as np
    
    def test_audio_player():
        """Test basic audio playback functionality"""
        player = AudioPlayer()
        
        if player.start_playback():
            print("Audio player started. Playing test tone...")
            
            # Generate a test tone
            duration = 2.0  # seconds
            frequency = 440  # Hz (A4 note)
            sample_rate = 24000
            
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = np.sin(2 * np.pi * frequency * t)
            
            # Convert to 16-bit integers
            tone_16bit = (tone * 32767).astype(np.int16)
            
            # Play the tone
            player.play_audio_stream(tone_16bit.tobytes())
            
            # Wait for playback to finish
            time.sleep(duration + 1)
            
            print("Test completed")
            player.stop_playback()
        else:
            print("Failed to start audio player")
    
    # Uncomment to test
    # test_audio_player()