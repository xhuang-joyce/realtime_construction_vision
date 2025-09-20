"""
Real-time Vision Analysis Application

A comprehensive real-time computer vision system that uses OpenAI's APIs
to analyze video streams and provide audio/text responses.
"""

__version__ = "1.0.0"
__author__ = "AI Hack 2025"

from .camera_capture import CameraCapture
from .openai_realtime import OpenAIRealtimeClient
from .frame_processor import FrameProcessor, RealtimeVisionPipeline
from .audio_handler import AudioPlayer, AudioResponseHandler
from .config import Config, ErrorHandler

__all__ = [
    'CameraCapture',
    'OpenAIRealtimeClient',
    # 'VisionAnalysisClient', 
    'FrameProcessor',
    'RealtimeVisionPipeline',
    'AudioPlayer',
    'AudioResponseHandler',
    'Config',
    'ErrorHandler'
]