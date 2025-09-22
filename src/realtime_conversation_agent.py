import asyncio
import time
import threading
import logging
import json
from typing import Optional, Callable, Dict, Any
from datetime import datetime
import queue
import cv2
import base64

from .camera_capture import CameraCapture
from .screen_capture import ScreenCapture
from .microphone_capture import MicrophoneCapture
from .openai_realtime import OpenAIRealtimeClient
from .audio_handler import AudioResponseHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealtimeConversationAgent:
    """
    True Realtime Conversation Agent with continuous voice chat and visual context awareness
    """
    
    def __init__(self, 
                 api_key: str,
                 camera_index: int = 0,
                 microphone_index: Optional[int] = None,
                 visual_context_interval: int = 2,
                 input_source: str = "camera"):  # "camera" or "screen"
        """
        Initialize realtime conversation agent
        
        Args:
            api_key: OpenAI API key
            camera_index: Camera device index
            microphone_index: Microphone device index (None for default)
            visual_context_interval: Update visual context every N seconds
            input_source: "camera" for camera input, "screen" for screen capture
        """
        self.api_key = api_key
        self.visual_context_interval = visual_context_interval
        self.input_source = input_source
        
        # Initialize visual input component based on source
        if input_source == "screen":
            self.visual_input = ScreenCapture(fps=15)
            logger.info("Using screen capture as visual input")
        else:
            self.visual_input = CameraCapture(camera_index)
            logger.info(f"Using camera {camera_index} as visual input")
        
        # Initialize other components
        self.microphone = MicrophoneCapture()  # device_index passed in start_capture()
        self.microphone_index = microphone_index  # Save device index
        self.ai_client = OpenAIRealtimeClient(api_key)
        self.audio_handler = AudioResponseHandler(api_key)
        
        self.is_running = False
        self.conversation_active = False
        self.last_visual_update = 0
        self.current_visual_context = None

        self.main_loop = None
        
        self.audio_buffer = []
        self.min_audio_duration_ms = 100  
        self.audio_sample_rate = 24000  # 24kHz
        self.min_audio_samples = int(self.min_audio_duration_ms * self.audio_sample_rate / 1000)
        self.current_audio_samples = 0
        
        self.is_processing_audio = False
        
        self.visual_update_task = None
        self.conversation_task = None
        
        self._setup_callbacks()
        
        logger.info("Realtime Conversation Agent initialized")
    
    def _setup_callbacks(self):
        
        self.ai_client.set_text_callback(self._handle_ai_text_response)
        self.ai_client.set_audio_callback(self._handle_ai_audio_response)
        self.ai_client.set_error_callback(self._handle_ai_error)
        
        self.microphone.set_speech_callbacks(
            on_start=self._handle_speech_start,
            on_end=self._handle_speech_end
        )
        self.microphone.set_audio_callback(self._handle_audio_data)
        
        logger.info("Callbacks configured")
    
    async def start_conversation(self):

        try:
            logger.info("Starting realtime conversation agent...")
            
            self.main_loop = asyncio.get_running_loop()
            
            # Start visual input (camera or screen)
            if self.input_source == "screen":
                self.visual_input.start_capture()
                logger.info("Screen capture started")
            else:
                if not self.visual_input.start_capture():
                    raise Exception("Failed to start camera")
            
            if not self.microphone.start_capture(device_index=self.microphone_index):
                raise Exception("Failed to start microphone")
            
            self.microphone.set_vad_threshold(300) 
            logger.info("VAD optimized for lower latency")
            
            self.audio_handler.start()
            
            # OpenAI Realtime API
            if not await self.ai_client.connect():
                raise Exception("Failed to connect to OpenAI Realtime API")
            
            self.is_running = True
            
            self.visual_update_task = asyncio.create_task(self._visual_context_loop())
            self.conversation_task = asyncio.create_task(self._conversation_loop())
            
            await self._update_visual_context()
            
            await self._send_welcome_message()
            
            logger.info("🎉 Realtime conversation agent started successfully!")
            logger.info("💬 You can now speak naturally - the AI can see and hear you!")
            
            await asyncio.gather(self.visual_update_task, self.conversation_task)
            
        except Exception as e:
            logger.error(f"Failed to start conversation agent: {e}")
            await self.stop_conversation()
            raise
    
    async def _send_welcome_message(self):
        try:
            welcome_text = (
                "Hello! I'm your realtime AI Construction copilot."
            )
            
            await self.ai_client.send_text_message(welcome_text)
            
        except Exception as e:
            logger.error(f"Failed to send welcome message: {e}")
    
    async def _visual_context_loop(self):
        while self.is_running:
            try:
                await self._update_visual_context()
                await asyncio.sleep(self.visual_context_interval)
            except Exception as e:
                logger.error(f"Error in visual context loop: {e}")
                await asyncio.sleep(1)
    
    async def _conversation_loop(self):
        while self.is_running:
            try:

                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in conversation loop: {e}")
                await asyncio.sleep(1)
    
    async def _update_visual_context(self):
        """Update visual context from camera or screen"""
        try:
            current_frame = self.visual_input.get_current_frame()
            if current_frame is not None:
                # Convert to base64
                base64_image = self.visual_input.frame_to_base64(current_frame)
                self.current_visual_context = base64_image
                self.last_visual_update = time.time()
                
                # Send visual context to AI (silent update, no response required)
                await self._send_visual_context_update(base64_image)
                
                logger.debug(f"Visual context updated from {self.input_source}")
        except Exception as e:
            logger.error(f"Error updating visual context: {e}")
    
    async def _send_visual_context_update(self, base64_image: str):
        try:
            context_message = {
                "type": "conversation.item.create",
                "previous_item_id": None,
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text", 
                            "text": "[Visual context update - no response needed]"
                        },
                        {
                            "type": "input_image", 
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    ]
                }
            }
            
            await self.ai_client.websocket.send(json.dumps(context_message))
            
        except Exception as e:
            logger.error(f"Error sending visual context: {e}")
    
    def _handle_speech_start(self):
        """处理语音开始"""
        logger.info("🎤 Speech detected - listening...")
        self.conversation_active = True
        
        # 重置音频缓冲区
        self.audio_buffer = []
        self.current_audio_samples = 0
        self.is_processing_audio = False
        
        # 如果 AI 正在说话，停止它
        # TODO: 实现打断机制
    
    def _handle_speech_end(self):
        """处理语音结束"""
        logger.info("🎤 Speech ended - processing...")
        
        # 处理积累的音频数据
        if self.audio_buffer and not self.is_processing_audio:
            self._process_accumulated_audio()
    
    def _handle_audio_data(self, audio_data: bytes):
        """处理音频数据 - 积累音频直到有足够的长度"""
        try:
            if self.conversation_active and not self.is_processing_audio:
                # 积累音频数据
                self.audio_buffer.append(audio_data)
                
                # 估算音频样本数 (假设16位单声道)
                samples_in_chunk = len(audio_data) // 2
                self.current_audio_samples += samples_in_chunk
                
                # 如果积累了足够的音频，处理它
                if self.current_audio_samples >= self.min_audio_samples:
                    self._process_accumulated_audio()
                    
        except Exception as e:
            logger.error(f"Error handling audio data: {e}")
    
    def _process_accumulated_audio(self):
        """处理积累的音频数据"""
        try:
            if not self.audio_buffer or self.is_processing_audio:
                return
                
            # 防止重复处理
            self.is_processing_audio = True
            
            # 合并所有音频数据
            combined_audio = b''.join(self.audio_buffer)
            
            logger.info(f"Processing {len(combined_audio)} bytes of audio ({self.current_audio_samples} samples)")
            
            # 使用线程安全的方式发送到 AI
            if self.main_loop and self.main_loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self._process_audio_input(combined_audio),
                    self.main_loop
                )
                logger.debug("Accumulated audio processing scheduled")
            else:
                logger.warning("Main event loop not available for audio processing")
                self.is_processing_audio = False
                
        except Exception as e:
            logger.error(f"Error processing accumulated audio: {e}")
            self.is_processing_audio = False
    
    async def _process_audio_input(self, audio_data: bytes):
        """处理音频输入"""
        try:
            # 发送音频到 AI
            await self.ai_client.send_audio_data(audio_data)
            
            # 如果有当前视觉上下文，一起发送
            if self.current_visual_context:
                await self._send_current_visual_context_with_audio()
            
            # 提交音频并请求回应
            await self.ai_client.commit_audio_input()
            
            logger.info("✅ Audio input processed and committed")
            
        except Exception as e:
            logger.error(f"Error processing audio input: {e}")
        finally:
            # 重置处理状态
            self.is_processing_audio = False
            self.audio_buffer = []
            self.current_audio_samples = 0
    
    async def _send_current_visual_context_with_audio(self):
        """随语音一起发送当前视觉上下文"""
        try:
            if self.current_visual_context:
                context_message = {
                    "type": "conversation.item.create",
                    "previous_item_id": None,
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text", 
                                "text": "Here's what I'm currently seeing:"
                            },
                            {
                                "type": "input_image", 
                                "image_url": f"data:image/jpeg;base64,{self.current_visual_context}"
                            }
                        ]
                    }
                }
                
                await self.ai_client.websocket.send(json.dumps(context_message))
                logger.debug("Visual context sent with audio")
                
        except Exception as e:
            logger.error(f"Error sending visual context with audio: {e}")
    
    def _handle_ai_text_response(self, text: str):
        """处理 AI 文本回应"""
        logger.info(f"🤖 AI (text): {text}")
        # 可以在这里添加文本显示逻辑
    
    def _handle_ai_audio_response(self, audio_data: bytes):
        """处理 AI 音频回应"""
        logger.info(f"🤖 AI (audio): {len(audio_data)} bytes")
        # 播放音频回应
        self.audio_handler.handle_audio_response(audio_data)
    
    def _handle_ai_error(self, error_msg: str):
        """处理 AI 错误"""
        logger.error(f"🚨 AI Error: {error_msg}")
        # 可以在这里添加错误恢复逻辑
    
    async def stop_conversation(self):
        """停止对话"""
        logger.info("Stopping realtime conversation agent...")
        
        self.is_running = False
        
        # 取消任务
        if self.visual_update_task:
            self.visual_update_task.cancel()
        if self.conversation_task:
            self.conversation_task.cancel()
        
        # Stop components
        if self.visual_input:
            self.visual_input.stop_capture()
        if self.microphone:
            self.microphone.stop_capture()
        if self.audio_handler:
            self.audio_handler.stop()
        if self.ai_client:
            await self.ai_client.disconnect()
        
        logger.info("✅ Realtime conversation agent stopped")
    
    async def ask_about_vision(self, question: str):
        """询问关于当前视觉内容的问题"""
        try:
            if not self.current_visual_context:
                await self._update_visual_context()
            
            if self.current_visual_context:
                await self.ai_client.send_image_for_analysis(
                    self.current_visual_context, 
                    question
                )
                logger.info(f"Asked about vision: {question}")
            else:
                logger.warning("No visual context available")
                
        except Exception as e:
            logger.error(f"Error asking about vision: {e}")


# 使用示例
async def main():
    """主函数示例"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ Please set OPENAI_API_KEY in your .env file")
        return
    
    agent = RealtimeConversationAgent(api_key)
    
    try:
        await agent.start_conversation()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    finally:
        await agent.stop_conversation()


if __name__ == "__main__":
    asyncio.run(main())