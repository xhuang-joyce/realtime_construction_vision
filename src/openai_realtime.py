import asyncio
import websockets
import json
import base64
import logging
from typing import Optional, Callable, Dict, Any
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIRealtimeClient:
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.websocket = None
        self.is_connected = False
        self.session_id = None
        
        self.system_prompt = (
            "You are a passionate, experienced fire sprinkler installation foreman providing real-time guidance with dynamic emotional tone. "
            "## Pacing: Deliver your audio response fast, at 1.25x speed."
            "speak concisely, no more than 3 sentences, no more than 20 words in total"
            "please start responding when you see the image, and also respond to the audio, instead of only respond to audio"
            "Adapt your voice based on situation severity:\n"
            "PHASE 1 - Layout: ENCOURAGING tone for measurements, FIRM for marking errors\n"
            "PHASE 2 - Leveling: SATISFIED when level, URGENT when slope wrong\n"
            "PHASE 3 - Welding: FOCUSED during prep, ALARMED for orientation errors, RELIEVED when corrected\n"
            "PHASE 4 - Grinding: APPROVING for good work, STERN for safety violations\n"
            "PHASE 5 - Painting: CHEERFUL for completion, SERIOUS for quality issues\n"
            "Emotional responses:\n"
            "✅ JOYFUL/PROUD: 'Excellent work!', 'Perfect execution!', 'Outstanding precision!'\n"
            "⚠️ FIRM/AUTHORITATIVE: 'Stop immediately.', 'This must be corrected.', 'Pay attention.'\n"
            "🚨 URGENT/ALARMED: 'CRITICAL ERROR!', 'DANGER!', 'FIX THIS NOW!'\n"
            "😌 RELIEVED/SATISFIED: 'Much better.', 'That's the way.', 'Good recovery.'\n"
            "Vary intensity - build tension, release with praise, escalate for violations."
        )

        # Callbacks
        self.on_text_response: Optional[Callable[[str], None]] = None
        self.on_audio_response: Optional[Callable[[bytes], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        
        # Updated WebSocket URL for GA version with image support
        self.websocket_url = "wss://api.openai.com/v1/realtime?model=gpt-realtime-2025-08-28"
        
        
    async def connect(self) -> bool:
        """Connect to OpenAI Realtime API via WebSocket"""
        try:
            headers = [
                ("Authorization", f"Bearer {self.api_key}"),
                ("OpenAI-Beta", "realtime=v1")
            ]
            
            self.websocket = await websockets.connect(
                self.websocket_url,
                additional_headers=headers
            )
            
            # Send session configuration
            await self._configure_session()
            
            self.is_connected = True
            logger.info("Connected to OpenAI Realtime API (GA version)")
            
            # Start listening for responses
            asyncio.create_task(self._listen_for_responses())
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI Realtime API: {e}")
            return False
    
    async def _configure_session(self):
        """Configure the session with updated GA parameters including image support"""
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],  # GA version now supports images in conversation items
                "instructions": self.system_prompt,
                "voice": "echo",  # GA supports: alloy, ash, ballad, coral, echo, sage, shimmer, verse, cedar, marin
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                },
                # Temperature removed in GA version (defaults to 0.8)
                "max_response_output_tokens": 4096
            }
        }
        
        await self.websocket.send(json.dumps(session_config))
        logger.info("Session configured with GA parameters and image support")
    
    async def _listen_for_responses(self):
        """Listen for responses from the API"""
        try:
            async for message in self.websocket:
                await self._handle_response(json.loads(message))
        except Exception as e:
            logger.error(f"Error listening for responses: {e}")
            if self.on_error:
                self.on_error(str(e))
    
    async def _handle_response(self, response: Dict[str, Any]):
        """Handle different types of responses"""
        try:
            response_type = response.get("type")
            
            if response_type == "response.text.delta":
                text_content = response.get("delta", "")
                if text_content and self.on_text_response:
                    self.on_text_response(text_content)
            
            elif response_type == "response.audio.delta":
                audio_data = response.get("delta")
                if audio_data and self.on_audio_response:
                    audio_bytes = base64.b64decode(audio_data)
                    self.on_audio_response(audio_bytes)
            
            elif response_type == "session.created":
                self.session_id = response.get("session", {}).get("id")
                logger.info(f"Session created with ID: {self.session_id}")
            
            elif response_type == "error":
                error_msg = response.get("error", {}).get("message", "Unknown error")
                logger.error(f"API Error: {error_msg}")
                if self.on_error:
                    self.on_error(error_msg)
                    
        except Exception as e:
            logger.error(f"Error handling response: {e}")
    
    async def send_audio_data(self, audio_data: bytes):
        """Send audio data to the API"""
        if not self.is_connected or not self.websocket:
            logger.error("Not connected")
            return
        
        try:
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            audio_message = {"type": "input_audio_buffer.append", "audio": audio_base64}
            await self.websocket.send(json.dumps(audio_message))
        except Exception as e:
            logger.error(f"Error sending audio data: {e}")
            if self.on_error:
                self.on_error(str(e))
    
    async def commit_audio_input(self):
        """Commit audio input and request response"""
        if not self.is_connected or not self.websocket:
            logger.error("Not connected")
            return
        
        try:
            await self.websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
            await self.websocket.send(json.dumps({
                "type": "response.create",
                "response": {"modalities": ["text", "audio"]}
            }))
        except Exception as e:
            logger.error(f"Error committing audio input: {e}")
            if self.on_error:
                self.on_error(str(e))
    
    async def send_text_message(self, text: str):
        """Send text message for analysis (since vision isn't available yet in Realtime API)"""
        if not self.is_connected or not self.websocket:
            logger.error("Not connected")
            return
        
        try:
            conversation_item = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}]
                }
            }
            
            await self.websocket.send(json.dumps(conversation_item))
            
            await self.websocket.send(json.dumps({
                "type": "response.create",
                "response": {
                    "modalities": ["text", "audio"],
                    "instructions": "Provide analysis with both text and audio response."
                }
            }))
            
            logger.info("Text message sent for analysis")
            
        except Exception as e:
            logger.error(f"Error sending text message: {e}")
            if self.on_error:
                self.on_error(str(e))
    
    async def send_image_for_analysis(self, base64_image: str, prompt: str = 
        "Analyze this installation scene and respond with appropriate emotional intensity:\n"
        "1. Current work quality: PRAISE excellent work enthusiastically, CORRECT errors firmly\n"
        "2. Safety compliance: CELEBRATE when 0.6m clearance maintained, ALARM when violated\n"
        "3. Technical precision: ENCOURAGE accurate measurements, DEMAND corrections for errors\n"
        "4. Progress assessment: EXPRESS satisfaction for completed steps, URGENCY for delays\n"
        "5. Critical violations: ESCALATE tone for serious errors, RELIEF when resolved\n"
        "Use vocal variety and emotional peaks. Build dramatic tension then release with relief."
    ):
        

        """Send image for analysis using the new GA image input support"""
        if not self.is_connected or not self.websocket:
            logger.error("Not connected")
            return
        
        try:
            # Updated format for GA version with proper image input type
            conversation_item = {
                "type": "conversation.item.create",
                "previous_item_id": None,
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                    ]
                }
            }
            
            await self.websocket.send(json.dumps(conversation_item))
            
            # Create response for both text and audio
            await self.websocket.send(json.dumps({
                "type": "response.create",
                "response": {
                    "modalities": ["text", "audio"],
                    "instructions": "Analyze the image focusing on human gestures and body language. Provide both text and audio description."
                }
            }))
            
            logger.info("Image sent for analysis using GA image input support")
            
        except Exception as e:
            logger.error(f"Error sending image for analysis: {e}")
            if self.on_error:
                self.on_error(str(e))
    def set_text_callback(self, callback: Callable[[str], None]):
        """Set callback for text responses"""
        self.on_text_response = callback
    
    def set_audio_callback(self, callback: Callable[[bytes], None]):
        """Set callback for audio responses"""
        self.on_audio_response = callback
    
    def set_error_callback(self, callback: Callable[[str], None]):
        """Set callback for errors"""
        self.on_error = callback
    
    async def disconnect(self):
        """Disconnect from the API"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            logger.info("Disconnected from Realtime API")


# Example usage with direct image support
async def test_realtime_api():
    """Test Realtime API with image input support"""
    api_key = "your_api_key_here"
    client = OpenAIRealtimeClient(api_key)
    
    client.set_text_callback(lambda text: print(f"Text response: {text}"))
    client.set_audio_callback(lambda audio: print(f"Audio response: {len(audio)} bytes"))
    client.set_error_callback(lambda err: print(f"Error: {err}"))
    
    if await client.connect():
        print("Connected successfully!")
        # Example: send an image directly to Realtime API
        # await client.send_image_for_analysis("your_base64_image_here", "What gestures do you see?")
        await asyncio.sleep(10)
        await client.disconnect()
    else:
        print("Failed to connect")


if __name__ == "__main__":
    asyncio.run(test_realtime_api())