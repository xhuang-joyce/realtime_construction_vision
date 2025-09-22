#!/usr/bin/env python3

import asyncio
import sys
import signal
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import Config
from src.realtime_conversation_agent import RealtimeConversationAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealtimeConversationApp:
    """真正的实时对话应用程序"""
    
    def __init__(self, config: Config):
        self.config = config
        self.agent = None
        self.is_running = False
        
    async def initialize(self):
        """初始化应用程序"""
        try:
            logger.info("🚀 Initializing True Realtime Conversation Agent...")
            
            # Create realtime conversation agent
            self.agent = RealtimeConversationAgent(
                api_key=self.config.get('api_key'),
                camera_index=self.config.get('camera_index', 0),
                microphone_index=self.config.get('microphone_index'),
                visual_context_interval=self.config.get('visual_context_interval', 2),
                input_source=self.config.get('input_source', 'camera')
            )
            
            logger.info("✅ Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize agent: {e}")
            return False
    
    async def run(self):
        """运行应用程序"""
        try:
            logger.info("🎬 Starting True Realtime Conversation...")
            logger.info("💡 This is a REAL conversation agent that:")
            logger.info("   🎤 Listens to your voice continuously")
            logger.info("   👁️  Sees what you see through the camera")
            logger.info("   🧠 Understands context from both audio and vision")
            logger.info("   🗣️  Responds with natural speech")
            logger.info("")
            logger.info("💬 Try saying things like:")
            logger.info("   - 'What do you see in front of me?'")
            logger.info("   - 'Describe what's happening in this room'")
            logger.info("   - 'What gestures am I making?'")
            logger.info("   - Or just have a natural conversation!")
            logger.info("")
            logger.info("🎯 Press Ctrl+C to stop")
            logger.info("")
            
            self.is_running = True
            
            # 启动实时对话
            await self.agent.start_conversation()
            
        except KeyboardInterrupt:
            logger.info("\n👋 Received interrupt signal, shutting down gracefully...")
        except Exception as e:
            logger.error(f"❌ Error during conversation: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("🧹 Cleaning up resources...")
            self.is_running = False
            
            if self.agent:
                await self.agent.stop_conversation()
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
    
    def setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            logger.info(f"\n🛑 Received signal {signum}, stopping...")
            self.is_running = False
            # 创建停止任务
            if self.agent:
                asyncio.create_task(self.agent.stop_conversation())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="True Realtime AI Conversation Agent",
        epilog="""
Examples:
  python realtime_agent.py                    # Start with default settings
  python realtime_agent.py --camera-index 1  # Use camera 1
  python realtime_agent.py --visual-interval 5  # Update visual context every 5 seconds
        """
    )
    
    parser.add_argument("--config", type=str, 
                       help="Path to configuration file")
    parser.add_argument("--camera-index", type=int,
                       help="Camera device index")
    parser.add_argument("--microphone-index", type=int,
                       help="Microphone device index (None for default)")
    parser.add_argument("--visual-interval", type=int,
                       help="Visual context update interval in seconds")
    parser.add_argument("--log-level", type=str, 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help="Set logging level")
    parser.add_argument("--list-devices", action="store_true",
                       help="List available audio/video devices and exit")
    
    args = parser.parse_args()
    
    try:
        # 设置日志级别
        if args.log_level:
            logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        # 列出设备（如果需要）
        if args.list_devices:
            from src.camera_capture import CameraCapture
            from src.microphone_capture import MicrophoneCapture
            
            print("📹 Available cameras:")
            CameraCapture.list_available_cameras()
            
            print("\n🎤 Available microphones:")
            MicrophoneCapture.list_audio_devices()
            return 0
        
        # 加载配置
        config = Config(args.config)
        
        # 命令行参数覆盖配置
        if args.camera_index is not None:
            config.set('camera_index', args.camera_index)
        if args.microphone_index is not None:
            config.set('microphone_index', args.microphone_index)
        if args.visual_interval is not None:
            config.set('visual_context_interval', args.visual_interval)
        
        # 验证 API key
        if not config.get('api_key'):
            print("❌ Error: OPENAI_API_KEY not found in environment variables")
            print("Please set your API key in the .env file")
            return 1
        
        print("🌟 Welcome to the True Realtime AI Conversation Agent!")
        print("📋 Configuration:")
        print(f"   🔑 API Key: {'✅ Set' if config.get('api_key') else '❌ Missing'}")
        print(f"   📹 Camera: {config.get('camera_index', 0)}")
        print(f"   🎤 Microphone: {config.get('microphone_index', 'Default')}")
        print(f"   ⏰ Visual update interval: {config.get('visual_context_interval', 2)}s")
        print()
        
        # 创建并运行应用
        app = RealtimeConversationApp(config)
        app.setup_signal_handlers()
        
        if not await app.initialize():
            print("❌ Failed to initialize application")
            return 1
        
        await app.run()
        
        print("\n✅ Application terminated successfully")
        return 0
        
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)