# 🤖 True Realtime AI Conversation Agent

基于 OpenAI Realtime API (2025年8月28日更新) 构建的**真正实时对话系统**，支持图像输入和持续语音交互。

## 🌟 什么是真正的实时对话代理？

与传统的"发送图片→等待回复"系统不同，这是一个**真正的实时对话代理**：

### ✨ 核心特性
- 🎤 **持续语音对话**：用户可以随时说话，AI 随时回应
- 👁️ **视觉上下文感知**：AI 能看到你的环境，并在对话中引用
- 🧠 **多模态融合**：语音、视觉、文本无缝结合
- ⚡ **真实时交互**：无需等待，自然对话流程
- 🎯 **手势理解**：专门优化用于理解人类手势和身体语言

### 🆚 与传统系统的区别

**传统系统**（类似当前大多数实现）：
```
用户拍照 → 发送API → 等待回复 → 播放音频 → 结束
```

**真正的实时代理**（本项目）：
```
持续监听 ← → 持续看见 ← → 持续对话 ← → 持续理解
```

## 🌟 Features

- **🎤 Voice Interaction**: Speak naturally to the AI while it watches through your camera
- **📹 Real-time Video Analysis**: Continuous analysis of camera feed with focus on human gestures
- **🤖 Intelligent Responses**: AI provides both spoken audio and text responses
- **🎯 Gesture Understanding**: Expert-level analysis of body language and human gestures
- **⚡ Real-time Processing**: Low-latency interaction using OpenAI's Realtime API
- **🔄 Multimodal Integration**: Seamless combination of vision and voice input

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Update your API key in `.env` file:
```
OPENAI_API_KEY=your_actual_api_key_here
```

3. Run the application:
```bash
python main.py
```

## 🚀 Quick Start

### Voice Interaction Mode (Recommended)
```bash
# Start with voice interaction - you can talk to the AI!
python main.py --realtime

# Or use the launcher
python launch.py --voice
```

### Vision-Only Mode
```bash
# Traditional mode - AI analyzes camera without voice input
python main.py --vision-only

# Or use the launcher  
python launch.py --vision
```

### Testing
```bash
# Test all components
python test_components.py

# Test voice interaction specifically
python test_multimodal.py

# List available devices
python main.py --list-devices
```

## 💬 How to Use

### Voice Interaction Mode:
1. The AI can see through your camera and hear your voice
2. Speak naturally - try phrases like:
   - "Hello, what do you see?"
   - "Can you describe what's in front of you?"
   - "What gesture am I making?" (while waving or pointing)
   - "Tell me about the scene"
3. The AI will respond with both voice and text
4. Have a natural conversation while it watches!

### Vision-Only Mode:
- The AI analyzes your camera feed continuously
- Provides automated descriptions with text-to-speech
- Focus on gesture and body language analysis
- No voice input required

## System Requirements

- Python 3.8+
- Webcam/Camera
- Internet connection for OpenAI API
- Audio output capability