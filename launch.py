#!/usr/bin/env python3
"""
Quick launcher for Real-time Vision Analysis with Voice Interaction

This script provides easy commands to start the application in different modes.
"""

import sys
import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Launch Real-time Vision Analysis")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--voice", action="store_true",
                           help="Launch with voice interaction (recommended)")
    mode_group.add_argument("--vision", action="store_true", 
                           help="Launch vision-only mode")
    mode_group.add_argument("--test", action="store_true",
                           help="Run component tests")
    mode_group.add_argument("--test-voice", action="store_true",
                           help="Test voice interaction")
    mode_group.add_argument("--devices", action="store_true",
                           help="List available devices")
    
    # Additional options
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera index (default: 0)")
    parser.add_argument("--interval", type=int, default=3,
                       help="Frame interval in seconds (default: 3)")
    parser.add_argument("--input-source", type=str, choices=['camera', 'screen'], 
                       default='camera', help="Input source: camera or screen capture")
    
    args = parser.parse_args()
    
    # Determine command to run
    if args.test:
        cmd = [sys.executable, "test_components.py"]
    elif args.test_voice:
        cmd = [sys.executable, "test_multimodal.py"]
    elif args.devices:
        cmd = [sys.executable, "main.py", "--list-devices"]
    elif args.vision:
        cmd = [sys.executable, "main.py", "--vision-only", 
               f"--camera-index={args.camera}", f"--frame-interval={args.interval}",
               f"--input-source={args.input_source}"]
    else:  # Default to voice mode
        cmd = [sys.executable, "main.py", "--realtime",
               f"--camera-index={args.camera}", f"--frame-interval={args.interval}",
               f"--input-source={args.input_source}"]
    
    # Show what we're running
    print("🚀 Real-time Vision Analysis Launcher")
    print("=" * 50)
    
    if args.test:
        print("🧪 Running component tests...")
    elif args.test_voice:
        print("🎤 Testing voice interaction...")
    elif args.devices:
        print("📱 Listing available devices...")
    elif args.vision:
        print("📹 Starting vision-only mode...")
    else:
        print("🎤📹 Starting voice interaction mode...")
        print("💡 You can speak to the AI while it watches your camera!")
    
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    # Run the command
    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
        return result.returncode
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())