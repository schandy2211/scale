#!/usr/bin/env python3
"""
SCALE Demo Launcher
Launch the sleek web interface for our AI molecular design system.
"""

import os
import sys
import subprocess
import webbrowser
import time
from threading import Timer

def install_requirements():
    """Install Flask requirements if needed."""
    try:
        import flask
        import flask_socketio
        print("✅ Flask dependencies already installed")
        return True
    except ImportError:
        print("📦 Installing Flask dependencies...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "Flask==2.3.3", "Flask-SocketIO==5.3.6", 
                "python-socketio==5.8.0", "eventlet==0.33.3"
            ])
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False

def open_browser():
    """Open browser after a short delay."""
    time.sleep(2)
    webbrowser.open('http://localhost:8080')

def main():
    print("🚀 SCALE - AI Molecular Design Demo")
    print("=" * 40)
    
    # Check dependencies
    if not install_requirements():
        return
    
    # Set working directory
    webapp_dir = os.path.join(os.path.dirname(__file__), 'webapp')
    os.chdir(webapp_dir)
    
    print("🌐 Starting web server at http://localhost:8080")
    print("💡 The demo will open in your browser automatically")
    print("🔧 Press Ctrl+C to stop the server")
    print("-" * 40)
    
    # Open browser in background
    timer = Timer(2.0, open_browser)
    timer.start()
    
    try:
        # Run Flask app
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Demo stopped. Thanks for using SCALE!")

if __name__ == "__main__":
    main()
