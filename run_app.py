#!/usr/bin/env python3
"""
Run the FloodWise PH application with better process control.
This script provides better control over the Streamlit server.
"""
import os
import sys
import signal
import subprocess
import time
from pathlib import Path

# Configuration
PORT = 8501
HOST = '0.0.0.0'  # Listen on all network interfaces
APP_SCRIPT = 'app.py'

class StreamlitRunner:
    def __init__(self):
        self.process = None
        self.port = PORT
        self.host = HOST
        
    def start(self):
        """Start the Streamlit server."""
        if self.process is not None:
            print("Server is already running!")
            return
            
        cmd = [
            'streamlit', 'run',
            '--server.port', str(self.port),
            '--server.address', self.host,
            '--server.headless', 'true',
            '--server.enableCORS', 'true',
            '--server.enableXsrfProtection', 'true',
            '--server.fileWatcherType', 'none',
            '--browser.gatherUsageStats', 'false',
            APP_SCRIPT
        ]
        
        print(f"Starting Streamlit server on http://{self.host}:{self.port}")
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        
        # Wait a bit to catch early errors
        time.sleep(2)
        if self.process.poll() is not None:
            _, stderr = self.process.communicate()
            print(f"Error starting server: {stderr.decode()}")
            self.process = None
            return False
            
        return True
    
    def stop(self):
        """Stop the Streamlit server."""
        if self.process is not None:
            print("Stopping server...")
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process = None
            print("Server stopped.")

def main():
    runner = StreamlitRunner()
    
    # Handle keyboard interrupt
    def signal_handler(sig, frame):
        print("\nShutting down...")
        runner.stop()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start the server
    if runner.start():
        print("Server is running. Press Ctrl+C to stop.")
        # Keep the main thread alive
        while True:
            time.sleep(1)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
