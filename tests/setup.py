#!/usr/bin/env python3
"""
Setup script for Philippines Flood Control Projects Chatbot
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def setup_env_file():
    """Set up environment file."""
    if not os.path.exists('.env'):
        print("Setting up environment file...")
        try:
            with open('.env.example', 'r') as example_file:
                content = example_file.read()
            
            with open('.env', 'w') as env_file:
                env_file.write(content)
            
            print("✅ .env file created from template")
            print("📝 Please edit .env file and add your OpenAI API key")
            return True
        except Exception as e:
            print(f"❌ Error creating .env file: {e}")
            return False
    else:
        print("✅ .env file already exists")
        return True

def main():
    """Main setup function."""
    print("🌊 Philippines Flood Control Projects Chatbot Setup")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("Setup failed. Please check the error messages above.")
        return False
    
    # Setup environment file
    if not setup_env_file():
        print("Warning: Environment file setup failed, but you can still run the app.")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file and add your OpenAI API key (optional)")
    print("2. Run the application with: streamlit run app.py")
    print("3. Upload your CSV file with flood control project data")
    print("4. Start asking questions about the projects!")
    
    return True

if __name__ == "__main__":
    main()
