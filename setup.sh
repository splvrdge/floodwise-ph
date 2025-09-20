#!/bin/bash

# Create necessary directories
mkdir -p .streamlit
mkdir -p data

# Create config.toml if it doesn't exist
if [ ! -f ".streamlit/config.toml" ]; then
    cat > .streamlit/config.toml <<EOL
[server]
headless = true
port = 8501
enableCORS = true
enableXsrfProtection = true
maxUploadSize = 200
fileWatcherType = "none"

[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8f9fa"
textColor = "#262730"
font = "sans serif"

[browser]
gatherUsageStats = false
EOL
fi

echo "✅ Setup complete! Your app is ready for deployment on Streamlit Cloud."
echo ""
echo "Next steps:"
echo "1. Push this repository to GitHub"
echo "2. Go to https://share.streamlit.io"
echo "3. Click 'New app' and select your repository"
echo "4. Set the main file path to 'app.py'"
echo "5. Add your OpenAI API key in the 'Secrets' section"
echo "6. Deploy!"
echo ""
echo "For more detailed instructions, check the README.md file."
