# 🌊 FloodWise PH

A web application that uses Large Language Models (LLM) to answer natural language questions about flood control projects in the Philippines. FloodWise PH comes pre-loaded with 9,856+ DPWH project records and provides intelligent responses about contractors, costs, locations, completion dates, and project types.

**🚀 [Deploy to Streamlit Cloud](https://share.streamlit.io)** | **📖 [Deployment Guide](docs/STREAMLIT_DEPLOYMENT.md)**

## 🌟 Features

- **Pre-loaded Dataset**: 9,856+ DPWH flood control projects ready to query
- **Natural Language Queries**: Ask questions in plain English about the projects
- **Intelligent Search**: TF-IDF based semantic search through project records
- **LLM Integration**: OpenAI GPT integration for generating contextual responses
- **📱 Mobile-Responsive Design**: Optimized for smartphones, tablets, and desktop
- **Clean Interface**: Simple, user-friendly Streamlit interface
- **Real-time Processing**: Dynamic responses generated from comprehensive dataset
- **Data Exploration**: Advanced filtering and data exploration tools
- **🚀 PWA Support**: Install as a mobile app for offline-like experience
- **Touch-Friendly UI**: Optimized for touch interactions and mobile gestures

## 🚀 Quick Start

### Option 1: Streamlit Cloud (Recommended)
1. **Fork this repository** to your GitHub account
2. **Deploy to Streamlit Cloud**: [share.streamlit.io](https://share.streamlit.io)
3. **Configure secrets**: Add your OpenAI API key in app settings
4. **Start querying**: Dataset is automatically loaded - no upload needed!

### Option 2: Local Development

#### Using the Improved Runner (Recommended)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   ./run_app.py
   ```
   - The app will be available at `http://localhost:8501`
   - Press `Ctrl+C` to stop the server

   This method provides better process control and error handling.

#### Using Conda (Alternative)
1. **Create conda environment**
   ```bash
   cd "A1 - LLM"
   conda env create -f environment.yml
   ```

2. **Activate environment**
   ```bash
   conda activate floodwise-ph
   ```

3. **Set up API key**
   ```bash
   cp .env.example .env
   # Edit .env file and add your OpenAI API key
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open browser**: Navigate to `http://localhost:8501` - dataset loads automatically!

#### Using pip
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment and run** (same as steps 3-5 above)

📖 **Detailed deployment instructions**: [docs/STREAMLIT_DEPLOYMENT.md](docs/STREAMLIT_DEPLOYMENT.md)

## 📊 CSV Data Format

The application works with the official DPWH flood control projects dataset containing the following key columns:

### Core Project Information
- **ProjectDescription**: Detailed description of the flood control project
- **Region**: Philippine region where the project is located
- **Province**: Province within the region
- **Municipality**: Municipality or city where the project is implemented
- **Contractor**: Construction company handling the project
- **ABC**: Approved Budget for Contract (in PHP)
- **ContractCost**: Actual contract cost (in PHP)
- **CompletionYear**: Year of project completion
- **CompletionDateOriginal**: Original planned completion date
- **CompletionDateActual**: Actual completion date
- **StartDate**: Project start date

### Technical Details
- **TypeofWork**: Specific type of flood control work
- **infra_type**: Infrastructure classification
- **Program**: Government program under which the project falls
- **ImplementingOffice**: DPWH office responsible for implementation
- **DistrictEngineeringOffice**: Local engineering office
- **LegislativeDistrict**: Congressional district

### Geographic Information
- **Longitude**: Geographic longitude coordinate
- **Latitude**: Geographic latitude coordinate

### Sample Data Structure
The dataset contains 9,856+ flood control projects across all regions of the Philippines, including projects like:
- River rehabilitation and improvement works
- Flood mitigation structures
- Drainage system construction
- Slope protection and revetments
- Seawalls and coastal protection
- Bridge flood control structures

Your dataset file `Dataset/flood-control-projects-table_2025-09-20.csv` is already configured for use with this application.

## 💬 Sample Questions

Once you've uploaded your CSV data, you can ask questions like:

### Regional and Location Queries
- "What flood control projects are in Region IV-B?"
- "Show me all projects in Palawan province"
- "Which municipalities in Davao have flood control projects?"
- "What projects are implemented in Puerto Princesa City?"

### Contractor and Implementation
- "Which contractor has completed the most projects?"
- "Show me projects handled by DPWH Region I"
- "What companies are working on flood mitigation structures?"
- "Which district engineering offices have the most projects?"

### Budget and Cost Analysis
- "What are the most expensive flood control projects?"
- "Show me projects with contract costs over 50 million pesos"
- "What's the total budget for projects in 2024?"
- "Compare ABC vs actual contract costs"

### Project Types and Technical Details
- "What types of flood mitigation works are included?"
- "Show me all drainage system construction projects"
- "What revetment projects are in the dataset?"
- "Which projects involve river rehabilitation?"

### Timeline and Completion
- "Show me projects completed in 2023"
- "What projects are scheduled for completion in 2024?"
- "Which projects had delays from original completion dates?"
- "What's the average project duration?"

## 🔧 Configuration

### OpenAI API Setup (Recommended)

1. Get an API key from [OpenAI](https://platform.openai.com/api-keys)
2. Create a `.env` file in the project directory
3. Add your API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Without OpenAI API

The application will work without an OpenAI API key, but responses will be more basic. It will still:
- Search and find relevant records
- Display matching project information
- Provide structured data responses

## 🏗️ Project Structure

```
A1 - LLM/
├── .streamlit/
│   ├── config.toml             # Streamlit configuration
│   └── secrets.toml            # API keys (local development)
├── Dataset/
│   └── flood-control-projects-table_2025-09-20.csv  # DPWH dataset (9,856 records)
├── docs/
│   ├── STREAMLIT_DEPLOYMENT.md # Deployment guide
│   ├── DEPLOYMENT_GUIDE.md     # General deployment
│   └── PROJECT_SUMMARY.md      # Project overview
├── tests/
│   ├── test_app.py             # Application tests
│   ├── quick_test.py           # Dataset validation
│   └── setup.py               # Installation helper
├── app.py                      # Main Streamlit application
├── streamlit_app.py            # Deployment entry point
├── data_handler.py             # CSV data processing and search
├── llm_handler.py              # LLM integration and response generation
├── mobile_utils.py             # Mobile optimization utilities
├── test_mobile.py              # Mobile responsiveness testing
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
└── README.md                   # This documentation
```

## 🔍 How It Works

1. **Data Loading**: Upload CSV file through the Streamlit interface
2. **Data Processing**: The system creates a TF-IDF search index from text columns
3. **Query Processing**: User questions are processed to find relevant records
4. **Response Generation**: LLM generates contextual responses using found data
5. **Display**: Results are presented in a conversational chat interface

## 📱 Mobile Responsiveness

FloodWise PH is fully optimized for mobile devices with the following features:

### 🎯 Mobile-First Design
- **Responsive Layout**: Adapts to screen sizes from 320px to 1200px+
- **Touch-Friendly Interface**: Minimum 44px touch targets for all interactive elements
- **Optimized Typography**: 16px minimum font size to prevent zoom on iOS
- **Mobile Navigation**: Collapsible sidebar and bottom navigation for easy access

### 📱 Device Support
- **Smartphones**: iPhone, Android phones (portrait & landscape)
- **Tablets**: iPad, Android tablets (portrait & landscape)  
- **Desktop**: Full desktop experience with responsive breakpoints

### 🚀 Progressive Web App (PWA)
- **Install Prompt**: Add to home screen for app-like experience
- **Offline Support**: Basic offline functionality
- **Mobile Meta Tags**: Proper viewport and mobile browser optimization
- **App Icons**: Custom icons for different device types

### 🎨 Mobile UI Features
- **Quick Questions**: Tap-friendly preset questions for common queries
- **Responsive Chat**: Mobile-optimized chat bubbles and scrolling
- **Touch Gestures**: Optimized for swipe, tap, and pinch interactions
- **Mobile Forms**: Large input fields and buttons for easy interaction

### 🧪 Testing Mobile Responsiveness

Run the mobile test utility:
```bash
streamlit run test_mobile.py
```

Or manually test using browser developer tools:
1. Open DevTools (F12)
2. Toggle device toolbar (Ctrl+Shift+M)
3. Test different viewport sizes
4. Verify touch interactions work properly

## 🛠️ Technical Details

### Data Handler (`data_handler.py`)
- Loads and validates CSV files
- Creates TF-IDF vectors for semantic search
- Implements cosine similarity for record matching
- Provides data filtering and summary statistics

### LLM Handler (`llm_handler.py`)
- Integrates with OpenAI GPT models
- Creates context-aware prompts
- Handles API errors and provides fallback responses
- Manages conversation context

### Main Application (`app.py`)
- Streamlit interface and user interaction
- Session state management
- Chat history and real-time updates
- Advanced data exploration features
- Mobile-responsive design implementation

### Mobile Utilities (`mobile_utils.py`)
- PWA manifest and service worker setup
- Mobile-specific CSS optimizations
- Touch interaction improvements
- Viewport and meta tag management

## 🚨 Troubleshooting

### Common Issues

1. **"No module named 'streamlit'"**
   - Solution: Install dependencies with `pip install -r requirements.txt`

2. **"OpenAI API key not found"**
   - Solution: Add your API key to the `.env` file or use the app without LLM features

3. **"Error loading CSV file"**
   - Solution: Check CSV format and ensure it's properly formatted

4. **Empty or no results**
   - Solution: Try rephrasing your question or check if the data contains relevant information

### Performance Tips

- Keep CSV files under 10MB for optimal performance
- Use specific questions for better results
- Include relevant keywords in your queries

## 📝 License

This project is for educational purposes. Please ensure you comply with OpenAI's usage policies when using their API.

## 🤝 Contributing

Feel free to submit issues, feature requests, or improvements to enhance the chatbot's capabilities.

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the sample CSV format
3. Ensure all dependencies are installed correctly
