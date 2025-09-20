# 🚀 Deployment Guide - Philippines Flood Control Projects Chatbot

## Quick Start (5 Minutes)

### 1. Install Dependencies
```bash
cd "A1 - LLM"
pip install -r requirements.txt
```

### 2. Set Up OpenAI API (Optional but Recommended)
```bash
# Copy the environment template
cp .env.example .env

# Edit .env file and add your OpenAI API key
# OPENAI_API_KEY=your_actual_api_key_here
```

### 3. Run the Application
```bash
streamlit run app.py
```

### 4. Upload Your Dataset
- Open the app in your browser (usually http://localhost:8501)
- Use the sidebar to upload `Dataset/flood-control-projects-table_2025-09-20.csv`
- Click "Load Data"
- Start asking questions!

## 📊 Your Dataset Overview

Your dataset contains **9,856 flood control projects** with comprehensive information:

### Key Features Available for Queries:
- **Geographic Coverage**: All Philippine regions, provinces, and municipalities
- **Project Types**: Flood mitigation, drainage systems, revetments, seawalls, slope protection
- **Financial Data**: ABC (Approved Budget), Contract Costs, ranging from thousands to millions of pesos
- **Timeline Data**: Projects from 2021-2024 with start and completion dates
- **Implementation Details**: DPWH offices, contractors, legislative districts

### Sample Queries You Can Try:
```
"What are the most expensive projects in Palawan?"
"Show me all drainage projects completed in 2024"
"Which contractors work on flood mitigation structures?"
"What's the total budget for Region IV-B projects?"
"Compare project costs between different regions"
```

## 🔧 Configuration Options

### With OpenAI API (Recommended)
- **Pros**: Intelligent, contextual responses; natural language understanding
- **Cons**: Requires API key and usage costs
- **Best for**: Production use, detailed analysis, complex queries

### Without OpenAI API (Fallback Mode)
- **Pros**: Free to use, no external dependencies
- **Cons**: Basic responses, limited natural language processing
- **Best for**: Testing, simple data retrieval, budget-conscious deployment

## 🌐 Production Deployment Options

### Option 1: Streamlit Cloud (Free)
1. Push your code to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add your OpenAI API key in secrets
4. Deploy with one click

### Option 2: Local Network Deployment
```bash
# Run on specific host/port for network access
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### Option 3: Docker Deployment
```dockerfile
# Create Dockerfile for containerized deployment
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

## 🔍 Advanced Usage Tips

### 1. Query Optimization
- Use specific location names for better results
- Include project types in your queries
- Ask about specific years or date ranges
- Combine multiple criteria (e.g., "expensive projects in Davao completed in 2023")

### 2. Data Exploration
- Use the "Advanced Data Exploration" section for filtering
- Check dataset summary for available columns
- Explore different project types and regions

### 3. Performance Tips
- The app loads all 9,856 records into memory for fast searching
- TF-IDF indexing provides quick semantic search
- Responses are generated in real-time from your data

## 🛠️ Troubleshooting

### Common Issues:
1. **"Module not found"** → Run `pip install -r requirements.txt`
2. **"File not found"** → Ensure CSV file is in the correct location
3. **"API key error"** → Check your .env file or use without OpenAI
4. **Slow responses** → Large dataset may take a few seconds to process

### Performance Optimization:
- For very large datasets (>50MB), consider data preprocessing
- Use specific queries rather than very broad ones
- Clear chat history periodically to free memory

## 📈 Usage Analytics

Your dataset enables analysis of:
- **Regional Distribution**: Projects across 13+ regions
- **Budget Analysis**: Total investments in flood control infrastructure
- **Timeline Tracking**: Project completion rates and delays
- **Contractor Performance**: Company involvement and project success
- **Infrastructure Types**: Distribution of different flood control solutions

## 🔒 Security Considerations

- Keep your OpenAI API key secure in the .env file
- Don't commit .env files to version control
- Consider rate limiting for production deployments
- Validate user inputs for production use

## 📞 Support & Maintenance

- Monitor API usage if using OpenAI
- Update dependencies regularly
- Back up your dataset and configuration
- Test with new data uploads periodically

---

**Ready to deploy!** Your chatbot is configured to provide intelligent insights about Philippines flood control projects using your comprehensive 9,856-record dataset.
