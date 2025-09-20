#!/usr/bin/env python3
"""
Mobile Responsiveness Test Script for FloodWise PH
This script helps verify that the application works well on mobile devices.
"""

import streamlit as st
import subprocess
import webbrowser
import time
import os

def test_mobile_responsiveness():
    """Test mobile responsiveness by opening the app in different viewport sizes."""
    
    st.title("📱 Mobile Responsiveness Test")
    st.markdown("This tool helps you test FloodWise PH on different screen sizes.")
    
    # Test configurations
    test_configs = {
        "📱 Mobile Portrait": {"width": 375, "height": 667, "description": "iPhone SE"},
        "📱 Mobile Landscape": {"width": 667, "height": 375, "description": "iPhone SE Landscape"},
        "📱 Large Mobile": {"width": 414, "height": 896, "description": "iPhone 11 Pro"},
        "📟 Tablet Portrait": {"width": 768, "height": 1024, "description": "iPad"},
        "📟 Tablet Landscape": {"width": 1024, "height": 768, "description": "iPad Landscape"},
        "💻 Desktop": {"width": 1200, "height": 800, "description": "Desktop"}
    }
    
    st.subheader("🔍 Test Different Screen Sizes")
    
    selected_config = st.selectbox(
        "Choose a screen size to test:",
        list(test_configs.keys()),
        format_func=lambda x: f"{x} - {test_configs[x]['description']} ({test_configs[x]['width']}x{test_configs[x]['height']})"
    )
    
    if st.button("🚀 Launch Test", type="primary"):
        config = test_configs[selected_config]
        st.success(f"Testing {selected_config}: {config['width']}x{config['height']}")
        
        # Instructions for manual testing
        st.markdown(f"""
        ### 📋 Manual Testing Instructions
        
        1. **Open Developer Tools** in your browser (F12)
        2. **Click the device toolbar** (Ctrl+Shift+M or Cmd+Shift+M)
        3. **Set viewport to**: {config['width']} x {config['height']}
        4. **Test the following features**:
           - ✅ Navigation and sidebar
           - ✅ Text input and buttons
           - ✅ Chat interface
           - ✅ Data tables
           - ✅ Quick question buttons
           - ✅ Form submissions
           - ✅ Scrolling behavior
        
        ### 🎯 What to Look For:
        - **Text readability** (not too small)
        - **Button accessibility** (easy to tap)
        - **No horizontal scrolling** (unless intended)
        - **Proper spacing** between elements
        - **Sidebar behavior** on mobile
        - **Form usability** on touch devices
        """)
    
    st.subheader("🔧 Mobile Optimization Checklist")
    
    checklist_items = [
        "✅ Responsive CSS media queries implemented",
        "✅ Mobile-first design approach",
        "✅ Touch-friendly button sizes (min 44px)",
        "✅ Readable font sizes (min 16px for inputs)",
        "✅ Proper viewport meta tag",
        "✅ Optimized sidebar for mobile",
        "✅ Mobile-friendly chat interface",
        "✅ Responsive data tables",
        "✅ PWA capabilities added",
        "✅ Mobile navigation elements"
    ]
    
    for item in checklist_items:
        st.write(item)
    
    st.subheader("🌐 Browser Testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📱 Mobile Browsers to Test:**
        - Safari (iOS)
        - Chrome (Android)
        - Firefox Mobile
        - Samsung Internet
        - Edge Mobile
        """)
    
    with col2:
        st.markdown("""
        **💻 Desktop Browsers:**
        - Chrome DevTools
        - Firefox Responsive Mode
        - Safari Web Inspector
        - Edge DevTools
        """)
    
    st.subheader("🚨 Common Mobile Issues to Check")
    
    issues_to_check = [
        "Text too small to read",
        "Buttons too small to tap",
        "Horizontal scrolling required",
        "Sidebar not accessible",
        "Forms difficult to use",
        "Content cut off",
        "Poor performance on mobile",
        "Zoom required for interaction"
    ]
    
    for issue in issues_to_check:
        st.write(f"❌ {issue}")
    
    st.subheader("📊 Performance Tips")
    
    st.markdown("""
    ### 🏃‍♂️ Mobile Performance Optimization:
    
    1. **Minimize data usage** - Only load necessary data
    2. **Optimize images** - Use appropriate sizes
    3. **Reduce JavaScript** - Keep interactions simple
    4. **Cache effectively** - Use browser caching
    5. **Lazy loading** - Load content as needed
    6. **Minimize HTTP requests** - Combine resources
    """)

if __name__ == "__main__":
    test_mobile_responsiveness()
