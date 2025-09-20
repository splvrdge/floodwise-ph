import streamlit as st
import streamlit.components.v1 as components

def inject_mobile_meta_tags():
    """Inject mobile-specific meta tags and PWA capabilities."""
    mobile_meta_html = """
    <head>
        <!-- Mobile viewport optimization -->
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, shrink-to-fit=no">
        
        <!-- Mobile web app capabilities -->
        <meta name="mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="default">
        <meta name="apple-mobile-web-app-title" content="FloodWise PH">
        
        <!-- Theme colors for mobile browsers -->
        <meta name="theme-color" content="#667eea">
        <meta name="msapplication-navbutton-color" content="#667eea">
        <meta name="apple-mobile-web-app-status-bar-style" content="#667eea">
        
        <!-- Prevent text size adjustment on mobile -->
        <meta name="format-detection" content="telephone=no">
        <meta name="format-detection" content="address=no">
        
        <!-- Favicon and app icons -->
        <link rel="icon" type="image/png" sizes="32x32" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🌊</text></svg>">
        <link rel="apple-touch-icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🌊</text></svg>">
        
        <!-- PWA manifest -->
        <link rel="manifest" href="data:application/json,{
            'name': 'FloodWise PH',
            'short_name': 'FloodWise',
            'description': 'Philippines Flood Control Intelligence Platform',
            'start_url': '/',
            'display': 'standalone',
            'background_color': '#ffffff',
            'theme_color': '#667eea',
            'icons': [
                {
                    'src': 'data:image/svg+xml,<svg xmlns=\\'http://www.w3.org/2000/svg\\' viewBox=\\'0 0 100 100\\'><text y=\\'.9em\\' font-size=\\'90\\'>🌊</text></svg>',
                    'sizes': '192x192',
                    'type': 'image/svg+xml'
                }
            ]
        }">
        
        <!-- Additional mobile optimizations -->
        <style>
            /* Prevent text selection on mobile for better UX */
            .mobile-no-select {
                -webkit-user-select: none;
                -moz-user-select: none;
                -ms-user-select: none;
                user-select: none;
                -webkit-touch-callout: none;
            }
            
            /* Smooth scrolling for mobile */
            html {
                -webkit-overflow-scrolling: touch;
                scroll-behavior: smooth;
            }
            
            /* Hide scrollbars on mobile while keeping functionality */
            @media (max-width: 768px) {
                ::-webkit-scrollbar {
                    width: 0px;
                    background: transparent;
                }
                
                /* Improve tap targets */
                button, input, select, textarea, a {
                    -webkit-tap-highlight-color: rgba(0,0,0,0.1);
                    tap-highlight-color: rgba(0,0,0,0.1);
                }
                
                /* Prevent zoom on input focus for iOS */
                input[type="text"], 
                input[type="email"], 
                input[type="password"], 
                input[type="number"], 
                textarea, 
                select {
                    font-size: 16px !important;
                }
            }
            
            /* Loading animation for mobile */
            .mobile-loading {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            /* Mobile-friendly tooltips */
            .mobile-tooltip {
                position: relative;
                display: inline-block;
            }
            
            .mobile-tooltip .tooltiptext {
                visibility: hidden;
                width: 200px;
                background-color: #555;
                color: white;
                text-align: center;
                border-radius: 6px;
                padding: 5px;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                margin-left: -100px;
                opacity: 0;
                transition: opacity 0.3s;
                font-size: 14px;
            }
            
            .mobile-tooltip:hover .tooltiptext,
            .mobile-tooltip:active .tooltiptext {
                visibility: visible;
                opacity: 1;
            }
        </style>
        
        <!-- Service Worker for PWA (basic implementation) -->
        <script>
            if ('serviceWorker' in navigator) {
                window.addEventListener('load', function() {
                    navigator.serviceWorker.register('data:text/javascript,console.log("SW registered")')
                        .then(function(registration) {
                            console.log('ServiceWorker registration successful');
                        }, function(err) {
                            console.log('ServiceWorker registration failed: ', err);
                        });
                });
            }
            
            // Add to home screen prompt for mobile
            let deferredPrompt;
            window.addEventListener('beforeinstallprompt', (e) => {
                e.preventDefault();
                deferredPrompt = e;
                
                // Show install button or banner
                const installButton = document.createElement('button');
                installButton.textContent = 'Install App';
                installButton.style.cssText = `
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    background: #667eea;
                    color: white;
                    border: none;
                    padding: 10px 15px;
                    border-radius: 25px;
                    font-size: 14px;
                    cursor: pointer;
                    z-index: 1000;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                `;
                
                installButton.addEventListener('click', () => {
                    deferredPrompt.prompt();
                    deferredPrompt.userChoice.then((choiceResult) => {
                        if (choiceResult.outcome === 'accepted') {
                            console.log('User accepted the install prompt');
                        }
                        deferredPrompt = null;
                        installButton.remove();
                    });
                });
                
                document.body.appendChild(installButton);
                
                // Auto-hide after 10 seconds
                setTimeout(() => {
                    if (installButton.parentNode) {
                        installButton.remove();
                    }
                }, 10000);
            });
            
            // Mobile-specific JavaScript optimizations
            document.addEventListener('DOMContentLoaded', function() {
                // Prevent double-tap zoom on mobile
                let lastTouchEnd = 0;
                document.addEventListener('touchend', function (event) {
                    const now = (new Date()).getTime();
                    if (now - lastTouchEnd <= 300) {
                        event.preventDefault();
                    }
                    lastTouchEnd = now;
                }, false);
                
                // Add mobile class to body for CSS targeting
                if (window.innerWidth <= 768) {
                    document.body.classList.add('mobile-device');
                }
                
                // Handle orientation changes
                window.addEventListener('orientationchange', function() {
                    setTimeout(function() {
                        window.scrollTo(0, 0);
                    }, 100);
                });
            });
        </script>
    </head>
    """
    
    # Inject the HTML
    components.html(mobile_meta_html, height=0)

def add_mobile_navigation():
    """Add mobile-specific navigation elements - DISABLED per user request."""
    # Mobile navigation removed per user request
    pass

def optimize_for_mobile():
    """Apply comprehensive mobile optimizations."""
    # Inject meta tags and PWA capabilities
    inject_mobile_meta_tags()
    
    # Add mobile navigation
    add_mobile_navigation()
    
    # Additional mobile-specific CSS
    st.markdown("""
    <style>
        /* Mobile-specific improvements */
        @media (max-width: 768px) {
            /* Improve readability on mobile */
            .stMarkdown p {
                line-height: 1.6;
                font-size: 16px;
            }
            
            /* Better spacing for mobile */
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
                margin-top: 1rem;
                margin-bottom: 0.5rem;
            }
            
            /* Mobile-friendly alerts */
            .stAlert {
                margin: 0.5rem 0;
                padding: 0.75rem;
                border-radius: 8px;
            }
            
            /* Improve form elements on mobile */
            .stTextInput input {
                padding: 12px;
                border-radius: 8px;
                border: 2px solid #e0e0e0;
                transition: border-color 0.3s ease;
            }
            
            .stTextInput input:focus {
                border-color: #667eea;
                outline: none;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            
            /* Mobile-friendly buttons */
            .stButton button {
                padding: 12px 24px;
                border-radius: 25px;
                font-weight: 600;
                text-transform: none;
                transition: all 0.3s ease;
            }
            
            /* Improve selectbox on mobile */
            .stSelectbox select {
                padding: 12px;
                border-radius: 8px;
                font-size: 16px;
            }
            
            /* Better file uploader on mobile */
            .stFileUploader {
                border: 2px dashed #667eea;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
                background: #f8f9fa;
            }
            
            /* Mobile-friendly expanders */
            .streamlit-expanderHeader {
                padding: 12px;
                border-radius: 8px;
                background: #f8f9fa;
                margin: 8px 0;
            }
            
            /* Improve dataframe display on mobile */
            .stDataFrame {
                font-size: 12px;
                overflow-x: auto;
            }
            
            .stDataFrame table {
                min-width: 100%;
            }
            
            /* Mobile-friendly metrics */
            .metric-container {
                background: white;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin: 10px 0;
            }
        }
        
        /* Tablet optimizations */
        @media (min-width: 769px) and (max-width: 1024px) {
            .main .block-container {
                max-width: 90%;
                padding: 2rem;
            }
        }
        
        /* Accessibility improvements */
        button:focus,
        input:focus,
        select:focus,
        textarea:focus {
            outline: 2px solid #667eea;
            outline-offset: 2px;
        }
        
        /* High contrast mode support */
        @media (prefers-contrast: high) {
            .stButton button {
                border: 2px solid currentColor;
            }
            
            .user-message,
            .assistant-message {
                border: 1px solid currentColor;
            }
        }
        
        /* Reduced motion support */
        @media (prefers-reduced-motion: reduce) {
            * {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)
