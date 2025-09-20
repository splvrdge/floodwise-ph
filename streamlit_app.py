# Streamlit App Entry Point
# This file ensures compatibility with Streamlit Cloud deployment

from app import main, show_footer

if __name__ == "__main__":
    main()
    show_footer()
