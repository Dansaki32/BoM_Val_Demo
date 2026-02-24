import streamlit as st
from pathlib import Path
import base64
import pandas as pd

def load_custom_font_and_theme():
    font_path = Path("Styling/Styling/qr_font.ttf")
    logo_path = Path("Styling/Styling/qr_logo.png")
    font_css = ""
    if font_path.exists():
        with open(font_path, "rb") as f:
            font_data = f.read()
        b64_font = base64.b64encode(font_data).decode()
        font_css = f"""
            @font-face {{
                font-family: 'QRFont';
                src: url(data:font/ttf;base64,{b64_font}) format('truetype');
                font-weight: normal;
                font-style: normal;
            }}
            html, body, [class*='css']  {{
                font-family: 'QRFont', Arial, sans-serif;
            }}
        """
    # Slack-inspired color scheme
    # System navigation: #AD1212
    # Presence indication: #D63030
    # Selected items: #FF81AA
    # Notifications: #FFFFFF
    # Accent: #AD1212, Secondary: #D63030, Highlight: #FF81AA
    theme_css = f"""
        <style>
        {font_css}
        .qr-accent {{
            color: #AD1212;
            font-weight: bold;
        }}
        .qr-error {{
            background: #FFD6D6;
            color: #AD1212;
            border-left: 5px solid #D63030;
            padding: 0.5em 1em;
            margin-bottom: 1em;
            border-radius: 4px;
        }}
        .stButton>button {{
            background-color: #AD1212;
            color: #FFF;
            border-radius: 6px;
            border: none;
            font-weight: bold;
        }}
        .stButton>button:hover {{
            background-color: #D63030;
            color: #FFF;
        }}
        .stSidebar {{
            background-color: #FFF8FA !important;
        }}
        .stRadio>div>label[data-baseweb="radio"]>div:first-child {{
            border-color: #AD1212 !important;
        }}
        .stRadio>div>label[data-baseweb="radio"]>div:last-child {{
            color: #AD1212 !important;
        }}
        .st-bb {{
            color: #AD1212 !important;
        }}
        .st-bc {{
            background: #FF81AA !important;
        }}
        .stAlert {{
            border-left: 5px solid #AD1212 !important;
        }}
        </style>
    """
    st.markdown(theme_css, unsafe_allow_html=True)

def show_logo():
    logo_path = "Styling/Styling/qr_logo.png"
    if Path(logo_path).exists():
        st.sidebar.image(logo_path, use_column_width=True)

st.set_page_config(page_title="Feature Code Validator", layout="wide")
load_custom_font_and_theme()
show_logo()
st.sidebar.title("Feature Code Validator")

# --- Sidebar Navigation ---
page = st.sidebar.radio("Navigation", ["Dashboard", "Upload & Validate", "About"], index=0)

st.markdown("<h1 class='qr-accent'>OEM Feature Code Buildability Checker</h1>", unsafe_allow_html=True)

if page == "Dashboard":
    st.header("Dashboard")
    # Try to load the provided file if it exists
    file_path = Path("/workspaces/BoM_Val_Demo/FALTTUYY_20241126_Latest.xlsm")
    if file_path.exists():
        try:
            df = pd.read_excel(file_path, engine="openpyxl")
            st.success(f"Loaded file: {file_path.name}")
            st.write("**Preview:**")
            st.dataframe(df.head(20))
            st.write(f"**Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}")
            st.write(f"**Columns:** {', '.join(df.columns.astype(str).tolist())}")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("Upload a file to see buildability results.")
elif page == "Upload & Validate":
    st.header("Upload Feature Code File")
    uploaded_file = st.file_uploader("Upload Feature Code File", type=["csv", "xlsx", "xlsm"])
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
            st.success("File uploaded and parsed!")
            st.write("**Preview:**")
            st.dataframe(df.head(20))
            st.write(f"**Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}")
            st.write(f"**Columns:** {', '.join(df.columns.astype(str).tolist())}")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("Please upload a feature code file to begin.")
elif page == "About":
    st.header("About")
    st.markdown("""
        This tool helps interrogate OEM (Ford) vehicle feature codes for buildability and feature interactions. 
        Inspired by quickrelease.co.uk styling and Slack dark theme colors. 
        
        - Upload a configuration file
        - Review errors and build problems
        - Drill down into each part for details
    """)

