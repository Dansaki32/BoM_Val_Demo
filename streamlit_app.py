import streamlit as st
from pathlib import Path
import base64
import pandas as pd
from typing import Optional
import io

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class Config:
    """Centralized configuration"""
    FONT_PATH = Path("Styling/Styling/qr_font.ttf")
    LOGO_PATH = Path("Styling/Styling/qr_logo.png")
    DEFAULT_FILE_PATH = Path("/workspaces/BoM_Val_Demo/FALTTUYY_20241126_Latest.xlsm")
    
    # Color scheme
    COLORS = {
        'primary': '#AD1212',
        'secondary': '#D63030',
        'highlight': '#FF81AA',
        'background': '#18191A',
        'sidebar_bg': '#1A1D21',
        'card_bg': '#23272A',
        'error_bg': '#3A2323',
        'text': '#FFFFFF',
    }

# ============================================================================
# STYLING & THEME
# ============================================================================

@st.cache_data
def load_custom_font() -> str:
    """Load custom font and return CSS"""
    if not Config.FONT_PATH.exists():
        return ""
    
    try:
        with open(Config.FONT_PATH, "rb") as f:
            font_data = f.read()
        b64_font = base64.b64encode(font_data).decode()
        return f"""
            @font-face {{
                font-family: 'QRFont';
                src: url(data:font/ttf;base64,{b64_font}) format('truetype');
                font-weight: normal;
                font-style: normal;
            }}
        """
    except Exception as e:
        st.warning(f"Could not load custom font: {e}")
        return ""

def apply_custom_theme():
    """Apply custom CSS theme"""
    font_css = load_custom_font()
    c = Config.COLORS
    
    theme_css = f"""
        <style>
        {font_css}
        
        /* Global Styles */
        html, body, [class*='css'], .stApp {{
            background-color: {c['background']} !important;
            color: {c['text']} !important;
            font-family: 'QRFont', -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
        }}
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            color: {c['text']} !important;
        }}
        
        .qr-accent {{
            color: {c['highlight']};
            font-weight: bold;
        }}
        
        /* Buttons */
        .stButton>button {{
            background-color: {c['primary']};
            color: {c['text']};
            border-radius: 6px;
            border: none;
            font-weight: bold;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: {c['secondary']};
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }}
        
        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {c['sidebar_bg']} !important;
        }}
        section[data-testid="stSidebar"] > div {{
            background-color: {c['sidebar_bg']} !important;
        }}
        
        /* Radio Buttons */
        .stRadio > label {{
            color: {c['text']} !important;
        }}
        .stRadio > div {{
            background-color: transparent !important;
        }}
        
        /* File Uploader */
        .stFileUploader > div {{
            background-color: {c['card_bg']} !important;
            border: 2px dashed {c['highlight']} !important;
            border-radius: 8px;
            padding: 1rem;
        }}
        
        /* DataFrames */
        .stDataFrame, .stTable {{
            background-color: {c['card_bg']} !important;
        }}
        .stDataFrame [data-testid="stDataFrameResizable"] {{
            background-color: {c['card_bg']} !important;
        }}
        
        /* Alerts */
        .stAlert {{
            background-color: {c['error_bg']} !important;
            border-left: 5px solid {c['primary']} !important;
            border-radius: 4px;
        }}
        .stSuccess {{
            border-left-color: #28a745 !important;
        }}
        .stWarning {{
            border-left-color: #ffc107 !important;
        }}
        .stError {{
            border-left-color: {c['primary']} !important;
        }}
        
        /* Info Cards */
        .info-card {{
            background-color: {c['card_bg']};
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid {c['highlight']};
            margin: 1rem 0;
        }}
        
        /* Metrics */
        [data-testid="stMetricValue"] {{
            color: {c['highlight']} !important;
        }}
        
        /* Expander */
        .streamlit-expanderHeader {{
            background-color: {c['card_bg']} !important;
            border-radius: 4px;
        }}
        
        /* Input Fields */
        .stTextInput > div > div > input {{
            background-color: {c['card_bg']} !important;
            color: {c['text']} !important;
            border: 1px solid {c['highlight']} !important;
        }}
        
        /* Selectbox */
        .stSelectbox > div > div {{
            background-color: {c['card_bg']} !important;
            color: {c['text']} !important;
        }}
        
        /* Progress Bar */
        .stProgress > div > div > div {{
            background-color: {c['primary']} !important;
        }}
        
        </style>
    """
    st.markdown(theme_css, unsafe_allow_html=True)

def show_logo():
    """Display logo in sidebar if available"""
    if Config.LOGO_PATH.exists():
        try:
            st.sidebar.image(str(Config.LOGO_PATH), use_column_width=True)
        except Exception as e:
            st.sidebar.warning("Logo could not be loaded")
    else:
        # Fallback: Show text logo
        st.sidebar.markdown(
            "<h2 style='text-align: center; color: #FF81AA;'>🔧 QR</h2>", 
            unsafe_allow_html=True
        )

# ============================================================================
# DATA HANDLING
# ============================================================================

@st.cache_data
def load_dataframe(file_path: Path) -> Optional[pd.DataFrame]:
    """Load DataFrame from file path with error handling"""
    if not file_path.exists():
        return None
    
    try:
        if file_path.suffix in ['.xlsx', '.xlsm']:
            return pd.read_excel(file_path, engine="openpyxl")
        elif file_path.suffix == '.csv':
            return pd.read_csv(file_path)
        else:
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def load_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Load DataFrame from uploaded file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        return None

def validate_feature_codes(df: pd.DataFrame) -> dict:
    """
    Placeholder validation logic for feature codes
    Returns a dictionary with validation results
    """
    results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'errors': [],
        'warnings': [],
        'passed': True
    }
    
    # Example validation checks
    if df.empty:
        results['errors'].append("DataFrame is empty")
        results['passed'] = False
    
    # Check for required columns (customize based on your needs)
    required_columns = ['Feature_Code', 'Part_Number']  # Example columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        results['warnings'].append(f"Missing recommended columns: {', '.join(missing_columns)}")
    
    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        results['warnings'].append(f"Found null values in {null_counts[null_counts > 0].to_dict()}")
    
    # Add more validation logic here
    
    return results

# ============================================================================
# UI COMPONENTS
# ============================================================================

def display_dataframe_info(df: pd.DataFrame):
    """Display DataFrame information in a nice format"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory Usage", f"{memory_usage:.2f} MB")
    
    with st.expander("📋 Column Details", expanded=False):
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Non-Null': df.count().values,
            'Null': df.isnull().sum().values
        })
        st.dataframe(col_info, use_container_width=True)

def display_validation_results(results: dict):
    """Display validation results with color coding"""
    st.subheader("Validation Results")
    
    if results['passed']:
        st.success("✅ All validation checks passed!")
    else:
        st.error("❌ Validation failed - see errors below")
    
    if results['errors']:
        st.markdown("### Errors")
        for error in results['errors']:
            st.error(f"❌ {error}")
    
    if results['warnings']:
        st.markdown("### Warnings")
        for warning in results['warnings']:
            st.warning(f"⚠️ {warning}")

# ============================================================================
# PAGES
# ============================================================================

def page_dashboard():
    """Dashboard page"""
    st.header("📊 Dashboard")
    
    # Check if data is in session state
    if 'current_df' in st.session_state and st.session_state.current_df is not None:
        df = st.session_state.current_df
        st.success(f"✅ Loaded: {st.session_state.get('current_filename', 'Uploaded file')}")
        
        # Display metrics
        display_dataframe_info(df)
        
        # Data preview
        st.subheader("Data Preview")
        preview_rows = st.slider("Rows to display", 5, 100, 20)
        st.dataframe(df.head(preview_rows), use_container_width=True)
        
        # Validation
        if st.button("🔍 Run Validation", type="primary"):
            with st.spinner("Validating..."):
                results = validate_feature_codes(df)
                st.session_state.validation_results = results
                display_validation_results(results)
        
        # Show previous validation results if available
        elif 'validation_results' in st.session_state:
            display_validation_results(st.session_state.validation_results)
            
    else:
        # Try to load default file
        if Config.DEFAULT_FILE_PATH.exists():
            df = load_dataframe(Config.DEFAULT_FILE_PATH)
            if df is not None:
                st.session_state.current_df = df
                st.session_state.current_filename = Config.DEFAULT_FILE_PATH.name
                st.rerun()
        else:
            st.info("📁 No data loaded. Please upload a file in the 'Upload & Validate' section.")
            
            # Show example structure
            with st.expander("💡 Expected File Structure"):
                st.markdown("""
                Your file should contain:
                - **Feature_Code**: The feature code identifier
                - **Part_Number**: Associated part number
                - **Description**: Part description
                - Additional columns as needed
                
                Supported formats: CSV, XLSX, XLSM
                """)

def page_upload_validate():
    """Upload and validate page"""
    st.header("📤 Upload Feature Code File")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xlsm"],
        help="Upload a CSV or Excel file containing feature codes"
    )
    
    if uploaded_file:
        with st.spinner("Loading file..."):
            df = load_uploaded_file(uploaded_file)
            
        if df is not None:
            # Store in session state
            st.session_state.current_df = df
            st.session_state.current_filename = uploaded_file.name
            
            st.success(f"✅ Successfully loaded: {uploaded_file.name}")
            
            # Display info
            display_dataframe_info(df)
            
            # Preview
            st.subheader("Data Preview")
            st.dataframe(df.head(20), use_container_width=True)
            
            # Download button for processed data
            st.subheader("Export Options")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"processed_{uploaded_file.name}.csv",
                    mime="text/csv",
                )
            
            with col2:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False)
                st.download_button(
                    label="📥 Download as Excel",
                    data=buffer.getvalue(),
                    file_name=f"processed_{uploaded_file.name}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    else:
        st.info("👆 Please upload a file to begin")
        
        # Sample data option
        if st.button("Generate Sample Data"):
            sample_df = pd.DataFrame({
                'Feature_Code': ['FC001', 'FC002', 'FC003', 'FC004', 'FC005'],
                'Part_Number': ['PN-001', 'PN-002', 'PN-003', 'PN-004', 'PN-005'],
                'Description': ['Part 1', 'Part 2', 'Part 3', 'Part 4', 'Part 5'],
                'Quantity': [1, 2, 1, 3, 1],
                'Buildable': [True, True, False, True, True]
            })
            st.session_state.current_df = sample_df
            st.session_state.current_filename = "sample_data.csv"
            st.rerun()

def page_about():
    """About page"""
    st.header("ℹ️ About")
    
    st.markdown("""
    <div class='info-card'>
        <h3>🔧 OEM Feature Code Buildability Checker</h3>
        <p>
            This tool helps interrogate OEM (Ford) vehicle feature codes for buildability 
            and feature interactions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Key Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - ✅ Upload configuration files
        - 📊 Interactive data exploration
        - 🔍 Feature code validation
        - 📈 Visual analytics
        """)
    
    with col2:
        st.markdown("""
        - 💾 Export processed data
        - ⚡ Fast processing
        - 🎨 Custom dark theme
        - 📱 Responsive design
        """)
    
    st.markdown("### 🚀 How to Use")
    with st.expander("Step-by-step guide"):
        st.markdown("""
        1. **Upload your file** in the 'Upload & Validate' section
        2. **Review the data** preview and column information
        3. **Navigate to Dashboard** to see detailed analytics
        4. **Run validation** to check for errors and warnings
        5. **Export results** as needed
        """)
    
    st.markdown("### 📊 Supported File Formats")
    st.markdown("""
    - **CSV** (.csv) - Comma-separated values
    - **Excel** (.xlsx, .xlsm) - Microsoft Excel files
    """)
    
    st.markdown("### 🎨 Design")
    st.markdown("""
    Inspired by quickrelease.co.uk styling and Slack's dark theme for 
    a modern, professional appearance.
    """)
    
    # Version info
    st.markdown("---")
    st.caption("Version 2.0 | Built with Streamlit")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(
        page_title="Feature Code Validator",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply theme
    apply_custom_theme()
    
    # Initialize session state
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    if 'current_filename' not in st.session_state:
        st.session_state.current_filename = None
    
    # Sidebar
    show_logo()
    st.sidebar.title("Feature Code Validator")
    
    # Navigation
    page = st.sidebar.radio(
        "📍 Navigation",
        ["Dashboard", "Upload & Validate", "About"],
        index=0
    )
    
    # Show current file in sidebar
    if st.session_state.current_filename:
        st.sidebar.success(f"📄 Current: {st.session_state.current_filename}")
        if st.sidebar.button("🗑️ Clear Data"):
            st.session_state.current_df = None
            st.session_state.current_filename = None
            st.session_state.validation_results = None
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Main title
    st.markdown(
        "<h1 class='qr-accent'>🔧 OEM Feature Code Buildability Checker</h1>",
        unsafe_allow_html=True
    )
    
    # Route to appropriate page
    if page == "Dashboard":
        page_dashboard()
    elif page == "Upload & Validate":
        page_upload_validate()
    elif page == "About":
        page_about()

if __name__ == "__main__":
    main()
