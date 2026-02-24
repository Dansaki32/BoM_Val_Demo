
import streamlit as st
from pathlib import Path
import base64
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import io
from datetime import datetime

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class Config:
    """Centralized configuration"""
    FONT_PATH = Path("Styling/Styling/qr_font.ttf")
    LOGO_PATH = Path("Styling/Styling/qr_logo.png")
    DEFAULT_FILE_PATH = Path("/workspaces/BoM_Val_Demo/FALTTUYY_20241126_Latest.xlsm")
    
    # QuickRelease.co.uk inspired color scheme
    COLORS = {
        'primary': '#AD1212',
        'secondary': '#D63030',
        'highlight': '#FF81AA',
        'background': '#18191A',
        'sidebar_bg': '#1A1D21',
        'card_bg': '#23272A',
        'error_bg': '#3A2323',
        'success': '#28a745',
        'warning': '#ffc107',
        'text': '#FFFFFF',
        'text_muted': '#B0B0B0',
    }
    
    SEVERITY_COLORS = {
        'CRITICAL': '#AD1212',
        'ERROR': '#D63030',
        'WARNING': '#ffc107',
        'INFO': '#17a2b8'
    }

# ============================================================================
# STYLING & THEME
# ============================================================================

@st.cache_data
def load_custom_font() -> str:
    """Load custom font and return CSS - with error handling"""
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
    except Exception:
        return ""

def apply_custom_theme():
    """Apply QuickRelease.co.uk inspired theme with improved visibility"""
    font_css = load_custom_font()
    c = Config.COLORS
    
    theme_css = f"""
        <style>
        {font_css}
        
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*='css'], .stApp {{
            background-color: {c['background']} !important;
            color: {c['text']} !important;
            font-family: 'QRFont', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
        }}
        
        /* Fix text visibility */
        p, span, div, label, .stMarkdown {{
            color: {c['text']} !important;
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            color: {c['text']} !important;
            font-weight: 600 !important;
        }}
        
        .main-title {{
            background: linear-gradient(135deg, {c['primary']} 0%, {c['secondary']} 100%);
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        
        .stButton>button {{
            background: linear-gradient(135deg, {c['primary']} 0%, {c['secondary']} 100%);
            color: {c['text']} !important;
            border-radius: 8px;
            border: none;
            font-weight: 600;
            padding: 0.75rem 1.5rem;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(173, 18, 18, 0.4);
        }}
        
        section[data-testid="stSidebar"] {{
            background-color: {c['sidebar_bg']} !important;
            border-right: 2px solid {c['primary']};
        }}
        
        section[data-testid="stSidebar"] * {{
            color: {c['text']} !important;
        }}
        
        .stRadio > label {{
            color: {c['text']} !important;
            font-weight: 500;
        }}
        
        .stRadio > div > label {{
            background-color: {c['card_bg']} !important;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
            transition: all 0.2s ease;
            border: 2px solid transparent;
            color: {c['text']} !important;
        }}
        
        .stRadio > div > label:hover {{
            background-color: {c['primary']} !important;
            border-color: {c['highlight']};
        }}
        
        .stFileUploader {{
            background-color: {c['card_bg']} !important;
            border: 2px dashed {c['highlight']} !important;
            border-radius: 12px;
            padding: 2rem;
        }}
        
        .stFileUploader label {{
            color: {c['text']} !important;
        }}
        
        .stDataFrame {{
            background-color: {c['card_bg']} !important;
        }}
        
        .dataframe {{
            background-color: {c['card_bg']} !important;
            color: {c['text']} !important;
        }}
        
        .dataframe thead th {{
            background-color: {c['primary']} !important;
            color: {c['text']} !important;
            font-weight: 600;
        }}
        
        .dataframe tbody td {{
            color: {c['text']} !important;
        }}
        
        .stAlert {{
            background-color: {c['error_bg']} !important;
            color: {c['text']} !important;
            border-left: 4px solid {c['primary']} !important;
        }}
        
        .info-card {{
            background: linear-gradient(135deg, {c['card_bg']} 0%, rgba(173, 18, 18, 0.1) 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 4px solid {c['highlight']};
            margin: 1rem 0;
            color: {c['text']} !important;
        }}
        
        .error-card {{
            background: linear-gradient(135deg, {c['error_bg']} 0%, rgba(173, 18, 18, 0.2) 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 4px solid {c['primary']};
            margin: 1rem 0;
            color: {c['text']} !important;
        }}
        
        .metric-container {{
            background-color: {c['card_bg']};
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            border: 2px solid transparent;
            transition: all 0.3s ease;
        }}
        
        .metric-container:hover {{
            border-color: {c['highlight']};
            transform: translateY(-2px);
        }}
        
        [data-testid="stMetricValue"] {{
            color: {c['highlight']} !important;
        }}
        
        [data-testid="stMetricLabel"] {{
            color: {c['text_muted']} !important;
        }}
        
        .streamlit-expanderHeader {{
            background-color: {c['card_bg']} !important;
            color: {c['text']} !important;
        }}
        
        .stTextInput > div > div > input {{
            background-color: {c['card_bg']} !important;
            color: {c['text']} !important;
            border: 2px solid {c['highlight']} !important;
        }}
        
        .stSelectbox {{
            color: {c['text']} !important;
        }}
        
        /* Fix for success/warning/error messages */
        .stSuccess {{
            background-color: rgba(40, 167, 69, 0.1) !important;
            color: {c['text']} !important;
            border-left-color: {c['success']} !important;
        }}
        
        .stWarning {{
            background-color: rgba(255, 193, 7, 0.1) !important;
            color: {c['text']} !important;
            border-left-color: {c['warning']} !important;
        }}
        
        .stInfo {{
            color: {c['text']} !important;
        }}
        
        /* Scrollbar */
        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {c['background']};
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {c['primary']};
            border-radius: 5px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {c['secondary']};
        }}
        
        </style>
    """
    st.markdown(theme_css, unsafe_allow_html=True)

def show_logo():
    """Display logo in sidebar with error handling"""
    if Config.LOGO_PATH.exists():
        try:
            st.sidebar.image(str(Config.LOGO_PATH), use_column_width=True)
        except Exception:
            st.sidebar.markdown(
                "<h2 style='text-align: center; color: #FF81AA;'>🔧 Feature Validator</h2>", 
                unsafe_allow_html=True
            )
    else:
        st.sidebar.markdown(
            "<h2 style='text-align: center; color: #FF81AA;'>🔧 Feature Validator</h2>", 
            unsafe_allow_html=True
        )

# ============================================================================
# DATA MODELS
# ============================================================================

class ValidationResult:
    """Data class for validation results"""
    def __init__(self, part_number: str, feature_code: str, severity: str, 
                 message: str, details: str = ""):
        self.part_number = part_number
        self.feature_code = feature_code
        self.severity = severity
        self.message = message
        self.details = details
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            'Part Number': self.part_number,
            'Feature Code': self.feature_code,
            'Severity': self.severity,
            'Message': self.message,
            'Details': self.details,
            'Timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }

# ============================================================================
# DATA HANDLING & VALIDATION
# ============================================================================

@st.cache_data
def load_dataframe(file_path: Path) -> Optional[pd.DataFrame]:
    """Load DataFrame from file path"""
    if not file_path.exists():
        return None
    
    try:
        if file_path.suffix in ['.xlsx', '.xlsm']:
            return pd.read_excel(file_path, engine="openpyxl")
        elif file_path.suffix == '.csv':
            return pd.read_csv(file_path)
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

def validate_feature_codes(df: pd.DataFrame) -> Tuple[List[ValidationResult], Dict]:
    """Validate feature codes and return results"""
    results = []
    stats = {
        'total_parts': len(df),
        'total_features': 0,
        'critical': 0,
        'errors': 0,
        'warnings': 0,
        'info': 0,
        'valid_parts': 0
    }
    
    if df.empty:
        results.append(ValidationResult(
            'N/A', 'N/A', 'CRITICAL',
            'DataFrame is empty',
            'No data available for validation'
        ))
        stats['critical'] += 1
        return results, stats
    
    required_columns = ['Feature_Code', 'Part_Number']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        results.append(ValidationResult(
            'N/A', 'N/A', 'ERROR',
            f'Missing required columns: {", ".join(missing_columns)}',
            'These columns are required for validation'
        ))
        stats['errors'] += 1
    
    for idx, row in df.iterrows():
        part_num = row.get('Part_Number', f'Row {idx}')
        feature_code = row.get('Feature_Code', 'N/A')
        
        stats['total_features'] += 1
        
        if pd.isna(feature_code):
            results.append(ValidationResult(
                part_num, 'N/A', 'ERROR',
                'Missing feature code',
                f'Part {part_num} has no feature code assigned'
            ))
            stats['errors'] += 1
            continue
        
        if not str(feature_code).strip():
            results.append(ValidationResult(
                part_num, feature_code, 'ERROR',
                'Empty feature code',
                'Feature code is empty or whitespace only'
            ))
            stats['errors'] += 1
            continue
        
        buildable = row.get('Buildable', True)
        if not buildable:
            results.append(ValidationResult(
                part_num, feature_code, 'CRITICAL',
                'Part not buildable',
                f'Feature code {feature_code} configuration is not buildable'
            ))
            stats['critical'] += 1
            continue
        
        quantity = row.get('Quantity', 1)
        if quantity > 5:
            results.append(ValidationResult(
                part_num, feature_code, 'WARNING',
                'High quantity',
                f'Quantity {quantity} is unusually high for this part'
            ))
            stats['warnings'] += 1
        
        if not any(r.part_number == part_num and r.severity in ['CRITICAL', 'ERROR'] 
                   for r in results):
            stats['valid_parts'] += 1
    
    return results, stats

def get_part_details(df: pd.DataFrame, part_number: str) -> Dict:
    """Get detailed information for a specific part"""
    part_data = df[df['Part_Number'] == part_number]
    if part_data.empty:
        return {}
    return part_data.iloc[0].to_dict()

# ============================================================================
# UI COMPONENTS
# ============================================================================

def display_summary_metrics(stats: Dict):
    """Display summary metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Parts", f"{stats['total_parts']:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Critical Issues", stats['critical'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Errors", stats['errors'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Warnings", stats['warnings'])
        st.markdown('</div>', unsafe_allow_html=True)

def display_validation_results(results: List[ValidationResult], stats: Dict):
    """Display validation results with filtering"""
    st.markdown("### 🔍 Validation Results")
    
    if not results:
        st.success("✅ No issues found! All parts are valid.")
        return
    
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        severity_filter = st.multiselect(
            "Filter by Severity",
            options=['CRITICAL', 'ERROR', 'WARNING', 'INFO'],
            default=['CRITICAL', 'ERROR', 'WARNING', 'INFO']
        )
    
    with col2:
        search_part = st.text_input("🔎 Search Part Number", "")
    
    with col3:
        search_feature = st.text_input("🔎 Search Feature Code", "")
    
    filtered_results = [
        r for r in results
        if r.severity in severity_filter
        and (not search_part or search_part.lower() in r.part_number.lower())
        and (not search_feature or search_feature.lower() in r.feature_code.lower())
    ]
    
    st.markdown(f"**Showing {len(filtered_results)} of {len(results)} issues**")
    
    for result in filtered_results:
        severity_color = Config.SEVERITY_COLORS.get(result.severity, '#FFFFFF')
        
        with st.expander(
            f"{result.severity}: {result.part_number} - {result.feature_code}",
            expanded=False
        ):
            st.markdown(f'<div class="error-card">', unsafe_allow_html=True)
            st.markdown(f'**Severity:** <span style="color: {severity_color}; font-weight: bold;">{result.severity}</span>', 
                       unsafe_allow_html=True)
            st.markdown(f'**Part Number:** {result.part_number}')
            st.markdown(f'**Feature Code:** {result.feature_code}')
            st.markdown(f'**Message:** {result.message}')
            if result.details:
                st.markdown(f'**Details:** {result.details}')
            st.markdown(f'**Time:** {result.timestamp.strftime("%Y-%m-%d %H:%M:%S")}')
            st.markdown('</div>', unsafe_allow_html=True)

def display_dataframe_info(df: pd.DataFrame):
    """Display DataFrame information"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Rows", f"{df.shape[0]:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Columns", df.shape[1])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory Usage", f"{memory_usage:.2f} MB")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("📋 Column Details", expanded=False):
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Non-Null': df.count().values,
            'Null': df.isnull().sum().values,
            'Unique': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True, height=300)

def display_part_drill_down(df: pd.DataFrame, results: List[ValidationResult]):
    """Display part-level drill-down view"""
    st.markdown("### 🔎 Part Drill-Down")
    
    parts_with_issues = list(set([r.part_number for r in results if r.part_number != 'N/A']))
    
    if not parts_with_issues:
        st.info("No parts with issues to display")
        return
    
    selected_part = st.selectbox(
        "Select Part to Investigate",
        options=sorted(parts_with_issues)
    )
    
    if selected_part:
        part_issues = [r for r in results if r.part_number == selected_part]
        part_details = get_part_details(df, selected_part)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown(f"#### Part Information")
            for key, value in part_details.items():
                st.markdown(f"**{key}:** {value}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="error-card">', unsafe_allow_html=True)
            st.markdown(f"#### Issues Summary")
            st.metric("Total Issues", len(part_issues))
            critical = sum(1 for r in part_issues if r.severity == 'CRITICAL')
            errors = sum(1 for r in part_issues if r.severity == 'ERROR')
            warnings = sum(1 for r in part_issues if r.severity == 'WARNING')
            st.markdown(f"**Critical:** {critical} | **Errors:** {errors} | **Warnings:** {warnings}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("#### Detailed Issues")
        for issue in part_issues:
            severity_color = Config.SEVERITY_COLORS.get(issue.severity, '#FFFFFF')
            with st.expander(f"{issue.severity}: {issue.message}", expanded=True):
                st.markdown(f'<span style="color: {severity_color}; font-weight: bold;">{issue.severity}</span>', 
                           unsafe_allow_html=True)
                st.markdown(f"**Message:** {issue.message}")
                if issue.details:
                    st.markdown(f"**Details:** {issue.details}")

def display_analytics(results: List[ValidationResult], df: pd.DataFrame):
    """Display analytics and visualizations"""
    if results:
        results_df = pd.DataFrame([r.to_dict() for r in results])
        
        st.markdown("### 📈 Issue Severity Distribution")
        severity_counts = results_df['Severity'].value_counts()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.bar_chart(severity_counts)
        
        with col2:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("**Severity Breakdown:**")
            for severity, count in severity_counts.items():
                color = Config.SEVERITY_COLORS.get(severity, '#FFFFFF')
                st.markdown(
                    f'<span style="color: {color}; font-weight: bold;">{severity}</span>: {count}',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🔝 Most Common Issues")
        message_counts = results_df['Message'].value_counts().head(10)
        st.dataframe(
            message_counts.reset_index().rename(columns={'index': 'Issue', 'Message': 'Count'}),
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown("### ⚠️ Parts with Most Issues")
        part_issue_counts = results_df['Part Number'].value_counts().head(10)
        st.dataframe(
            part_issue_counts.reset_index().rename(columns={'index': 'Part Number', 'Part Number': 'Issue Count'}),
            use_container_width=True
        )
    else:
        st.success("✅ No issues found! All parts passed validation.")
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("**Numerical Columns Summary:**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        else:
            st.info("No numerical columns found")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("**Categorical Columns:**")
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols[:5]:
                unique_count = df[col].nunique()
                st.markdown(f"**{col}:** {unique_count} unique values")
        else:
            st.info("No categorical columns found")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PAGES
# ============================================================================

def page_dashboard():
    """Main dashboard page"""
    st.markdown('<div class="main-title"><h1>📊 Feature Code Validation Dashboard</h1></div>', 
                unsafe_allow_html=True)
    
    if 'current_df' in st.session_state and st.session_state.current_df is not None:
        df = st.session_state.current_df
        
        st.success(f"✅ Loaded: **{st.session_state.get('current_filename', 'Uploaded file')}**")
        
        display_dataframe_info(df)
        
        st.markdown("---")
        st.markdown("### 🔍 Run Validation")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("Click the button below to validate all feature codes and check for buildability issues.")
        with col2:
            validate_button = st.button("🚀 Run Validation", type="primary", use_container_width=True)
        
        if validate_button:
            with st.spinner("🔄 Validating feature codes..."):
                results, stats = validate_feature_codes(df)
                st.session_state.validation_results = results
                st.session_state.validation_stats = stats
        
        if 'validation_results' in st.session_state and 'validation_stats' in st.session_state:
            st.markdown("---")
            display_summary_metrics(st.session_state.validation_stats)
            
            st.markdown("---")
            tab1, tab2, tab3 = st.tabs(["📋 All Issues", "🔍 Part Drill-Down", "📊 Analytics"])
            
            with tab1:
                display_validation_results(
                    st.session_state.validation_results,
                    st.session_state.validation_stats
                )
            
            with tab2:
                display_part_drill_down(df, st.session_state.validation_results)
            
            with tab3:
                display_analytics(st.session_state.validation_results, df)
        
        st.markdown("---")
        st.markdown("### 📄 Data Preview")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            preview_rows = st.slider("Number of rows to display", 5, 100, 20)
        with col2:
            show_all_cols = st.checkbox("Show all columns", value=False)
        
        if show_all_cols:
            st.dataframe(df.head(preview_rows), use_container_width=True, height=400)
        else:
            display_cols = df.columns[:10] if len(df.columns) > 10 else df.columns
            st.dataframe(df[display_cols].head(preview_rows), use_container_width=True, height=400)
            if len(df.columns) > 10:
                st.info(f"ℹ️ Showing {len(display_cols)} of {len(df.columns)} columns")
    
    else:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### 👋 Welcome to the Feature Code Validator")
        st.markdown("""
        This tool helps you validate Ford OEM vehicle feature codes for buildability and feature interactions.
        
        **Getting Started:**
        1. 📁 Upload your configuration file in the 'Upload & Validate' section
        2. 🔍 Review the data preview and column information
        3. 🚀 Run validation to check for errors and warnings
        4. 📊 Analyze results and drill down into specific parts
        
        **Supported File Formats:**
        - CSV (.csv)
        - Excel (.xlsx, .xlsm)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if Config.DEFAULT_FILE_PATH.exists():
            if st.button("📂 Load Default File", type="primary"):
                df = load_dataframe(Config.DEFAULT_FILE_PATH)
                if df is not None:
                    st.session_state.current_df = df
                    st.session_state.current_filename = Config.DEFAULT_FILE_PATH.name
                    st.rerun()

def page_upload_validate():
    """Upload and validate page"""
    st.markdown('<div class="main-title"><h1>📤 Upload Feature Code File</h1></div>', 
                unsafe_allow_html=True)
    
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("""
    Upload your feature code configuration file to begin validation.
    
    **Supported formats:** CSV, XLSX, XLSM  
    **Maximum file size:** 200MB
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xlsm"],
        help="Upload a CSV or Excel file containing feature codes"
    )
    
    if uploaded_file:
        with st.spinner("📥 Loading file..."):
            df = load_uploaded_file(uploaded_file)
        
        if df is not None:
            # Store in session state
            st.session_state.current_df = df
            st.session_state.current_filename = uploaded_file.name
            
            # Clear previous validation results to ensure freshness
            if 'validation_results' in st.session_state:
                del st.session_state.validation_results
            if 'validation_stats' in st.session_state:
                del st.session_state.validation_stats
            
            st.success(f"✅ Successfully loaded: **{uploaded_file.name}**")
            
            # Display file information
            st.markdown("---")
            st.markdown("### 📊 File Information")
            display_dataframe_info(df)
            
            # Data preview
            st.markdown("---")
            st.markdown("### 📄 Data Preview")
            
            preview_rows = st.slider("Rows to preview", 5, 50, 20)
            st.dataframe(df.head(preview_rows), use_container_width=True, height=400)
            
            # Column information
            with st.expander("📋 Column Information", expanded=False):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.values,
                    'Non-Null Count': df.count().values,
                    'Null Count': df.isnull().sum().values,
                    'Unique Values': [df[col].nunique() for col in df.columns]
                })
                st.dataframe(col_info, use_container_width=True)
            
            # Export options
            st.markdown("---")
            st.markdown("### 💾 Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"processed_{uploaded_file.name}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                buffer = io.BytesIO()
                try:
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False)
                    st.download_button(
                        label="📥 Download as Excel",
                        data=buffer.getvalue(),
                        file_name=f"processed_{uploaded_file.name}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                except ImportError:
                    st.warning("⚠️ Install 'openpyxl' to enable Excel export.")
            
            with col3:
                if st.button("🔍 Go to Dashboard", type="primary", use_container_width=True):
                    # In a real app, you might control navigation via state
                    st.info("Navigate to the Dashboard tab to analyze this data.")
    
    else:
        # Sample data option if no file uploaded
        st.markdown("---")
        st.markdown("### 🧪 Or Try Sample Data")
        
        if st.button("Generate Sample Data", use_container_width=True):
            # Create realistic looking sample data
            data = {
                'Part_Number': [f'PN-{1000+i}' for i in range(50)],
                'Feature_Code': [f'FC{100+i:03d}' for i in range(50)],
                'Description': [f'Sample Part {i+1}' for i in range(50)],
                'Quantity': np.random.randint(1, 10, 50),
                'Buildable': np.random.choice([True, False], 50, p=[0.85, 0.15]),
                'Category': np.random.choice(['Engine', 'Transmission', 'Interior', 'Exterior'], 50),
                'Status': np.random.choice(['Active', 'Pending', 'Review'], 50)
            }
            
            # Inject some errors for demonstration
            data['Feature_Code'][5] = None  # Missing feature code
            data['Feature_Code'][10] = "   "  # Empty feature code
            data['Quantity'][15] = 100  # Warning level quantity
            
            sample_df = pd.DataFrame(data)
            
            st.session_state.current_df = sample_df
            st.session_state.current_filename = "sample_data_generated.csv"
            
            # Clear old results
            if 'validation_results' in st.session_state:
                del st.session_state.validation_results
            
            st.rerun()

def page_analytics():
    """Analytics and insights page"""
    st.markdown('<div class="main-title"><h1>📊 Analytics & Insights</h1></div>', 
                unsafe_allow_html=True)
    
    if 'current_df' not in st.session_state or st.session_state.current_df is None:
        st.warning("⚠️ No data loaded. Please upload a file first.")
        return
    
    df = st.session_state.current_df
    
    if 'validation_results' not in st.session_state:
        st.info("ℹ️ Run validation on the Dashboard to see detailed analytics.")
        if st.button("🚀 Run Validation Now", type="primary"):
            with st.spinner("🔄 Validating..."):
                results, stats = validate_feature_codes(df)
                st.session_state.validation_results = results
                st.session_state.validation_stats = stats
                st.rerun()
        return
    
    display_analytics(st.session_state.validation_results, df)

def page_about():
    """About page"""
    st.markdown('<div class="main-title"><h1>ℹ️ About</h1></div>', 
                unsafe_allow_html=True)
    
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("""
    ### 🔧 OEM Feature Code Buildability Checker
    
    This advanced validation tool helps interrogate Ford OEM vehicle feature codes for 
    buildability and complex feature interactions.
    
    #### 🎯 Key Features
    
    **Validation & Analysis**
    - ✅ Comprehensive feature code validation
    - 📊 Interactive data exploration
    - 🔍 Deep-dive part analysis
    - 📈 Visual analytics and reporting
    
    **User Experience**
    - ⚡ Fast, responsive interface
    - 🎨 QuickRelease.co.uk inspired design
    - 💾 Multiple export formats
    
    #### 🚀 How to Use
    
    1. **Upload Configuration File** in 'Upload & Validate'
    2. **Run Validation** in 'Dashboard'
    3. **Analyze Results** via the 'All Issues' tab or 'Analytics' page
    
    #### 🔐 Data Privacy
    - All processing happens locally in-memory
    - No data is sent to external servers
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # System information
    st.markdown("---")
    st.markdown("### 🖥️ System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown(f"""
        **Application:**
        - Version: 2.1.0
        - Build Date: {datetime.now().strftime('%Y-%m-%d')}
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown(f"""
        **Session Info:**
        - Data Loaded: {'Yes' if 'current_df' in st.session_state else 'No'}
        - Validation Run: {'Yes' if 'validation_results' in st.session_state else 'No'}
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(
        page_title="Feature Code Validator | Ford OEM",
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
    st.sidebar.title("Feature Validator")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "📍 Navigation",
        ["Dashboard", "Upload & Validate", "Analytics", "About"],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # Show current file info in sidebar
    if st.session_state.current_filename:
        st.sidebar.success(f"📄 **Current File:**  \n{st.session_state.current_filename}")
        
        if st.sidebar.button("🗑️ Clear Data", use_container_width=True):
            st.session_state.current_df = None
            st.session_state.current_filename = None
            if 'validation_results' in st.session_state:
                del st.session_state.validation_results
            if 'validation_stats' in st.session_state:
                del st.session_state.validation_stats
            st.rerun()
        
        st.sidebar.markdown("---")
    
    # Quick stats in sidebar
    if 'validation_stats' in st.session_state:
        st.sidebar.markdown("### 📊 Quick Stats")
        stats = st.session_state.validation_stats
        st.sidebar.metric("Critical", stats['critical'])
        st.sidebar.metric("Errors", stats['errors'])
        st.sidebar.markdown("---")
    
    # Footer
    st.sidebar.caption(f"© {datetime.now().year} Ford OEM")
    
    # Route to appropriate page
    if page == "Dashboard":
        page_dashboard()
    elif page == "Upload & Validate":
        page_upload_validate()
    elif page == "Analytics":
        page_analytics()
    elif page == "About":
        page_about()

if __name__ == "__main__":
    main()
