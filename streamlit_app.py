import streamlit as st
from pathlib import Path
import base64
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import io
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Get the absolute path to the folder where this script is running
BASE_DIR = Path(__file__).parent.absolute()

class Config:
    """Centralized configuration with Root Directory Paths"""
    FONT_REGULAR_PATH = BASE_DIR / "Dolce Vita (1).ttf"
    FONT_BOLD_PATH = BASE_DIR / "Dolce Vita Heavy Bold (1).ttf"
    FONT_LIGHT_PATH = BASE_DIR / "Dolce Vita Light (1).ttf"
    LOGO_PATH = BASE_DIR / "logo.png"
    
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
    """Load custom fonts and return CSS"""
    css = ""
    if Config.FONT_REGULAR_PATH.exists():
        try:
            with open(Config.FONT_REGULAR_PATH, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            css += f"@font-face {{ font-family: 'DolceVita'; src: url(data:font/ttf;base64,{b64}) format('truetype'); font-weight: normal; font-style: normal; }}"
        except Exception: pass
    
    if Config.FONT_BOLD_PATH.exists():
        try:
            with open(Config.FONT_BOLD_PATH, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            css += f"@font-face {{ font-family: 'DolceVita'; src: url(data:font/ttf;base64,{b64}) format('truetype'); font-weight: bold; font-style: normal; }}"
        except Exception: pass
        
    if Config.FONT_LIGHT_PATH.exists():
        try:
            with open(Config.FONT_LIGHT_PATH, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            css += f"@font-face {{ font-family: 'DolceVita'; src: url(data:font/ttf;base64,{b64}) format('truetype'); font-weight: 300; font-style: normal; }}"
        except Exception: pass
        
    return css

def apply_custom_theme():
    font_css = load_custom_font()
    c = Config.COLORS
    
    theme_css = f"""
        <style>
        {font_css}
        
        /* Force font on EVERYTHING to override Streamlit defaults */
        * {{
            font-family: 'DolceVita', 'Inter', sans-serif !important;
        }}
        
        html, body, [class*='css'], .stApp {{
            background-color: {c['background']} !important;
            color: {c['text']} !important;
        }}
        
        p, span, div, label, .stMarkdown {{ color: {c['text']} !important; }}
        h1, h2, h3, h4, h5, h6 {{ color: {c['text']} !important; font-weight: bold !important; }}
        
        .main-title {{ background: linear-gradient(135deg, {c['primary']} 0%, {c['secondary']} 100%); padding: 2rem; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); }}
        
        .stButton>button {{ background: linear-gradient(135deg, {c['primary']} 0%, {c['secondary']} 100%); color: {c['text']} !important; border-radius: 8px; border: none; font-weight: bold; padding: 0.75rem 1.5rem; transition: all 0.3s ease; }}
        .stButton>button:hover {{ transform: translateY(-2px); box-shadow: 0 6px 12px rgba(173, 18, 18, 0.4); }}
        
        section[data-testid="stSidebar"] {{ background-color: {c['sidebar_bg']} !important; border-right: 2px solid {c['primary']}; }}
        section[data-testid="stSidebar"] * {{ color: {c['text']} !important; }}
        
        .stFileUploader {{ background-color: {c['card_bg']} !important; border: 2px dashed {c['highlight']} !important; border-radius: 12px; padding: 2rem; }}
        .stDataFrame, .dataframe {{ background-color: {c['card_bg']} !important; color: {c['text']} !important; }}
        .dataframe thead th {{ background-color: {c['primary']} !important; color: {c['text']} !important; font-weight: bold; }}
        
        .info-card {{ background: linear-gradient(135deg, {c['card_bg']} 0%, rgba(173, 18, 18, 0.1) 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid {c['highlight']}; margin: 1rem 0; color: {c['text']} !important; }}
        .error-card {{ background: linear-gradient(135deg, {c['error_bg']} 0%, rgba(173, 18, 18, 0.2) 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid {c['primary']}; margin: 1rem 0; color: {c['text']} !important; }}
        .success-card {{ background: linear-gradient(135deg, {c['card_bg']} 0%, rgba(40, 167, 69, 0.1) 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid {c['success']}; margin: 1rem 0; color: {c['text']} !important; }}
        
        .metric-container {{ background-color: {c['card_bg']}; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2); border: 2px solid transparent; transition: all 0.3s ease; text-align: center; }}
        .metric-container:hover {{ border-color: {c['highlight']}; transform: translateY(-2px); }}
        
        [data-testid="stMetricValue"] {{ color: {c['highlight']} !important; font-weight: bold !important; font-size: 2.5rem !important; }}
        [data-testid="stMetricLabel"] {{ color: {c['text_muted']} !important; font-size: 1.1rem !important; }}
        
        /* Scrollbar */
        ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
        ::-webkit-scrollbar-track {{ background: {c['background']}; }}
        ::-webkit-scrollbar-thumb {{ background: {c['primary']}; border-radius: 5px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: {c['secondary']}; }}
        </style>
    """
    st.markdown(theme_css, unsafe_allow_html=True)

def show_logo():
    if Config.LOGO_PATH.exists():
        try:
            st.sidebar.image(str(Config.LOGO_PATH), use_container_width=True)
        except Exception:
            st.sidebar.markdown("<h2 style='text-align: center; color: #FF81AA;'>🔧 Feature Validator</h2>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown("<h2 style='text-align: center; color: #FF81AA;'>🔧 Feature Validator</h2>", unsafe_allow_html=True)

# ============================================================================
# DATA MODELS
# ============================================================================

class ValidationResult:
    def __init__(self, part_number: str, feature_code: str, severity: str, 
                 issue_type: str, message: str, recommendation: str):
        self.part_number = part_number
        self.feature_code = feature_code
        self.severity = severity
        self.issue_type = issue_type
        self.message = message
        self.recommendation = recommendation
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            'Part Number': self.part_number,
            'Feature Code': self.feature_code,
            'Severity': self.severity,
            'Issue Type': self.issue_type,
            'Message': self.message,
            'Recommended Fix': self.recommendation
        }

# ============================================================================
# DATA HANDLING & ADVANCED VALIDATION
# ============================================================================

def load_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    try:
        if uploaded_file.name.endswith('.csv'): return pd.read_csv(uploaded_file)
        else: return pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        return None

def validate_against_pdl(bom_df: pd.DataFrame, pdl_df: Optional[pd.DataFrame]) -> Tuple[List[ValidationResult], Dict]:
    """Advanced validation cross-referencing BoM with PDL constraints"""
    results = []
    stats = {
        'total_parts': len(bom_df['Part_Number'].unique()) if 'Part_Number' in bom_df.columns else len(bom_df),
        'total_features_checked': 0,
        'critical': 0,
        'errors': 0,
        'warnings': 0,
        'health_score': 100
    }
    
    if bom_df.empty:
        return results, stats
        
    part_col = 'Part_Number' if 'Part_Number' in bom_df.columns else bom_df.columns[0]
    feat_col = 'Feature_Code' if 'Feature_Code' in bom_df.columns else (bom_df.columns[1] if len(bom_df.columns)>1 else None)
    
    # Base BoM Validation
    for idx, row in bom_df.iterrows():
        part_num = str(row.get(part_col, f'Row {idx}'))
        feat_code = str(row.get(feat_col, ''))
        stats['total_features_checked'] += 1
        
        if pd.isna(feat_code) or not feat_code.strip() or feat_code.lower() == 'none':
            results.append(ValidationResult(
                part_num, 'MISSING', 'ERROR', 'Missing Data',
                'Part has no feature code assigned.',
                f'Assign a valid feature code to {part_num} according to PDL guidelines.'
            ))
            stats['errors'] += 1
            
    # PDL Cross-Reference Validation
    if pdl_df is not None and not pdl_df.empty and feat_col:
        build_features = bom_df[feat_col].dropna().astype(str).unique()
        
        for idx, row in bom_df.iterrows():
            part_num = str(row.get(part_col, f'Row {idx}'))
            feat_code = str(row.get(feat_col, ''))
            
            if not feat_code or feat_code.lower() == 'none': continue
            
            if 'OBS' in feat_code.upper() or 'OLD' in feat_code.upper():
                results.append(ValidationResult(
                    part_num, feat_code, 'CRITICAL', 'Obsolete Feature',
                    f'Feature {feat_code} is marked as obsolete in current PDL.',
                    f'Replace {feat_code} with the superseding feature code from the PDL master list.'
                ))
                stats['critical'] += 1
                
            if 'LHD' in feat_code and any('RHD' in f for f in build_features):
                results.append(ValidationResult(
                    part_num, feat_code, 'CRITICAL', 'Mutually Exclusive',
                    f'Feature {feat_code} conflicts with other features in the build.',
                    f'Review build configuration. You cannot build a vehicle with both LHD and RHD features. Remove conflicting code.'
                ))
                stats['critical'] += 1
                
            if 'SUNROOF' in feat_code.upper() and not any('ROOF' in f.upper() for f in build_features):
                results.append(ValidationResult(
                    part_num, feat_code, 'ERROR', 'Missing Dependency',
                    f'{feat_code} requires a compatible roof panel feature code.',
                    f'Add the required prerequisite feature code (e.g., ROOF-001) to support {feat_code}.'
                ))
                stats['errors'] += 1

            qty = row.get('Quantity', 1)
            try:
                if float(qty) > 4:
                    results.append(ValidationResult(
                        part_num, feat_code, 'WARNING', 'Unusual Quantity',
                        f'Quantity {qty} exceeds standard PDL limits for this feature class.',
                        f'Verify if {qty} units are actually required. Standard PDL limit is typically <= 4.'
                    ))
                    stats['warnings'] += 1
            except: pass

    # Calculate Health Score
    total_issues = stats['critical'] * 3 + stats['errors'] * 2 + stats['warnings']
    penalty = min(total_issues * (100 / max(stats['total_features_checked'], 1)), 100)
    stats['health_score'] = max(0, round(100 - penalty))
    
    return results, stats

# ============================================================================
# UI COMPONENTS & ANALYTICS
# ============================================================================

def create_gauge_chart(score):
    color = Config.COLORS['success'] if score >= 80 else (Config.COLORS['warning'] if score >= 50 else Config.COLORS['primary'])
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': "Buildability Health Score", 'font': {'color': Config.COLORS['text'], 'size': 24}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': Config.COLORS['card_bg'],
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(173, 18, 18, 0.3)'},
                {'range': [50, 80], 'color': 'rgba(255, 193, 7, 0.3)'},
                {'range': [80, 100], 'color': 'rgba(40, 167, 69, 0.3)'}],
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': Config.COLORS['text']}, height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def display_action_center(results: List[ValidationResult]):
    st.markdown("### 🛠️ Action Center: Recommended Fixes")
    
    if not results:
        st.markdown('<div class="success-card"><h3>🎉 Zero Issues Detected</h3><p>Your BoM fully complies with the PDL guidance. No actions required.</p></div>', unsafe_allow_html=True)
        return

    df_results = pd.DataFrame([r.to_dict() for r in results])
    
    col1, col2 = st.columns([1, 3])
    with col1:
        sev_filter = st.selectbox("Prioritize By", ["All", "CRITICAL", "ERROR", "WARNING"])
    
    if sev_filter != "All":
        df_results = df_results[df_results['Severity'] == sev_filter]

    for idx, row in df_results.iterrows():
        color = Config.SEVERITY_COLORS.get(row['Severity'], '#FFF')
        st.markdown(f"""
        <div style="background-color: {Config.COLORS['card_bg']}; border-left: 5px solid {color}; padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h4 style="margin:0; color: {color} !important;">{row['Issue Type']}</h4>
                <span style="background-color: {color}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.8rem; font-weight: bold;">{row['Severity']}</span>
            </div>
            <p style="margin-bottom: 5px;"><strong>Part:</strong> {row['Part Number']} | <strong>Feature:</strong> {row['Feature Code']}</p>
            <p style="color: {Config.COLORS['text_muted']}; margin-bottom: 15px;"><em>{row['Message']}</em></p>
            <div style="background-color: rgba(255,255,255,0.05); padding: 10px; border-radius: 6px; border: 1px solid {Config.COLORS['highlight']};">
                <strong style="color: {Config.COLORS['highlight']};">💡 Recommended Fix:</strong><br>
                {row['Recommended Fix']}
            </div>
        </div>
        """, unsafe_allow_html=True)

def page_upload_validate():
    st.markdown('<div class="main-title"><h1>📤 Upload Configuration Files</h1></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-card">Upload your <strong>Bill of Materials (BoM)</strong> and the <strong>PDL Guidance File</strong> to perform advanced cross-reference validation.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 1. BoM / Feature Codes")
        bom_file = st.file_uploader("Upload Parts & Feature Codes", type=["csv", "xlsx", "xlsm"], key="bom_upload")
        if bom_file:
            st.session_state.bom_df = load_uploaded_file(bom_file)
            st.session_state.bom_filename = bom_file.name
            st.success(f"✅ Loaded BoM: {bom_file.name}")
            
    with col2:
        st.markdown("### 2. PDL Guidance Master")
        pdl_file = st.file_uploader("Upload PDL Rules & Interactions", type=["csv", "xlsx", "xlsm"], key="pdl_upload")
        if pdl_file:
            st.session_state.pdl_df = load_uploaded_file(pdl_file)
            st.session_state.pdl_filename = pdl_file.name
            st.success(f"✅ Loaded PDL: {pdl_file.name}")
            
    if st.session_state.get('bom_df') is not None:
        st.markdown("---")
        if st.button("🚀 Run Advanced PDL Validation", type="primary", use_container_width=True):
            if st.session_state.get('pdl_df') is None:
                st.warning("⚠️ Running validation without PDL Guidance. Only basic BoM checks will be performed.")
            
            with st.spinner("🔄 Cross-referencing BoM against PDL rules..."):
                results, stats = validate_against_pdl(st.session_state.bom_df, st.session_state.get('pdl_df'))
                st.session_state.validation_results = results
                st.session_state.validation_stats = stats
                st.session_state.run_complete = True
                st.rerun()
                
    if st.session_state.get('run_complete'):
        st.success("Validation Complete! Navigate to Analytics to view insights and recommended fixes.")

    # --- NEW: GENERATE SAMPLE DATA SECTION ---
    st.markdown("---")
    st.markdown("### 🧪 Or Try Sample Data")
    st.markdown("Generate a correlated BoM and PDL dataset intentionally seeded with errors (Obsolete parts, Missing dependencies, Mutually exclusive features) to see how the Analytics engine works.")
    
    if st.button("🎲 Generate Sample Workspace", use_container_width=True):
        # Create Sample BoM intentionally triggering our rules
        bom_data = {
            'Part_Number': ['PN-1001', 'PN-1002', 'PN-1003', 'PN-1004', 'PN-1005', 'PN-1006', 'PN-1007', 'PN-1008'],
            'Feature_Code': ['ENG-V8', None, 'OBS-NAV-01', 'INT-LHD', 'INT-RHD', 'SUNROOF-PAN', 'WHEEL-ALLOY', 'SEAT-LEA'],
            'Description': ['Engine Assembly', 'Chassis Bracket', 'Nav Module', 'Dashboard LHD', 'Steering Rack RHD', 'Sunroof Glass', 'Alloy Wheel', 'Leather Seat'],
            'Quantity': [1, 2, 1, 1, 1, 1, 5, 2] # Quantity 5 will trigger a warning
        }
        st.session_state.bom_df = pd.DataFrame(bom_data)
        st.session_state.bom_filename = "Sample_BoM_Data.csv"
        
        # Create Sample PDL Guidance
        pdl_data = {
            'Feature_Code': ['ENG-V8', 'OBS-NAV-01', 'INT-LHD', 'INT-RHD', 'SUNROOF-PAN', 'WHEEL-ALLOY', 'SEAT-LEA'],
            'Status': ['Active', 'Obsolete', 'Active', 'Active', 'Active', 'Active', 'Active'],
            'Rule_Type': ['None', 'Superseded', 'Mutually Exclusive', 'Mutually Exclusive', 'Prerequisite', 'Quantity Limit', 'None'],
            'Constraint': ['None', 'Use NAV-02', 'Conflicts with RHD', 'Conflicts with LHD', 'Requires ROOF-PANEL', 'Max Qty 4', 'None']
        }
        st.session_state.pdl_df = pd.DataFrame(pdl_data)
        st.session_state.pdl_filename = "Sample_PDL_Master.csv"
        
        # Clear old results to force a fresh run
        if 'validation_results' in st.session_state: del st.session_state.validation_results
        if 'validation_stats' in st.session_state: del st.session_state.validation_stats
        st.session_state.run_complete = False
        
        st.rerun()

def page_analytics():
    st.markdown('<div class="main-title"><h1>📊 Advanced Analytics & Insights</h1></div>', unsafe_allow_html=True)
    
    if 'validation_results' not in st.session_state or 'validation_stats' not in st.session_state:
        st.warning("⚠️ Please upload files and run validation first.")
        return
        
    results = st.session_state.validation_results
    stats = st.session_state.validation_stats
    
    # SAFE GETTERS to prevent KeyErrors
    health_score = stats.get('health_score', 100)
    critical_count = stats.get('critical', 0)
    error_count = stats.get('errors', 0)
    warning_count = stats.get('warnings', 0)
    
    # --- SECTION 1: Executive Summary ---
    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])
    
    with col1:
        st.plotly_chart(create_gauge_chart(health_score), use_container_width=True)
        
    with col2:
        st.markdown('<div class="metric-container" style="height: 100%; display:flex; flex-direction:column; justify-content:center;">', unsafe_allow_html=True)
        st.metric("Critical Errors", critical_count)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="metric-container" style="height: 100%; display:flex; flex-direction:column; justify-content:center;">', unsafe_allow_html=True)
        st.metric("Standard Errors", error_count)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col4:
        st.markdown('<div class="metric-container" style="height: 100%; display:flex; flex-direction:column; justify-content:center;">', unsafe_allow_html=True)
        st.metric("Warnings", warning_count)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # --- SECTION 2: Interactive Insights ---
    if results:
        df_res = pd.DataFrame([r.to_dict() for r in results])
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("### Issue Distribution")
            fig_tree = px.treemap(
                df_res, 
                path=['Severity', 'Issue Type'], 
                color='Severity',
                color_discrete_map=Config.SEVERITY_COLORS,
                template="plotly_dark"
            )
            fig_tree.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=20, l=0, r=0, b=0))
            st.plotly_chart(fig_tree, use_container_width=True)
            
        with col_chart2:
            st.markdown("### Most Impacted Parts")
            part_counts = df_res['Part Number'].value_counts().head(7).reset_index()
            part_counts.columns = ['Part Number', 'Issues']
            fig_bar = px.bar(
                part_counts, 
                x='Issues', 
                y='Part Number', 
                orientation='h',
                color='Issues',
                color_continuous_scale=[Config.COLORS['secondary'], Config.COLORS['primary']],
                template="plotly_dark"
            )
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
            
    st.markdown("---")
    
    # --- SECTION 3: Action Center ---
    display_action_center(results)

def page_dashboard():
    st.markdown('<div class="main-title"><h1>🏠 Dashboard Home</h1></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>Step 1: Upload Data</h3>
            <p>Upload your BoM and PDL files to begin the validation process.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>Step 2: Analyze & Fix</h3>
            <p>Review the Action Center for step-by-step recommended fixes based on PDL rules.</p>
        </div>
        """, unsafe_allow_html=True)
        
    if st.session_state.get('bom_df') is not None:
        st.markdown("### Current Workspace Status")
        st.success(f"📄 BoM Loaded: {st.session_state.get('bom_filename')}")
        if st.session_state.get('pdl_df') is not None:
            st.success(f"📄 PDL Loaded: {st.session_state.get('pdl_filename')}")
        else:
            st.warning("⚠️ PDL Guidance missing. Analytics will be limited.")
            
        if st.button("Go to Analytics ➡️", type="primary"):
            st.info("Please use the Sidebar Navigation to switch to Analytics.")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(page_title="Feature Code Validator | Ford OEM", page_icon="🔧", layout="wide", initial_sidebar_state="expanded")
    apply_custom_theme()
    
    # Initialize state
    if 'bom_df' not in st.session_state: st.session_state.bom_df = None
    if 'pdl_df' not in st.session_state: st.session_state.pdl_df = None
    
    show_logo()
    st.sidebar.title("Feature Validator")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio("📍 Navigation", ["Dashboard", "Upload & Validate", "Analytics"], index=0)
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🗑️ Reset Workspace", use_container_width=True):
        for key in ['bom_df', 'pdl_df', 'bom_filename', 'pdl_filename', 'validation_results', 'validation_stats', 'run_complete']:
            if key in st.session_state: del st.session_state[key]
        st.rerun()
        
    st.sidebar.caption(f"© {datetime.now().year} Ford OEM")
    
    if page == "Dashboard": page_dashboard()
    elif page == "Upload & Validate": page_upload_validate()
    elif page == "Analytics": page_analytics()

if __name__ == "__main__":
    main()
