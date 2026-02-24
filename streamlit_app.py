import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import urllib.parse

# ============================================================================
# CONFIGURATION & CONSTANTS (QR_ BRAND)
# ============================================================================

BASE_DIR = Path(__file__).parent.absolute()
ASSETS_DIR = BASE_DIR / "Assets"
LOGO_PATH = ASSETS_DIR / "images" / "logo.png"

class Config:
    """QR_ Brand Color Palette - Dark Theme Optimized"""
    COLORS = {
        'primary_red': '#D7171F',
        'dark_red': '#A11117',
        'main_bg': '#232324',       # QR Dark Grey
        'sidebar_bg': '#151517',    # Deeper Dark Grey for contrast
        'card_bg': '#363738',       # QR Medium Dark Grey
        'card_hover': '#454647',    # Slightly lighter for hover states
        'text_main': '#FFFFFF',     # White
        'text_muted': '#A6A8AA',    # Softer Light Grey for better reading
        'blue_1': '#0070BB',
        'green': '#4D8B31',
        'orange': '#EE4B0F',
        'yellow': '#FFA602'
    }
    
    SEVERITY_COLORS = {
        'CRITICAL': '#D7171F', # QR Red
        'ERROR': '#EE4B0F',    # QR Orange
        'WARNING': '#FFA602',  # QR Yellow
        'INFO': '#0070BB'      # QR Blue
    }

# ============================================================================
# STYLING & THEME INJECTION
# ============================================================================

def apply_custom_theme():
    """Injects premium QR_ Brand CSS with strict typographic scaling"""
    c = Config.COLORS
    
    theme_css = f"""
        <style>
        /* Import Inter as a web-safe fallback for Segoe UI */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        /* Global Typography & Background */
        html, body, .stApp {{
            background-color: {c['main_bg']} !important;
            color: {c['text_main']} !important;
            font-family: 'Inter', 'Segoe UI', sans-serif !important;
            -webkit-font-smoothing: antialiased;
        }}

        /* Override Streamlit's default Pink focus rings with QR Red */
        :root {{
            --primary-color: {c['primary_red']};
            --background-color: {c['main_bg']};
            --secondary-background-color: {c['card_bg']};
            --text-color: {c['text_main']};
        }}

        /* =========================================
           STRICT TYPOGRAPHIC SCALE
           ========================================= */
        h1, h2, h3, h4, h5, h6, p, li, label, td, th {{ 
            font-family: 'Inter', 'Segoe UI', sans-serif !important; 
            color: {c['text_main']};
        }}
        
        h1 {{ font-size: 2.25rem !important; font-weight: 700 !important; line-height: 1.2 !important; margin-bottom: 1rem !important; }}
        h2 {{ font-size: 1.75rem !important; font-weight: 600 !important; line-height: 1.3 !important; margin-bottom: 0.875rem !important; }}
        h3 {{ font-size: 1.25rem !important; font-weight: 600 !important; line-height: 1.4 !important; margin-bottom: 0.75rem !important; }}
        h4 {{ font-size: 1.1rem !important; font-weight: 600 !important; line-height: 1.4 !important; margin-bottom: 0.5rem !important; }}
        p, li {{ font-size: 1rem !important; line-height: 1.6 !important; font-weight: 400 !important; }}
        small, .caption {{ font-size: 0.875rem !important; color: {c['text_muted']} !important; }}

        /* Modern Gradient Headers */
        .main-title {{ 
            background: linear-gradient(135deg, {c['primary_red']} 0%, {c['dark_red']} 100%); 
            padding: 2rem 2.5rem; 
            border-radius: 12px; 
            margin-bottom: 2.5rem; 
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4); 
            border: 1px solid rgba(255,255,255,0.05);
        }}
        .main-title h1 {{ color: {c['text_main']} !important; margin: 0 !important; letter-spacing: -0.5px; }}

        /* Primary Buttons */
        .stButton>button {{ 
            background: linear-gradient(135deg, {c['primary_red']} 0%, {c['dark_red']} 100%) !important; 
            color: {c['text_main']} !important; 
            border-radius: 8px; 
            border: 1px solid rgba(255,255,255,0.1); 
            font-size: 1rem !important;
            font-weight: 600 !important; 
            padding: 0.75rem 1.5rem; 
            transition: all 0.3s ease; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .stButton>button:hover {{ 
            transform: translateY(-2px); 
            box-shadow: 0 6px 12px rgba(215, 23, 31, 0.4); 
            border-color: rgba(255,255,255,0.3);
        }}
        .stButton>button:focus {{
            outline: none !important;
            box-shadow: 0 0 0 3px rgba(215, 23, 31, 0.5) !important;
        }}

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{ 
            background-color: {c['sidebar_bg']} !important; 
            border-right: 2px solid {c['primary_red']}; 
        }}
        
        /* Floating Content Cards */
        .info-card {{ 
            background-color: {c['card_bg']}; 
            padding: 1.75rem; 
            border-radius: 12px; 
            border-left: 5px solid {c['primary_red']}; 
            margin: 1rem 0; 
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2); 
            transition: transform 0.3s ease, background-color 0.3s ease;
            border: 1px solid rgba(255,255,255,0.05);
        }}
        .info-card:hover {{
            transform: translateY(-3px);
            background-color: {c['card_hover']};
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.3);
        }}
        .info-card h3 {{ color: {c['text_main']} !important; margin-top: 0 !important; }}
        .info-card p {{ color: {c['text_muted']} !important; margin-bottom: 0 !important; }}

        /* File Uploader Dark Mode Fixes */
        .stFileUploader {{ 
            background-color: {c['card_bg']} !important; 
            border: 2px dashed {c['text_muted']} !important; 
            border-radius: 12px; 
            padding: 2.5rem 2rem; 
            transition: border-color 0.3s ease;
        }}
        .stFileUploader:hover {{ border-color: {c['primary_red']} !important; }}
        [data-testid="stFileUploadDropzone"] * {{ color: {c['text_main']} !important; }}

        /* Metrics */
        .metric-container {{ 
            background-color: {c['card_bg']}; 
            padding: 1.5rem; 
            border-radius: 12px; 
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2); 
            border: 1px solid rgba(255,255,255,0.05); 
            transition: all 0.3s ease; 
            text-align: center; 
        }}
        .metric-container:hover {{ 
            border-color: {c['primary_red']}; 
            transform: translateY(-3px); 
            background-color: {c['card_hover']};
        }}
        [data-testid="stMetricValue"] {{ color: {c['text_main']} !important; font-weight: 700 !important; font-size: 2.5rem !important; line-height: 1 !important; }}
        [data-testid="stMetricLabel"] {{ color: {c['text_muted']} !important; font-size: 1rem !important; font-weight: 600 !important; letter-spacing: 0.5px; text-transform: uppercase; }}

        /* DataFrames */
        .stDataFrame, .dataframe {{ background-color: {c['card_bg']} !important; border-radius: 8px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1); }}
        .dataframe thead th {{ background-color: {c['sidebar_bg']} !important; color: {c['text_main']} !important; font-weight: 600; font-size: 0.9rem !important; border-bottom: 2px solid {c['primary_red']}; text-transform: uppercase; letter-spacing: 0.5px; }}
        .dataframe tbody td {{ color: {c['text_main']} !important; font-size: 0.95rem !important; }}
        
        /* Hyperlinks */
        a {{ color: {c['primary_red']} !important; text-decoration: none !important; font-weight: 600; border-bottom: 1px solid {c['primary_red']}; transition: opacity 0.2s; }}
        a:hover {{ opacity: 0.8; }}
        
        /* Expander styling */
        .streamlit-expanderHeader {{
            background-color: {c['card_bg']} !important;
            border-radius: 8px !important;
            color: {c['text_main']} !important;
            font-weight: 600 !important;
            font-size: 1.05rem !important;
        }}
        </style>
    """
    st.markdown(theme_css, unsafe_allow_html=True)

def show_logo():
    """Displays the QR_ Logo in the sidebar"""
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), use_container_width=True)
    else:
        st.sidebar.markdown(f"<h2 style='text-align: center; color: {Config.COLORS['primary_red']}; font-weight: 700; letter-spacing: 1px;'>QUICK RELEASE_</h2>", unsafe_allow_html=True)

# ============================================================================
# DATA MODELS & VALIDATION LOGIC
# ============================================================================

class ValidationResult:
    def __init__(self, part_number: str, feature_code: str, severity: str, 
                 issue_type: str, message: str, recommendation: str, engineer_id: str = "Unassigned"):
        self.part_number = part_number
        self.feature_code = feature_code
        self.severity = severity
        self.issue_type = issue_type
        self.message = message
        self.recommendation = recommendation
        self.engineer_id = engineer_id
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            'Engineer ID': self.engineer_id,
            'Part Number': self.part_number,
            'Feature Code': self.feature_code,
            'Severity': self.severity,
            'Issue Type': self.issue_type,
            'Message': self.message,
            'Recommended Fix': self.recommendation
        }

def load_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    try:
        if uploaded_file.name.endswith('.csv'): return pd.read_csv(uploaded_file)
        else: return pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        return None

def validate_against_pdl(bom_df: pd.DataFrame, pdl_df: Optional[pd.DataFrame]) -> Tuple[List[ValidationResult], Dict]:
    results = []
    stats = {
        'total_parts': len(bom_df['Part_Number'].unique()) if 'Part_Number' in bom_df.columns else len(bom_df),
        'total_features_checked': 0,
        'critical': 0,
        'errors': 0,
        'warnings': 0,
        'risk_score': 0 
    }
    
    if bom_df.empty: return results, stats
        
    part_col = 'Part_Number' if 'Part_Number' in bom_df.columns else bom_df.columns[0]
    feat_col = 'Feature_Code' if 'Feature_Code' in bom_df.columns else (bom_df.columns[1] if len(bom_df.columns)>1 else None)
    eng_col = next((c for c in bom_df.columns if c.upper() in ['ENGINEER_ID', 'D&R_ID', 'ENGINEER', 'OWNER']), None)
    
    for idx, row in bom_df.iterrows():
        part_num = str(row.get(part_col, f'Row {idx}'))
        feat_code = str(row.get(feat_col, ''))
        eng_id = str(row.get(eng_col, 'Unassigned')) if eng_col else 'Unassigned'
        if pd.isna(row.get(eng_col)) or eng_id.lower() == 'nan': eng_id = 'Unassigned'
        
        stats['total_features_checked'] += 1
        
        if pd.isna(feat_code) or not feat_code.strip() or feat_code.lower() == 'none':
            results.append(ValidationResult(
                part_num, 'MISSING', 'ERROR', 'Missing Data',
                'Part has no feature code assigned.',
                f'Assign a valid feature code to {part_num} according to PDL guidelines.', eng_id
            ))
            stats['errors'] += 1
            
    if pdl_df is not None and not pdl_df.empty and feat_col:
        build_features = bom_df[feat_col].dropna().astype(str).unique()
        
        for idx, row in bom_df.iterrows():
            part_num = str(row.get(part_col, f'Row {idx}'))
            feat_code = str(row.get(feat_col, ''))
            eng_id = str(row.get(eng_col, 'Unassigned')) if eng_col else 'Unassigned'
            if pd.isna(row.get(eng_col)) or eng_id.lower() == 'nan': eng_id = 'Unassigned'
            
            if not feat_code or feat_code.lower() == 'none': continue
            
            if 'OBS' in feat_code.upper() or 'OLD' in feat_code.upper():
                results.append(ValidationResult(
                    part_num, feat_code, 'CRITICAL', 'Obsolete Feature',
                    f'Feature {feat_code} is marked as obsolete in current PDL.',
                    f'Replace {feat_code} with the superseding feature code from the PDL master list.', eng_id
                ))
                stats['critical'] += 1
                
            if 'LHD' in feat_code and any('RHD' in f for f in build_features):
                results.append(ValidationResult(
                    part_num, feat_code, 'CRITICAL', 'Mutually Exclusive',
                    f'Feature {feat_code} conflicts with other features in the build.',
                    f'Review build configuration. You cannot build a vehicle with both LHD and RHD features.', eng_id
                ))
                stats['critical'] += 1
                
            if 'SUNROOF' in feat_code.upper() and not any('ROOF' in f.upper() for f in build_features):
                results.append(ValidationResult(
                    part_num, feat_code, 'ERROR', 'Missing Dependency',
                    f'{feat_code} requires a compatible roof panel feature code.',
                    f'Add the required prerequisite feature code (e.g., ROOF-001) to support {feat_code}.', eng_id
                ))
                stats['errors'] += 1

            qty = row.get('Quantity', 1)
            try:
                if float(qty) > 4:
                    results.append(ValidationResult(
                        part_num, feat_code, 'WARNING', 'Unusual Quantity',
                        f'Quantity {qty} exceeds standard PDL limits for this feature class.',
                        f'Verify if {qty} units are actually required. Standard PDL limit is typically <= 4.', eng_id
                    ))
                    stats['warnings'] += 1
            except: pass

    total_issues = stats['critical'] * 3 + stats['errors'] * 2 + stats['warnings']
    risk = min(total_issues * (100 / max(stats['total_features_checked'], 1)), 100)
    stats['risk_score'] = round(risk)
    
    return results, stats

def create_gauge_chart(score):
    if score < 20: color = Config.COLORS['green']
    elif score < 50: color = Config.COLORS['yellow']
    else: color = Config.COLORS['primary_red']
        
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': "Buildability Risk Score", 'font': {'color': Config.COLORS['text_main'], 'size': 20, 'family': 'Inter'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': Config.COLORS['text_muted']},
            'bar': {'color': color},
            'bgcolor': Config.COLORS['sidebar_bg'],
            'borderwidth': 2,
            'bordercolor': Config.COLORS['card_hover'],
            'steps': [
                {'range': [0, 20], 'color': 'rgba(77, 139, 49, 0.15)'}, 
                {'range': [20, 50], 'color': 'rgba(255, 166, 2, 0.15)'}, 
                {'range': [50, 100], 'color': 'rgba(215, 23, 31, 0.15)'}], 
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': Config.COLORS['text_main'], 'family': 'Inter'}, height=320, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# ============================================================================
# PAGE ROUTING
# ============================================================================

def page_upload_validate():
    st.markdown('<div class="main-title"><h1>📤 Upload Configuration Files</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-card"><p>Upload your <strong>Bill of Materials (BoM)</strong> and the <strong>PDL Guidance File</strong> to perform advanced cross-reference validation.</p></div>', unsafe_allow_html=True)
    
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
        if st.button("🔄 Run Advanced PDL Validation", type="primary", use_container_width=True):
            if st.session_state.get('pdl_df') is None:
                st.warning("⚠️ Running validation without PDL Guidance. Only basic BoM checks will be performed.")
            
            with st.spinner("🔄 Cross-referencing BoM against PDL rules..."):
                results, stats = validate_against_pdl(st.session_state.bom_df, st.session_state.get('pdl_df'))
                st.session_state.validation_results = results
                st.session_state.validation_stats = stats
                st.session_state.run_complete = True
                st.rerun()
                
    if st.session_state.get('run_complete'):
        st.success("✅ Validation Complete! Navigate to Analytics to view insights and recommended fixes.")

    st.markdown("---")
    st.markdown("### 📝 Or Try Sample Data")
    if st.button("🎲 Generate Large Sample Workspace", use_container_width=True):
        np.random.seed(42)
        num_parts = 250
        part_numbers = [f"PN-F{10000 + i}" for i in range(num_parts)]
        valid_features = ['ENG-V8', 'TRANS-AUTO', 'SEAT-LEA', 'WHEEL-18', 'NAV-02', 'AUDIO-PREM', 'LIGHT-LED', 'TRIM-CHROME']
        sample_engineers = ['john.smith@quickrelease.co.uk', 'mary.jane@quickrelease.co.uk', 'david.lee@quickrelease.co.uk']
        
        feature_codes, quantities, descriptions, assigned_engineers = [], [], [], []
        
        for i in range(num_parts):
            rand_val = np.random.random()
            assigned_engineers.append(np.random.choice(sample_engineers))
            if rand_val < 0.75:
                feature_codes.append(np.random.choice(valid_features))
                quantities.append(np.random.randint(1, 4))
                descriptions.append("Standard Production Component")
            elif rand_val < 0.85:
                feature_codes.append(None)
                quantities.append(1)
                descriptions.append("Unassigned Bracket")
            elif rand_val < 0.93:
                feature_codes.append('OBS-NAV-01')
                quantities.append(1)
                descriptions.append("Legacy Navigation Module")
            elif rand_val < 0.97:
                feature_codes.append('WHEEL-18')
                quantities.append(np.random.randint(5, 12)) 
                descriptions.append("Alloy Wheel")
            elif rand_val < 0.99:
                feature_codes.append(np.random.choice(['INT-LHD', 'INT-RHD']))
                quantities.append(1)
                descriptions.append("Directional Interior Trim")
            else:
                feature_codes.append('SUNROOF-PAN')
                quantities.append(1)
                descriptions.append("Panoramic Sunroof Glass")
                
        st.session_state.bom_df = pd.DataFrame({
            'Part_Number': part_numbers, 'Feature_Code': feature_codes,
            'Description': descriptions, 'Quantity': quantities, 'Engineer_ID': assigned_engineers
        })
        st.session_state.bom_filename = "Large_Sample_BoM_Data.csv"
        
        st.session_state.pdl_df = pd.DataFrame({
            'Feature_Code': ['ENG-V8', 'OBS-NAV-01', 'INT-LHD', 'INT-RHD', 'SUNROOF-PAN', 'WHEEL-18'],
            'Status': ['Active', 'Obsolete', 'Active', 'Active', 'Active', 'Active'],
            'Rule_Type': ['None', 'Superseded', 'Mutually Exclusive', 'Mutually Exclusive', 'Prerequisite', 'Quantity Limit'],
            'Constraint': ['None', 'Use NAV-02', 'Conflicts with RHD', 'Conflicts with LHD', 'Requires ROOF-PANEL', 'Max Qty 4']
        })
        st.session_state.pdl_filename = "Master_PDL_Rules.csv"
        
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
    
    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])
    with col1: st.plotly_chart(create_gauge_chart(stats.get('risk_score', 0)), use_container_width=True)
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Critical Errors", stats.get('critical', 0))
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Standard Errors", stats.get('errors', 0))
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Warnings", stats.get('warnings', 0))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    if results:
        df_res = pd.DataFrame([r.to_dict() for r in results])
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.markdown("### Issue Distribution")
            fig_tree = px.sunburst(df_res, path=['Severity', 'Issue Type'], color='Severity',
                                   color_discrete_map=Config.SEVERITY_COLORS)
            fig_tree.update_traces(textinfo="label+value") 
            fig_tree.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10, l=0, r=0, b=10), font=dict(family='Inter', color=Config.COLORS['text_main']))
            st.plotly_chart(fig_tree, use_container_width=True)
            
        with col_chart2:
            st.markdown("### Most Problematic Features")
            feature_counts = df_res['Feature Code'].replace('MISSING', 'Unassigned').value_counts().head(7).reset_index()
            feature_counts.columns = ['Feature Code', 'Issue Count']
            fig_bar = px.bar(feature_counts, x='Issue Count', y='Feature Code', orientation='h', text='Issue Count')
            fig_bar.update_traces(marker_color=Config.COLORS['primary_red'], textposition='outside')
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis={'categoryorder':'total ascending'}, margin=dict(t=10, l=0, r=20, b=10), font=dict(family='Inter', color=Config.COLORS['text_main']))
            st.plotly_chart(fig_bar, use_container_width=True)
            
    st.markdown("---")
    st.markdown("### 🛠️ Action Center: Recommended Fixes")
    if not results:
        st.markdown('<div class="info-card"><h3 style="color:#4D8B31;">✨ Zero Issues Detected</h3><p>Your BoM fully complies with the PDL guidance. No actions required.</p></div>', unsafe_allow_html=True)
        return

    df_results = pd.DataFrame([r.to_dict() for r in results])
    col1, col2 = st.columns([1, 3])
    with col1: sev_filter = st.selectbox("Prioritize By", ["All", "CRITICAL", "ERROR", "WARNING"])
    if sev_filter != "All": df_results = df_results[df_results['Severity'] == sev_filter]

    for idx, row in df_results.iterrows():
        color = Config.SEVERITY_COLORS.get(row['Severity'], Config.COLORS['text_muted'])
        card_html = f"""<div class="info-card" style="border-left: 6px solid {color}; padding: 1.5rem;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
<h4 style="margin:0; color: {color} !important;">{row['Issue Type']}</h4>
<span style="background-color: {color}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.8rem; font-weight: bold;">{row['Severity']}</span>
</div>
<p style="margin-bottom: 5px; font-size: 1rem;"><strong>Part:</strong> {row['Part Number']} | <strong>Feature:</strong> {row['Feature Code']} | <strong>Engineer:</strong> {row['Engineer ID']}</p>
<p style="color: {Config.COLORS['text_muted']}; margin-bottom: 15px;"><em>{row['Message']}</em></p>
<div style="background-color: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
<strong style="color: {Config.COLORS['primary_red']};">💡 Recommended Fix:</strong><br>{row['Recommended Fix']}
</div>
</div>"""
        st.markdown(card_html, unsafe_allow_html=True)

def page_auto_emails():
    st.markdown('<div class="main-title"><h1>📧 D&R Auto-Communications</h1></div>', unsafe_allow_html=True)
    
    if 'validation_results' not in st.session_state or not st.session_state.validation_results:
        st.warning("⚠️ Please run validation first to generate engineer communications.")
        return
        
    info_html = """<div class="info-card">
<p>Review the grouped BoM issues below. Click the draft button to open your default email client (e.g., Outlook).</p>
<p><strong>⚠️ Note regarding Supervisors:</strong> Because the system cannot directly query your local Outlook Active Directory structure, the 'CC' line has been pre-populated with a placeholder. <strong>Please replace the placeholder with the Engineer's Supervisor before sending.</strong></p>
</div>"""
    st.markdown(info_html, unsafe_allow_html=True)
    
    results = st.session_state.validation_results
    
    from collections import defaultdict
    issues_by_engineer = defaultdict(list)
    for res in results:
        issues_by_engineer[res.engineer_id].append(res)
        
    if 'Unassigned' in issues_by_engineer:
        unassigned_count = len(issues_by_engineer['Unassigned'])
        st.error(f"⚠️ {unassigned_count} issue(s) do not have a D&R Engineer assigned in the BoM.")
            
    st.markdown("### Engineer Action Queues")
    
    for eng_id, issues in issues_by_engineer.items():
        if eng_id == 'Unassigned': continue
        
        severity_order = {'CRITICAL': 1, 'ERROR': 2, 'WARNING': 3, 'INFO': 4}
        issues.sort(key=lambda x: severity_order.get(x.severity, 5))
        
        with st.expander(f"👤 {eng_id} — {len(issues)} Required Action(s)"):
            subject = f"ACTION REQUIRED: BoM Validation Issues Detected - Action needed for {len(issues)} Part(s)"
            
            # Use explicit carriage returns for Outlook
            body = f"Hello,\r\n\r\nThe automated Feature Validator has detected {len(issues)} issue(s) with the parts assigned to you in the latest Bill of Materials (BoM).\r\n\r\n"
            body += "Please review and correct the following in the system:\r\n\r\n"
            body += "="*60 + "\r\n\r\n"
            
            for i, issue in enumerate(issues, 1):
                body += f"[{i}] Part Number: {issue.part_number} | Feature Code: {issue.feature_code}\r\n"
                body += f"    Issue: {issue.issue_type} ({issue.severity})\r\n"
                body += f"    Details: {issue.message}\r\n"
                body += f"    Recommended Fix: {issue.recommendation}\r\n\r\n"
                
            body += "="*60 + "\r\n\r\n"
            body += "Thank you,\r\nBoM Validation Team\r\n"
            
            mailto_link = f"mailto:{eng_id}?cc=INSERT_SUPERVISOR_HERE@quickrelease.co.uk&subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"
            
            df_display = pd.DataFrame([r.to_dict() for r in issues])[['Severity', 'Issue Type', 'Part Number', 'Feature Code']]
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # Removed target="_blank" to prevent opening an empty tab
            btn_html = f"""<a href="{mailto_link}" style="text-decoration: none;">
<button style="background: linear-gradient(135deg, {Config.COLORS['primary_red']} 0%, {Config.COLORS['dark_red']} 100%); color: white; border: none; padding: 10px 20px; border-radius: 8px; font-weight: 600; cursor: pointer; margin-top: 10px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); font-family: 'Inter', sans-serif;">📨 Draft Email in Outlook</button>
</a>"""
            st.markdown(btn_html, unsafe_allow_html=True)

def page_nightletter():
    st.markdown('<div class="main-title"><h1>🌙 Executive Nightletter</h1></div>', unsafe_allow_html=True)
    
    if 'validation_results' not in st.session_state or 'validation_stats' not in st.session_state:
        st.warning("⚠️ Please run a validation first to generate the Nightletter.")
        return
        
    stats = st.session_state.validation_stats
    results = st.session_state.validation_results
    
    # --- Data Processing for Nightletter ---
    risk_score = stats.get('risk_score', 0)
    total_parts = stats.get('total_features_checked', 0)
    total_issues = len(results)
    
    df_res = pd.DataFrame([r.to_dict() for r in results])
    
    # Assess Status
    if risk_score < 20: 
        status_text = "🟢 ON TRACK - Minor corrections required."
        status_color = Config.COLORS['green']
    elif risk_score < 50: 
        status_text = "🟡 AT RISK - Moderate issues detected. Action required."
        status_color = Config.COLORS['yellow']
    else: 
        status_text = "🔴 CRITICAL - High volume of buildability errors. Immediate attention required."
        status_color = Config.COLORS['primary_red']

    # Top Features
    top_features_html = ""
    top_features_plain = ""
    if not df_res.empty:
        feature_counts = df_res['Feature Code'].replace('MISSING', 'Unassigned').value_counts().head(3)
        for feat, count in feature_counts.items():
            top_features_html += f"<li style='margin-bottom: 5px;'><strong style='color: {Config.COLORS['text_main']};'>{feat}</strong>: <span style='color: {Config.COLORS['text_muted']};'>{count} impacted parts</span></li>"
            top_features_plain += f"  • {feat}: {count} impacted parts\r\n"
    else:
        top_features_html = "<li>No issues detected today!</li>"
        top_features_plain = "  • No issues detected today!\r\n"

    # Top Engineers Workload
    top_engineers_html = ""
    top_engineers_plain = ""
    if not df_res.empty and 'Engineer ID' in df_res.columns:
        eng_counts = df_res[df_res['Engineer ID'] != 'Unassigned']['Engineer ID'].value_counts().head(3)
        if not eng_counts.empty:
            for eng, count in eng_counts.items():
                top_engineers_html += f"<li style='margin-bottom: 5px;'><strong style='color: {Config.COLORS['text_main']};'>{eng}</strong>: <span style='color: {Config.COLORS['text_muted']};'>{count} actions pending</span></li>"
                top_engineers_plain += f"  • {eng}: {count} actions pending\r\n"
        else:
            top_engineers_html = "<li>All issues are currently unassigned.</li>"
            top_engineers_plain = "  • All issues are currently unassigned.\r\n"
    else:
        top_engineers_html = "<li>No actions pending.</li>"
        top_engineers_plain = "  • No actions pending.\r\n"

    bom_name = st.session_state.get('bom_filename', 'Unknown BoM')
    date_str = datetime.now().strftime("%B %d, %Y")
    
    # --- 1. The HTML Visual Preview (For the App UI) ---
    # Completely flattened string to prevent Streamlit from wrapping it in a <pre><code> block
    html_preview = f"""<div style="background-color: {Config.COLORS['sidebar_bg']}; border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 2.5rem; margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
<h2 style="color: {Config.COLORS['text_main']}; border-bottom: 2px solid {Config.COLORS['primary_red']}; padding-bottom: 10px; margin-top: 0;">
🌙 Executive Nightletter <span style="float: right; color: {Config.COLORS['text_muted']}; font-size: 1rem; font-weight: 400; margin-top: 10px;">{date_str}</span>
</h2>
<div style="margin-bottom: 2rem;">
<p style="margin: 0;"><strong>File Analyzed:</strong> <span style="color: {Config.COLORS['blue_1']};">{bom_name}</span></p>
<p style="margin: 5px 0 0 0;"><strong>System Status:</strong> <span style="color: {status_color}; font-weight: 600;">{status_text}</span></p>
</div>
<div style="display: flex; gap: 20px; margin-bottom: 2rem;">
<div style="flex: 1; background-color: {Config.COLORS['card_bg']}; padding: 1.5rem; border-radius: 8px; border-top: 4px solid {status_color}; text-align: center;">
<h4 style="margin: 0; color: {Config.COLORS['text_muted']};">Risk Score</h4>
<h1 style="margin: 5px 0 0 0; color: {Config.COLORS['text_main']}; font-size: 2.5rem;">{risk_score}<span style="font-size: 1rem; color: {Config.COLORS['text_muted']};">/100</span></h1>
</div>
<div style="flex: 1; background-color: {Config.COLORS['card_bg']}; padding: 1.5rem; border-radius: 8px; border-top: 4px solid {Config.COLORS['blue_1']}; text-align: center;">
<h4 style="margin: 0; color: {Config.COLORS['text_muted']};">Parts Evaluated</h4>
<h1 style="margin: 5px 0 0 0; color: {Config.COLORS['text_main']}; font-size: 2.5rem;">{total_parts}</h1>
</div>
<div style="flex: 1; background-color: {Config.COLORS['card_bg']}; padding: 1.5rem; border-radius: 8px; border-top: 4px solid {Config.COLORS['primary_red']}; text-align: center;">
<h4 style="margin: 0; color: {Config.COLORS['text_muted']};">Total Issues</h4>
<h1 style="margin: 5px 0 0 0; color: {Config.COLORS['text_main']}; font-size: 2.5rem;">{total_issues}</h1>
</div>
</div>
<div style="display: flex; gap: 40px;">
<div style="flex: 1;">
<h4 style="color: {Config.COLORS['text_muted']}; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 5px;">🚨 Issue Breakdown</h4>
<ul style="list-style-type: none; padding-left: 0;">
<li style="margin-bottom: 5px;"><strong style="color: {Config.COLORS['primary_red']};">CRITICAL:</strong> {stats.get('critical', 0)}</li>
<li style="margin-bottom: 5px;"><strong style="color: {Config.COLORS['orange']};">ERRORS:</strong> {stats.get('errors', 0)}</li>
<li style="margin-bottom: 5px;"><strong style="color: {Config.COLORS['yellow']};">WARNINGS:</strong> {stats.get('warnings', 0)}</li>
</ul>
</div>
<div style="flex: 1.5;">
<h4 style="color: {Config.COLORS['text_muted']}; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 5px;">🎯 Top Problematic Features</h4>
<ul style="padding-left: 20px;">
{top_features_html}
</ul>
</div>
<div style="flex: 1.5;">
<h4 style="color: {Config.COLORS['text_muted']}; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 5px;">👤 Engineer Workload</h4>
<ul style="padding-left: 20px;">
{top_engineers_html}
</ul>
</div>
</div>
<p style="margin-top: 2rem; font-size: 0.85rem; color: {Config.COLORS['text_muted']}; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 10px;">
Engineers have been notified via the automated queue system to correct their respective items. Please review the complete dashboard for granular analytics.<br>
<em>— Quick Release_ Validation System</em>
</p>
</div>"""
    st.markdown(html_preview, unsafe_allow_html=True)

    # --- 2. The ASCII Plain Text Payload (For the Email Button) ---
    # Using explicit \r\n for Outlook formatting
    nightletter_subject = f"🌙 Daily BoM Validation Nightletter - {date_str}"
    nightletter_body = f"""=========================================================
          🌙 EXECUTIVE NIGHTLETTER - BoM VALIDATION
=========================================================
Date: {date_str}
File Analyzed: {bom_name}
Status: {status_text}

📊 KEY METRICS
---------------------------------------------------------
• Buildability Risk Score : {risk_score}/100
• Total Parts Evaluated   : {total_parts}
• Total Issues Detected   : {total_issues}

🚨 ISSUE BREAKDOWN
---------------------------------------------------------
• [CRITICAL] : {stats.get('critical', 0)}
• [ERROR]    : {stats.get('errors', 0)}
• [WARNING]  : {stats.get('warnings', 0)}

🎯 TOP PROBLEMATIC FEATURES
---------------------------------------------------------
{top_features_plain}
👤 ENGINEER WORKLOAD (Top Action Required)
---------------------------------------------------------
{top_engineers_plain}
=========================================================
Engineers have been notified via the automated queue system.
Generated by Quick Release_ Validation System
=========================================================\r\n"""

    mailto_link = f"mailto:management_team@quickrelease.co.uk?subject={urllib.parse.quote(nightletter_subject)}&body={urllib.parse.quote(nightletter_body)}"
    
    col1, col2 = st.columns([1, 1])
    with col1:
        # Removed target="_blank"
        btn_html = f"""<a href="{mailto_link}" style="text-decoration: none;">
<button style="background: linear-gradient(135deg, {Config.COLORS['green']} 0%, #3A6825 100%); color: white; border: none; padding: 15px 30px; border-radius: 8px; font-weight: 600; font-size: 1.1rem; cursor: pointer; box-shadow: 0 4px 6px rgba(0,0,0,0.3); width: 100%; font-family: 'Inter', sans-serif;">🚀 Send Plain-Text Nightletter via Outlook</button>
</a>"""
        st.markdown(btn_html, unsafe_allow_html=True)
    with col2:
        st.markdown("<p style='color: #A6A8AA; font-size: 0.9rem; margin-top: 10px;'><em>Tip: The button above generates a perfectly formatted plain-text email. If you prefer the styled visual version, you can highlight, copy, and paste the preview box above directly into an HTML-supported email client.</em></p>", unsafe_allow_html=True)

def page_dashboard():
    st.markdown('<div class="main-title"><h1>🏠 Dashboard Home</h1></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="info-card"><h3>Step 1: Upload Data</h3><p>Upload your BoM and PDL files to begin the validation process.</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="info-card"><h3>Step 2: Analyze & Fix</h3><p>Review the Action Center for step-by-step recommended fixes based on PDL rules.</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="info-card"><h3>Step 3: Communicate</h3><p>Generate automated emails to D&R Engineers and nightly management summaries.</p></div>""", unsafe_allow_html=True)
        
    if st.session_state.get('bom_df') is not None:
        st.markdown("### Current Workspace Status")
        st.success(f"📁 BoM Loaded: {st.session_state.get('bom_filename')}")
        if st.session_state.get('pdl_df') is not None:
            st.success(f"📁 PDL Loaded: {st.session_state.get('pdl_filename')}")
        else:
            st.warning("⚠️ PDL Guidance missing. Analytics will be limited.")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(page_title="Feature Code Validator | QR_", page_icon="📋", layout="wide", initial_sidebar_state="expanded")
    apply_custom_theme()
    
    if 'bom_df' not in st.session_state: st.session_state.bom_df = None
    if 'pdl_df' not in st.session_state: st.session_state.pdl_df = None
    
    show_logo()
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio("🧭 Navigation", ["Dashboard", "Upload & Validate", "Analytics", "Auto Emails", "Nightletter"], index=0)
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🗑️ Reset Workspace", use_container_width=True):
        for key in ['bom_df', 'pdl_df', 'bom_filename', 'pdl_filename', 'validation_results', 'validation_stats', 'run_complete']:
            if key in st.session_state: del st.session_state[key]
        st.rerun()
        
    st.sidebar.caption(f"© {datetime.now().year} Quick Release_")
    
    if page == "Dashboard": page_dashboard()
    elif page == "Upload & Validate": page_upload_validate()
    elif page == "Analytics": page_analytics()
    elif page == "Auto Emails": page_auto_emails()
    elif page == "Nightletter": page_nightletter()

if __name__ == "__main__":
    main()
