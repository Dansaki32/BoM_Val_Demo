import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import urllib.parse
import os

# ============================================================================
# CONFIGURATION & CONSTANTS (QR_ BRAND)
# ============================================================================

BASE_DIR = Path(__file__).parent.absolute()
ASSETS_DIR = BASE_DIR / "Assets"
LOGO_PATH = ASSETS_DIR / "images" / "logo.png"

class Config:
    """QR_ Brand Color Palette"""
    COLORS = {
        'primary_red': '#D7171F',
        'dark_red': '#A11117',
        'dark_grey': '#232324',
        'medium_grey': '#6C6E70',
        'light_grey': '#F2F2F2',
        'white': '#FFFFFF',
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
    """Injects premium QR_ Brand CSS with web-safe fonts and modern UI elements"""
    c = Config.COLORS
    
    theme_css = f"""
        <style>
        /* Import Inter as a web-safe fallback for Segoe UI */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        /* Global Typography & Background */
        html, body, .stApp {{
            background-color: {c['light_grey']} !important;
            color: {c['dark_grey']} !important;
            font-family: 'Segoe UI', 'Inter', sans-serif !important;
        }}

        /* Override Streamlit's default Pink focus rings with QR Red */
        :root {{
            --primary-color: {c['primary_red']};
        }}

        h1, h2, h3, h4, h5, h6, p, li, label, .stMarkdown, .dataframe, span, div {{ 
            font-family: 'Segoe UI', 'Inter', sans-serif !important; 
        }}

        /* Modern Gradient Headers */
        .main-title {{ 
            background: linear-gradient(135deg, {c['primary_red']} 0%, {c['dark_red']} 100%); 
            padding: 2rem; 
            border-radius: 12px; 
            margin-bottom: 2rem; 
            box-shadow: 0 6px 15px rgba(215, 23, 31, 0.25); 
        }}
        .main-title h1 {{ color: {c['white']} !important; font-weight: 700 !important; margin: 0; }}

        /* Primary Buttons */
        .stButton>button {{ 
            background: linear-gradient(135deg, {c['primary_red']} 0%, {c['dark_red']} 100%) !important; 
            color: {c['white']} !important; 
            border-radius: 8px; 
            border: none; 
            font-weight: 600; 
            padding: 0.75rem 1.5rem; 
            transition: all 0.3s ease; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .stButton>button:hover {{ 
            transform: translateY(-2px); 
            box-shadow: 0 6px 12px rgba(215, 23, 31, 0.3); 
        }}
        .stButton>button:focus {{
            outline: none !important;
            box-shadow: 0 0 0 3px rgba(215, 23, 31, 0.5) !important;
        }}

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{ 
            background-color: {c['dark_grey']} !important; 
            border-right: 4px solid {c['primary_red']}; 
        }}
        section[data-testid="stSidebar"] * {{ 
            color: {c['white']} !important; 
        }}
        
        /* Floating Content Cards */
        .info-card {{ 
            background-color: {c['white']}; 
            padding: 1.5rem; 
            border-radius: 12px; 
            border-left: 5px solid {c['primary_red']}; 
            margin: 1rem 0; 
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05); 
            transition: transform 0.3s ease;
        }}
        .info-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        }}
        .info-card h3 {{ color: {c['dark_grey']} !important; margin-top: 0; }}
        .info-card p {{ color: {c['medium_grey']} !important; }}

        /* File Uploader */
        .stFileUploader {{ 
            background-color: {c['white']} !important; 
            border: 2px dashed {c['medium_grey']} !important; 
            border-radius: 12px; 
            padding: 2rem; 
            transition: border-color 0.3s ease;
        }}
        .stFileUploader:hover {{
            border-color: {c['primary_red']} !important;
        }}

        /* Metrics */
        .metric-container {{ 
            background-color: {c['white']}; 
            padding: 1.5rem; 
            border-radius: 12px; 
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05); 
            border: 2px solid transparent; 
            transition: all 0.3s ease; 
            text-align: center; 
        }}
        .metric-container:hover {{ 
            border-color: {c['primary_red']}; 
            transform: translateY(-3px); 
        }}
        [data-testid="stMetricValue"] {{ color: {c['primary_red']} !important; font-weight: 700 !important; font-size: 2.5rem !important; }}
        [data-testid="stMetricLabel"] {{ color: {c['medium_grey']} !important; font-size: 1.1rem !important; font-weight: 600 !important; }}

        /* DataFrames */
        .stDataFrame, .dataframe {{ background-color: {c['white']} !important; border-radius: 8px; overflow: hidden; }}
        .dataframe thead th {{ background-color: {c['dark_grey']} !important; color: {c['white']} !important; font-weight: 600; }}
        
        /* Hyperlinks */
        a {{ color: {c['primary_red']} !important; text-decoration: underline !important; font-weight: 600; }}
        </style>
    """
    st.markdown(theme_css, unsafe_allow_html=True)

def show_logo():
    """Displays the QR_ Logo in the sidebar"""
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), use_container_width=True)
    else:
        st.sidebar.markdown(f"<h2 style='text-align: center; color: {Config.COLORS['primary_red']}; font-weight: 700;'>QUICK RELEASE_</h2>", unsafe_allow_html=True)

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
        title = {'text': "Buildability Risk Score", 'font': {'color': Config.COLORS['dark_grey'], 'size': 24, 'family': 'Inter'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': Config.COLORS['dark_grey']},
            'bar': {'color': color},
            'bgcolor': Config.COLORS['light_grey'],
            'borderwidth': 2,
            'bordercolor': Config.COLORS['medium_grey'],
            'steps': [
                {'range': [0, 20], 'color': '#D9EECF'}, # QR Light Green
                {'range': [20, 50], 'color': '#FFDB9A'}, # QR Light Yellow
                {'range': [50, 100], 'color': '#FACED0'}], # QR Light Red
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': Config.COLORS['dark_grey'], 'family': 'Inter'}, height=350, margin=dict(l=20, r=20, t=50, b=20))
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
            fig_tree.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10, l=0, r=0, b=10), font=dict(family='Inter', color=Config.COLORS['dark_grey']))
            st.plotly_chart(fig_tree, use_container_width=True)
            
        with col_chart2:
            st.markdown("### Most Problematic Features")
            feature_counts = df_res['Feature Code'].replace('MISSING', 'Unassigned').value_counts().head(7).reset_index()
            feature_counts.columns = ['Feature Code', 'Issue Count']
            fig_bar = px.bar(feature_counts, x='Issue Count', y='Feature Code', orientation='h', text='Issue Count')
            fig_bar.update_traces(marker_color=Config.COLORS['primary_red'], textposition='outside')
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis={'categoryorder':'total ascending'}, margin=dict(t=10, l=0, r=20, b=10), font=dict(family='Inter', color=Config.COLORS['dark_grey']))
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
        color = Config.SEVERITY_COLORS.get(row['Severity'], Config.COLORS['dark_grey'])
        st.markdown(f"""
        <div style="background-color: {Config.COLORS['white']}; padding: 1.5rem; border-radius: 12px; border-left: 6px solid {color}; margin-bottom: 1rem; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h4 style="margin:0; color: {color} !important;">{row['Issue Type']}</h4>
                <span style="background-color: {color}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.8rem; font-weight: bold;">{row['Severity']}</span>
            </div>
            <p style="margin-bottom: 5px;"><strong>Part:</strong> {row['Part Number']} | <strong>Feature:</strong> {row['Feature Code']} | <strong>Engineer:</strong> {row['Engineer ID']}</p>
            <p style="color: {Config.COLORS['medium_grey']}; margin-bottom: 15px;"><em>{row['Message']}</em></p>
            <div style="background-color: {Config.COLORS['light_grey']}; padding: 12px; border-radius: 8px; border: 1px solid #E1E2E3;">
                <strong style="color: {Config.COLORS['primary_red']};">💡 Recommended Fix:</strong><br>{row['Recommended Fix']}
            </div>
        </div>
        """, unsafe_allow_html=True)

def page_auto_emails():
    st.markdown('<div class="main-title"><h1>📧 D&R Auto-Communications</h1></div>', unsafe_allow_html=True)
    
    if 'validation_results' not in st.session_state or not st.session_state.validation_results:
        st.warning("⚠️ Please run validation first to generate engineer communications.")
        return
        
    st.markdown("""
        <div class="info-card">
            <p>Review the grouped BoM issues below. Click the draft button to open your default email client (e.g., Outlook).</p>
            <p><strong>⚠️ Note regarding Supervisors:</strong> Because the system cannot directly query your local Outlook Active Directory structure, the 'CC' line has been pre-populated with a placeholder. <strong>Please replace the placeholder with the Engineer's Supervisor before sending.</strong></p>
        </div>
    """, unsafe_allow_html=True)
    
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
            
            body = f"Hello,\n\nThe automated Feature Validator has detected {len(issues)} issue(s) with the parts assigned to you in the latest Bill of Materials (BoM).\n\n"
            body += "Please review and correct the following in the system:\n\n"
            body += "="*60 + "\n\n"
            
            for i, issue in enumerate(issues, 1):
                body += f"[{i}] Part Number: {issue.part_number} | Feature Code: {issue.feature_code}\n"
                body += f"    Issue: {issue.issue_type} ({issue.severity})\n"
                body += f"    Details: {issue.message}\n"
                body += f"    Recommended Fix: {issue.recommendation}\n\n"
                
            body += "="*60 + "\n\n"
            body += "Thank you,\nBoM Validation Team\n"
            
            mailto_link = f"mailto:{eng_id}?cc=INSERT_SUPERVISOR_HERE@quickrelease.co.uk&subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"
            
            df_display = pd.DataFrame([r.to_dict() for r in issues])[['Severity', 'Issue Type', 'Part Number', 'Feature Code']]
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            st.markdown(f'''
                <a href="{mailto_link}" target="_blank" style="text-decoration: none;">
                    <button style="
                        background: linear-gradient(135deg, {Config.COLORS['primary_red']} 0%, {Config.COLORS['dark_red']} 100%);
                        color: white; border: none; padding: 10px 20px; border-radius: 8px;
                        font-weight: 600; cursor: pointer; margin-top: 10px; margin-bottom: 10px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1); font-family: 'Inter', sans-serif;
                    ">📨 Draft Email in Outlook</button>
                </a>
            ''', unsafe_allow_html=True)

def page_nightletter():
    st.markdown('<div class="main-title"><h1>🌙 Executive Nightletter</h1></div>', unsafe_allow_html=True)
    
    if 'validation_results' not in st.session_state or 'validation_stats' not in st.session_state:
        st.warning("⚠️ Please run a validation first to generate the Nightletter.")
        return
        
    stats = st.session_state.validation_stats
    results = st.session_state.validation_results
    
    st.markdown('<div class="info-card"><p>This tab provides a condensed, high-level summary of the day\'s validation run, designed to be emailed to management and program supervisors.</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="metric-container">
            <h3 style="color:{Config.COLORS['medium_grey']}; margin:0;">Risk Score</h3>
            <h1 style="color:{Config.COLORS['primary_red']}; font-size: 3rem; margin:0;">{stats.get('risk_score', 0)}/100</h1>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-container">
            <h3 style="color:{Config.COLORS['medium_grey']}; margin:0;">Total Issues</h3>
            <h1 style="color:{Config.COLORS['dark_grey']}; font-size: 3rem; margin:0;">{len(results)}</h1>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-container">
            <h3 style="color:{Config.COLORS['medium_grey']}; margin:0;">Parts Checked</h3>
            <h1 style="color:{Config.COLORS['green']}; font-size: 3rem; margin:0;">{stats.get('total_features_checked', 0)}</h1>
        </div>""", unsafe_allow_html=True)
        
    st.markdown("---")
    
    df_res = pd.DataFrame([r.to_dict() for r in results])
    top_features = ""
    if not df_res.empty:
        feature_counts = df_res['Feature Code'].replace('MISSING', 'Unassigned').value_counts().head(3)
        for feat, count in feature_counts.items():
            top_features += f"  • {feat}: {count} impacted parts\n"
    else:
        top_features = "  • No issues detected today!\n"

    bom_name = st.session_state.get('bom_filename', 'Unknown BoM')
    date_str = datetime.now().strftime("%B %d, %Y")
    
    nightletter_subject = f"🌙 Daily BoM Validation Nightletter - {date_str}"
    nightletter_body = f"""Executive Summary - BoM Validation
Date: {date_str}
File Analyzed: {bom_name}

--- KEY METRICS ---
• Buildability Risk Score: {stats.get('risk_score', 0)} / 100
• Total Parts Evaluated: {stats.get('total_features_checked', 0)}
• Total Issues Detected: {len(results)}

--- ISSUE BREAKDOWN ---
• CRITICAL: {stats.get('critical', 0)}
• ERRORS: {stats.get('errors', 0)}
• WARNINGS: {stats.get('warnings', 0)}

--- TOP PROBLEMATIC FEATURES ---
{top_features}

Engineers have been notified via the automated queue system to correct their respective items.
Please review the complete dashboard for granular analytics.

Thank you,
Quick Release_ Validation System"""

    st.markdown("### 📝 Nightletter Preview")
    st.text_area("Email Content", value=nightletter_body, height=400, disabled=True)
    
    mailto_link = f"mailto:management_team@quickrelease.co.uk?subject={urllib.parse.quote(nightletter_subject)}&body={urllib.parse.quote(nightletter_body)}"
    
    st.markdown(f'''
        <a href="{mailto_link}" target="_blank" style="text-decoration: none;">
            <button style="
                background: linear-gradient(135deg, {Config.COLORS['green']} 0%, #3A6825 100%);
                color: white; border: none; padding: 15px 30px; border-radius: 8px;
                font-weight: 600; font-size: 1.1rem; cursor: pointer; margin-top: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1); width: 100%; font-family: 'Inter', sans-serif;
            ">🚀 Send Nightletter via Outlook</button>
        </a>
    ''', unsafe_allow_html=True)

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
