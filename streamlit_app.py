import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import urllib.parse
import textwrap
import io
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ============================================================================
# 1. CONFIGURATION & BRANDING
# ============================================================================

BASE_DIR = Path(__file__).parent.absolute()
ASSETS_DIR = BASE_DIR / "Assets"
LOGO_PATH = ASSETS_DIR / "images" / "logo.png"
DB_PATH = BASE_DIR / "history.db"

class Config:
    COLORS = {
        'primary_red': '#D7171F', 'dark_red': '#A11117', 'main_bg': '#232324',
        'sidebar_bg': '#151517', 'card_bg': '#363738', 'card_hover': '#454647',
        'text_main': '#FFFFFF', 'text_muted': '#A6A8AA', 'blue_1': '#0070BB',
        'green': '#4D8B31', 'orange': '#EE4B0F', 'yellow': '#FFA602'
    }
    SEVERITY_COLORS = {'CRITICAL': '#D7171F', 'ERROR': '#EE4B0F', 'WARNING': '#FFA602', 'INFO': '#0070BB'}

# ============================================================================
# 2. DATABASE INIT (HISTORICAL TRENDS)
# ============================================================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS runs 
                 (timestamp TEXT, parts INTEGER, critical INTEGER, errors INTEGER, warnings INTEGER, score INTEGER)''')
    conn.commit()
    conn.close()

def log_run_to_db(parts, critical, errors, warnings, score):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)", 
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), parts, critical, errors, warnings, score))
    conn.commit()
    conn.close()

def get_historical_data():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM runs ORDER BY timestamp ASC", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df

# ============================================================================
# 3. STYLING & HELPER FUNCTIONS
# ============================================================================

def apply_custom_theme():
    c = Config.COLORS
    theme_css = textwrap.dedent(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, .stApp {{ background-color: {c['main_bg']} !important; color: {c['text_main']} !important; font-family: 'Inter', sans-serif !important; }}
        h1, h2, h3, h4, h5, h6, p, li, label, td, th {{ font-family: 'Inter', sans-serif !important; color: {c['text_main']}; }}
        .main-title {{ background: linear-gradient(135deg, {c['primary_red']} 0%, {c['dark_red']} 100%); padding: 2rem; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 8px 20px rgba(0,0,0,0.4); }}
        .stButton>button {{ background: linear-gradient(135deg, {c['primary_red']} 0%, {c['dark_red']} 100%) !important; color: {c['text_main']} !important; border-radius: 8px; font-weight: 600 !important; border: none; }}
        section[data-testid="stSidebar"] {{ background-color: {c['sidebar_bg']} !important; border-right: 2px solid {c['primary_red']}; }}
        .info-card {{ background-color: {c['card_bg']}; padding: 1.5rem; border-radius: 12px; border-left: 5px solid {c['primary_red']}; margin: 1rem 0; box-shadow: 0 4px 10px rgba(0,0,0,0.2); }}
        .metric-container {{ background-color: {c['card_bg']}; padding: 1.5rem; border-radius: 12px; text-align: center; border: 1px solid rgba(255,255,255,0.05); }}
        [data-testid="stMetricValue"] {{ color: {c['text_main']} !important; font-weight: 700 !important; font-size: 2.5rem !important; }}
        [data-testid="stMetricLabel"] {{ color: {c['text_muted']} !important; font-weight: 600 !important; text-transform: uppercase; }}
        </style>
    """)
    st.markdown(theme_css, unsafe_allow_html=True)

def show_logo():
    """Displays the QR_ Logo in the sidebar"""
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), use_container_width=True)
    else:
        st.sidebar.markdown(f"<h2 style='text-align: center; color: {Config.COLORS['primary_red']}; font-weight: 700; letter-spacing: 1px;'>QUICK RELEASE_</h2>", unsafe_allow_html=True)

# ============================================================================
# 4. VECTORIZED VALIDATION ENGINE
# ============================================================================

def run_vectorized_validation(bom_df, pdl_df, col_map):
    p_col, f_col, e_col, q_col = col_map['part'], col_map['feat'], col_map['eng'], col_map['qty']
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 1. Check Missing Features (Vectorized)
    status_text.text("Checking for missing data...")
    missing_mask = bom_df[f_col].isna() | (bom_df[f_col].astype(str).str.strip() == '') | (bom_df[f_col].astype(str).str.lower() == 'none')
    for _, row in bom_df[missing_mask].iterrows():
        results.append({
            'Part Number': str(row.get(p_col, 'Unknown')), 'Feature Code': 'MISSING',
            'Engineer ID': str(row.get(e_col, 'Unassigned')), 'Severity': 'ERROR',
            'Issue Type': 'Missing Data', 'Message': 'Part has no feature code assigned.',
            'Recommended Fix': 'Assign a valid feature code.', 'Resolved': False
        })
    progress_bar.progress(25)
    
    # Process PDL Rules Dynamically
    if pdl_df is not None and not pdl_df.empty and 'Rule_Type' in pdl_df.columns:
        valid_bom = bom_df[~missing_mask].copy()
        valid_bom[f_col] = valid_bom[f_col].astype(str)
        build_features = valid_bom[f_col].unique()
        
        status_text.text("Merging BoM with PDL Rules...")
        merged = valid_bom.merge(pdl_df, left_on=f_col, right_on='Feature_Code', how='inner')
        progress_bar.progress(50)
        
        # 2. Obsolete Features
        status_text.text("Validating Obsolete Features...")
        obs_df = merged[merged['Rule_Type'].str.upper() == 'OBSOLETE']
        for _, row in obs_df.iterrows():
            results.append({
                'Part Number': str(row[p_col]), 'Feature Code': str(row[f_col]),
                'Engineer ID': str(row.get(e_col, 'Unassigned')), 'Severity': 'CRITICAL',
                'Issue Type': 'Obsolete Feature', 'Message': 'Feature is marked as obsolete.',
                'Recommended Fix': f"Replace with superseding feature: {row.get('Constraint_Value', 'Unknown')}", 'Resolved': False
            })
        progress_bar.progress(70)
        
        # 3. Mutually Exclusive
        status_text.text("Checking Mutually Exclusive Conflicts...")
        mut_df = merged[merged['Rule_Type'].str.upper() == 'MUTUALLY_EXCLUSIVE']
        for _, row in mut_df.iterrows():
            conflict_feat = str(row.get('Constraint_Value', ''))
            if conflict_feat in build_features:
                results.append({
                    'Part Number': str(row[p_col]), 'Feature Code': str(row[f_col]),
                    'Engineer ID': str(row.get(e_col, 'Unassigned')), 'Severity': 'CRITICAL',
                    'Issue Type': 'Mutually Exclusive', 'Message': f"Conflicts with {conflict_feat} in the same build.",
                    'Recommended Fix': 'Review build configuration. Remove conflicting part.', 'Resolved': False
                })
        progress_bar.progress(85)
        
        # 4. Prerequisites
        status_text.text("Verifying Prerequisites...")
        req_df = merged[merged['Rule_Type'].str.upper() == 'REQUIRES']
        for _, row in req_df.iterrows():
            req_feat = str(row.get('Constraint_Value', ''))
            if req_feat not in build_features:
                results.append({
                    'Part Number': str(row[p_col]), 'Feature Code': str(row[f_col]),
                    'Engineer ID': str(row.get(e_col, 'Unassigned')), 'Severity': 'ERROR',
                    'Issue Type': 'Missing Dependency', 'Message': f"Requires prerequisite feature {req_feat}.",
                    'Recommended Fix': f"Add part with feature {req_feat} to the BoM.", 'Resolved': False
                })
                
        # 5. Max Quantity
        status_text.text("Checking Quantity Limits...")
        qty_df = merged[merged['Rule_Type'].str.upper() == 'MAX_QTY']
        for _, row in qty_df.iterrows():
            try:
                max_q = float(row.get('Constraint_Value', 999))
                actual_q = float(row.get(q_col, 1))
                if actual_q > max_q:
                    results.append({
                        'Part Number': str(row[p_col]), 'Feature Code': str(row[f_col]),
                        'Engineer ID': str(row.get(e_col, 'Unassigned')), 'Severity': 'WARNING',
                        'Issue Type': 'Quantity Exceeded', 'Message': f"Qty {actual_q} exceeds PDL limit of {max_q}.",
                        'Recommended Fix': 'Verify quantity requirements with engineering.', 'Resolved': False
                    })
            except: pass

    progress_bar.progress(100)
    status_text.text("Validation Complete!")
    return pd.DataFrame(results)

def calculate_stats(results_df, total_parts):
    if results_df.empty:
        return {'critical': 0, 'errors': 0, 'warnings': 0, 'risk_score': 0, 'total_parts': total_parts}
    
    active = results_df[~results_df['Resolved']]
    crit = len(active[active['Severity'] == 'CRITICAL'])
    errs = len(active[active['Severity'] == 'ERROR'])
    warns = len(active[active['Severity'] == 'WARNING'])
    
    total_issues = (crit * 3) + (errs * 2) + warns
    risk = min(total_issues * (100 / max(total_parts, 1)), 100)
    
    return {'critical': crit, 'errors': errs, 'warnings': warns, 'risk_score': round(risk), 'total_parts': total_parts}

# ============================================================================
# 5. PAGES
# ============================================================================

def page_dashboard():
    st.markdown('<div class="main-title"><h1>🏠 Dashboard & Trends</h1></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown("""<div class="info-card"><h3>1. Map & Validate</h3><p>Upload files and map columns dynamically.</p></div>""", unsafe_allow_html=True)
    with col2: st.markdown("""<div class="info-card"><h3>2. Interactive Fixes</h3><p>Mark issues as resolved in real-time.</p></div>""", unsafe_allow_html=True)
    with col3: st.markdown("""<div class="info-card"><h3>3. Direct Comms</h3><p>Send SMTP emails directly from the app.</p></div>""", unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("### 📈 Historical Risk Burn-down")
    
    hist_df = get_historical_data()
    if not hist_df.empty:
        fig = px.line(hist_df, x='timestamp', y='score', markers=True, title="Risk Score Over Time")
        fig.update_traces(line_color=Config.COLORS['primary_red'], marker=dict(size=8))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical data yet. Run a validation to start tracking trends!")

def page_upload():
    st.markdown('<div class="main-title"><h1>📤 Upload & Map Data</h1></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        bom_file = st.file_uploader("1. Upload BoM", type=["csv", "xlsx"])
        if bom_file:
            st.session_state.bom_df = pd.read_csv(bom_file) if bom_file.name.endswith('.csv') else pd.read_excel(bom_file)
            st.session_state.bom_filename = bom_file.name
            
    with col2:
        pdl_file = st.file_uploader("2. Upload PDL Rules", type=["csv", "xlsx"])
        if pdl_file:
            st.session_state.pdl_df = pd.read_csv(pdl_file) if pdl_file.name.endswith('.csv') else pd.read_excel(pdl_file)
            st.session_state.pdl_filename = pdl_file.name
            
    if 'bom_df' in st.session_state:
        st.markdown("### 🔀 Column Mapping")
        cols = list(st.session_state.bom_df.columns)
        
        c1, c2, c3, c4 = st.columns(4)
        map_p = c1.selectbox("Part Number Col", cols, index=0)
        map_f = c2.selectbox("Feature Col", cols, index=1 if len(cols)>1 else 0)
        map_e = c3.selectbox("Engineer Col", cols + ['None'], index=len(cols))
        map_q = c4.selectbox("Quantity Col", cols + ['None'], index=len(cols))
        
        if st.button("🚀 Run Vectorized Validation", type="primary", use_container_width=True):
            col_map = {'part': map_p, 'feat': map_f, 'eng': map_e if map_e != 'None' else map_p, 'qty': map_q if map_q != 'None' else map_p}
            
            res_df = run_vectorized_validation(st.session_state.bom_df, st.session_state.get('pdl_df'), col_map)
            st.session_state.results_df = res_df
            st.session_state.total_parts = len(st.session_state.bom_df[map_p].unique())
            
            # Log to DB
            stats = calculate_stats(res_df, st.session_state.total_parts)
            log_run_to_db(stats['total_parts'], stats['critical'], stats['errors'], stats['warnings'], stats['risk_score'])
            
            st.success("✅ Validation Complete! Navigate to Analytics to view results.")

    st.markdown("---")
    if st.button("🎲 Load Sample Data & Rules"):
        np.random.seed(42)
        st.session_state.bom_df = pd.DataFrame({
            'Part_No': [f"PN-{i}" for i in range(100)],
            'Feat_Code': np.random.choice(['ENG-V8', 'OBS-NAV', 'INT-LHD', 'INT-RHD', 'SUNROOF', 'WHEEL', None], 100),
            'Eng_ID': np.random.choice(['john@qr.com', 'mary@qr.com'], 100),
            'Qty': np.random.randint(1, 6, 100)
        })
        st.session_state.pdl_df = pd.DataFrame({
            'Feature_Code': ['OBS-NAV', 'INT-LHD', 'INT-RHD', 'SUNROOF', 'WHEEL'],
            'Rule_Type': ['OBSOLETE', 'MUTUALLY_EXCLUSIVE', 'MUTUALLY_EXCLUSIVE', 'REQUIRES', 'MAX_QTY'],
            'Constraint_Value': ['NAV-02', 'INT-RHD', 'INT-LHD', 'ROOF', '4']
        })
        st.rerun()

def page_analytics():
    st.markdown('<div class="main-title"><h1>📊 Interactive Analytics</h1></div>', unsafe_allow_html=True)
    if 'results_df' not in st.session_state:
        st.warning("⚠️ Please run validation first.")
        return
        
    # Recalculate stats based on interactive editor
    stats = calculate_stats(st.session_state.results_df, st.session_state.total_parts)
    
    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=stats['risk_score'], title={'text': "Risk Score", 'font': {'color': 'white'}},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': Config.COLORS['primary_red']}, 'bgcolor': Config.COLORS['sidebar_bg']}
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=250, margin=dict(l=10, r=10, t=30, b=10))
    
    with col1: st.plotly_chart(fig, use_container_width=True)
    with col2: st.markdown(f'<div class="metric-container"><h3 style="color:{Config.COLORS["primary_red"]}">{stats["critical"]}</h3><p>Critical</p></div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="metric-container"><h3 style="color:{Config.COLORS["orange"]}">{stats["errors"]}</h3><p>Errors</p></div>', unsafe_allow_html=True)
    with col4: st.markdown(f'<div class="metric-container"><h3 style="color:{Config.COLORS["yellow"]}">{stats["warnings"]}</h3><p>Warnings</p></div>', unsafe_allow_html=True)

    st.markdown("### 🛠️ Action Center (Interactive)")
    st.caption("Check the 'Resolved' box to dynamically update your risk score.")
    
    # Interactive Data Editor
    edited_df = st.data_editor(
        st.session_state.results_df,
        column_config={"Resolved": st.column_config.CheckboxColumn("Resolved?", default=False)},
        disabled=["Part Number", "Feature Code", "Engineer ID", "Severity", "Issue Type", "Message", "Recommended Fix"],
        use_container_width=True, hide_index=True
    )
    st.session_state.results_df = edited_df

    # Excel Download
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        edited_df.to_excel(writer, index=False, sheet_name='Validation Results')
    st.download_button("📥 Download Report (.xlsx)", data=output.getvalue(), file_name="Validation_Report.xlsx", mime="application/vnd.ms-excel")

def page_communications():
    st.markdown('<div class="main-title"><h1>📧 Communications</h1></div>', unsafe_allow_html=True)
    if 'results_df' not in st.session_state:
        st.warning("⚠️ Please run validation first.")
        return
        
    active_issues = st.session_state.results_df[~st.session_state.results_df['Resolved']]
    
    with st.expander("⚙️ Direct SMTP Configuration (Optional)"):
        st.info("Configure your SMTP server to send emails directly in the background, bypassing Outlook character limits.")
        c1, c2 = st.columns(2)
        smtp_server = c1.text_input("SMTP Server", "smtp.office365.com")
        smtp_port = c2.number_input("SMTP Port", 587)
        smtp_user = c1.text_input("Email Address")
        smtp_pass = c2.text_input("Password", type="password")
        
    st.markdown("### Engineer Queues")
    for eng, group in active_issues.groupby('Engineer ID'):
        if eng == 'Unassigned' or eng == 'None': continue
        
        with st.expander(f"👤 {eng} — {len(group)} Actions"):
            st.dataframe(group[['Severity', 'Part Number', 'Issue Type']], use_container_width=True, hide_index=True)
            
            body = f"Action required for {len(group)} parts.\r\n\r\n"
            for _, r in group.iterrows():
                body += f"- {r['Part Number']} ({r['Feature Code']}): {r['Message']}\r\n"
            
            mailto = f"mailto:{eng}?subject=BoM Action Required&body={urllib.parse.quote(body)}"
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f'<a href="{mailto}"><button style="padding:10px; border-radius:5px; background:{Config.COLORS["blue_1"]}; color:white; border:none; width:100%;">Draft in Outlook</button></a>', unsafe_allow_html=True)
            with col2:
                if st.button(f"Send Direct via SMTP", key=f"send_{eng}"):
                    if not smtp_user or not smtp_pass:
                        st.error("Please configure SMTP settings above first.")
                    else:
                        try:
                            msg = MIMEMultipart()
                            msg['From'], msg['To'], msg['Subject'] = smtp_user, eng, "BoM Action Required"
                            msg.attach(MIMEText(body, 'plain'))
                            server = smtplib.SMTP(smtp_server, smtp_port)
                            server.starttls()
                            server.login(smtp_user, smtp_pass)
                            server.send_message(msg)
                            server.quit()
                            st.success(f"Email sent directly to {eng}!")
                        except Exception as e:
                            st.error(f"SMTP Error: {e}")

def page_nightletter():
    st.markdown('<div class="main-title"><h1>🌙 Executive Nightletter</h1></div>', unsafe_allow_html=True)
    if 'results_df' not in st.session_state:
        st.warning("⚠️ Please run validation first.")
        return
        
    stats = calculate_stats(st.session_state.results_df, st.session_state.total_parts)
    date_str = datetime.now().strftime("%B %d, %Y")
    
    html_content = textwrap.dedent(f"""
        <html><body style="font-family: Arial, sans-serif; color: #333; padding: 20px;">
        <h2 style="color: #D7171F; border-bottom: 2px solid #D7171F;">Executive Nightletter - {date_str}</h2>
        <h3>Key Metrics</h3>
        <ul>
            <li><b>Risk Score:</b> {stats['risk_score']}/100</li>
            <li><b>Parts Evaluated:</b> {stats['total_parts']}</li>
            <li><b>Total Active Issues:</b> {stats['critical'] + stats['errors'] + stats['warnings']}</li>
        </ul>
        <h3>Breakdown</h3>
        <ul>
            <li style="color: #D7171F;"><b>Critical:</b> {stats['critical']}</li>
            <li style="color: #EE4B0F;"><b>Errors:</b> {stats['errors']}</li>
            <li style="color: #FFA602;"><b>Warnings:</b> {stats['warnings']}</li>
        </ul>
        <p><i>Generated by Quick Release_ Validation System</i></p>
        </body></html>
    """)
    
    st.components.v1.html(html_content, height=400, scrolling=True)
    
    st.download_button("📥 Download Nightletter as HTML (Print to PDF)", data=html_content, file_name=f"Nightletter_{datetime.now().strftime('%Y%m%d')}.html", mime="text/html")

# ============================================================================
# 6. MAIN APP ROUTING (Native Multi-Page)
# ============================================================================

def main():
    st.set_page_config(page_title="Feature Validator | QR_", page_icon="📋", layout="wide")
    init_db()
    apply_custom_theme()
    show_logo()
    
    # Native Streamlit Navigation (Streamlit >= 1.36)
    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame()

    pages = {
        "Core Workflow": [
            st.Page(page_dashboard, title="Dashboard & Trends", icon="🏠"),
            st.Page(page_upload, title="Upload & Validate", icon="📤"),
            st.Page(page_analytics, title="Action Center", icon="📊")
        ],
        "Communications": [
            st.Page(page_communications, title="Engineer Emails", icon="📧"),
            st.Page(page_nightletter, title="Nightletter Export", icon="🌙")
        ]
    }
    
    pg = st.navigation(pages)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Reset Workspace", use_container_width=True):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()
        
    pg.run()

if __name__ == "__main__":
    main()
