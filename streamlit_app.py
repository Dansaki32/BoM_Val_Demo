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
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), use_container_width=True)
    else:
        st.sidebar.markdown(f"<h2 style='text-align: center; color: {Config.COLORS['primary_red']}; font-weight: 700; letter-spacing: 1px;'>QUICK RELEASE_</h2>", unsafe_allow_html=True)

def auto_detect_columns(df):
    """Magic column detection based on common naming conventions"""
    cols = [c.upper() for c in df.columns]
    part_col = next((c for c in df.columns if 'PART' in c.upper()), df.columns[0])
    feat_col = next((c for c in df.columns if 'FEAT' in c.upper() or 'CODE' in c.upper()), df.columns[1] if len(df.columns)>1 else None)
    eng_col = next((c for c in df.columns if c.upper() in ['ENGINEER_ID', 'D&R_ID', 'ENGINEER', 'OWNER', 'ENG_ID']), None)
    qty_col = next((c for c in df.columns if 'QTY' in c.upper() or 'QUANTITY' in c.upper()), None)
    return part_col, feat_col, eng_col, qty_col

# ============================================================================
# 4. VECTORIZED VALIDATION ENGINE
# ============================================================================

def run_vectorized_validation(bom_df, pdl_df):
    p_col, f_col, e_col, q_col = auto_detect_columns(bom_df)
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 1. Check Missing Features
    status_text.text("Checking for missing data...")
    if f_col:
        missing_mask = bom_df[f_col].isna() | (bom_df[f_col].astype(str).str.strip() == '') | (bom_df[f_col].astype(str).str.lower() == 'none')
        for _, row in bom_df[missing_mask].iterrows():
            eng_id = str(row[e_col]) if e_col and pd.notna(row[e_col]) else 'Unassigned'
            results.append({
                'Part Number': str(row.get(p_col, 'Unknown')), 'Feature Code': 'MISSING',
                'Engineer ID': eng_id, 'Severity': 'ERROR',
                'Issue Type': 'Missing Data', 'Message': 'Part has no feature code assigned.',
                'Recommended Fix': 'Assign a valid feature code.', 'Resolved': False
            })
    progress_bar.progress(25)
    
    # 2. Process PDL Rules Dynamically
    if pdl_df is not None and not pdl_df.empty and 'Rule_Type' in pdl_df.columns and f_col:
        valid_bom = bom_df[~missing_mask].copy() if f_col else bom_df.copy()
        valid_bom[f_col] = valid_bom[f_col].astype(str)
        build_features = valid_bom[f_col].unique()
        
        status_text.text("Merging BoM with PDL Rules...")
        merged = valid_bom.merge(pdl_df, left_on=f_col, right_on='Feature_Code', how='inner')
        progress_bar.progress(50)
        
        for _, row in merged.iterrows():
            rule = str(row.get('Rule_Type', '')).upper()
            val = str(row.get('Constraint_Value', ''))
            eng_id = str(row[e_col]) if e_col and pd.notna(row[e_col]) else 'Unassigned'
            
            if rule == 'OBSOLETE':
                results.append({
                    'Part Number': str(row[p_col]), 'Feature Code': str(row[f_col]),
                    'Engineer ID': eng_id, 'Severity': 'CRITICAL',
                    'Issue Type': 'Obsolete Feature', 'Message': 'Feature is marked as obsolete.',
                    'Recommended Fix': f"Replace with superseding feature: {val}", 'Resolved': False
                })
            elif rule == 'MUTUALLY_EXCLUSIVE' and val in build_features:
                results.append({
                    'Part Number': str(row[p_col]), 'Feature Code': str(row[f_col]),
                    'Engineer ID': eng_id, 'Severity': 'CRITICAL',
                    'Issue Type': 'Mutually Exclusive', 'Message': f"Conflicts with {val} in the same build.",
                    'Recommended Fix': 'Review build configuration. Remove conflicting part.', 'Resolved': False
                })
            elif rule == 'REQUIRES' and val not in build_features:
                results.append({
                    'Part Number': str(row[p_col]), 'Feature Code': str(row[f_col]),
                    'Engineer ID': eng_id, 'Severity': 'ERROR',
                    'Issue Type': 'Missing Dependency', 'Message': f"Requires prerequisite feature {val}.",
                    'Recommended Fix': f"Add part with feature {val} to the BoM.", 'Resolved': False
                })
            elif rule == 'MAX_QTY' and q_col:
                try:
                    if float(row.get(q_col, 1)) > float(val):
                        results.append({
                            'Part Number': str(row[p_col]), 'Feature Code': str(row[f_col]),
                            'Engineer ID': eng_id, 'Severity': 'WARNING',
                            'Issue Type': 'Quantity Exceeded', 'Message': f"Qty {row[q_col]} exceeds PDL limit of {val}.",
                            'Recommended Fix': 'Verify quantity requirements with engineering.', 'Resolved': False
                        })
                except: pass

    progress_bar.progress(100)
    status_text.text("Validation Complete!")
    return pd.DataFrame(results)

def calculate_stats(results_df, total_parts):
    if results_df is None or results_df.empty:
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
    with col1: st.markdown("""<div class="info-card"><h3>1. Upload Data</h3><p>Upload files and let the system auto-detect columns.</p></div>""", unsafe_allow_html=True)
    with col2: st.markdown("""<div class="info-card"><h3>2. Interactive Fixes</h3><p>Mark issues as resolved in real-time to update charts.</p></div>""", unsafe_allow_html=True)
    with col3: st.markdown("""<div class="info-card"><h3>3. Communications</h3><p>Generate emails to engineers and executive summaries.</p></div>""", unsafe_allow_html=True)
        
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
    st.markdown('<div class="main-title"><h1>📤 Upload & Validate</h1></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        bom_file = st.file_uploader("1. Upload BoM", type=["csv", "xlsx"])
        if bom_file:
            st.session_state.bom_df = pd.read_csv(bom_file) if bom_file.name.endswith('.csv') else pd.read_excel(bom_file)
            st.session_state.bom_filename = bom_file.name
            st.success(f"✅ Loaded: {bom_file.name}")
            
    with col2:
        pdl_file = st.file_uploader("2. Upload PDL Rules", type=["csv", "xlsx"])
        if pdl_file:
            st.session_state.pdl_df = pd.read_csv(pdl_file) if pdl_file.name.endswith('.csv') else pd.read_excel(pdl_file)
            st.session_state.pdl_filename = pdl_file.name
            st.success(f"✅ Loaded: {pdl_file.name}")
            
    if st.session_state.get('bom_df') is not None:
        st.markdown("---")
        if st.button("🚀 Run Advanced Validation", type="primary", use_container_width=True):
            res_df = run_vectorized_validation(st.session_state.bom_df, st.session_state.get('pdl_df'))
            st.session_state.results_df = res_df
            p_col, _, _, _ = auto_detect_columns(st.session_state.bom_df)
            st.session_state.total_parts = len(st.session_state.bom_df[p_col].unique())
            
            # Log to DB
            stats = calculate_stats(res_df, st.session_state.total_parts)
            log_run_to_db(stats['total_parts'], stats['critical'], stats['errors'], stats['warnings'], stats['risk_score'])
            
            st.success("✅ Validation Complete! Navigate to 'Action Center' in the sidebar to view results.")

    st.markdown("---")
    if st.button("🎲 Load Sample Data & Rules"):
        np.random.seed(42)
        st.session_state.bom_df = pd.DataFrame({
            'Part_No': [f"PN-{i}" for i in range(100)],
            'Feat_Code': np.random.choice(['ENG-V8', 'OBS-NAV', 'INT-LHD', 'INT-RHD', 'SUNROOF', 'WHEEL', None], 100),
            'Eng_ID': np.random.choice(['john.smith@quickrelease.co.uk', 'mary.jane@quickrelease.co.uk'], 100),
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
    
    if st.session_state.get('results_df') is None:
        st.warning("⚠️ Please go to 'Upload & Validate' and run the validation first.")
        return

    df = st.session_state.results_df
    active_df = df[~df['Resolved']].copy()
    stats = calculate_stats(df, st.session_state.total_parts)
    
    # --- Top Row: Gauge & Metrics ---
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

    st.markdown("---")

    # --- Middle Row: Restored Visual Graphs ---
    if not active_df.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🚨 Issue Distribution")
            fig_sun = px.sunburst(active_df, path=['Severity', 'Issue Type'], color='Severity', color_discrete_map=Config.SEVERITY_COLORS)
            fig_sun.update_traces(textinfo="label+value")
            fig_sun.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=0, l=0, r=0, b=0), font=dict(family='Inter', color=Config.COLORS['text_main']))
            st.plotly_chart(fig_sun, use_container_width=True)
        with c2:
            st.markdown("### 🎯 Top Problematic Features")
            feat_counts = active_df['Feature Code'].value_counts().head(7).reset_index()
            feat_counts.columns = ['Feature Code', 'Count']
            fig_bar = px.bar(feat_counts, x='Count', y='Feature Code', orientation='h', text='Count')
            fig_bar.update_traces(marker_color=Config.COLORS['primary_red'], textposition='outside')
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis={'categoryorder':'total ascending'}, margin=dict(t=0, l=0, r=0, b=0), font=dict(family='Inter', color=Config.COLORS['text_main']))
            st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🛠️ Action Center (Interactive)")
    st.caption("Mark issues as **Resolved** in the table below. Charts above will update automatically.")
    
    if not df.empty:
        edited_df = st.data_editor(
            df,
            column_config={
                "Resolved": st.column_config.CheckboxColumn("Resolved?", default=False),
                "Severity": st.column_config.Column("Severity", width="medium"),
                "Issue Type": st.column_config.Column("Issue Type", width="medium"),
                "Message": st.column_config.Column("Message", width="large"),
                "Recommended Fix": st.column_config.Column("Recommended Fix", width="large"),
            },
            disabled=["Part Number", "Feature Code", "Engineer ID", "Severity", "Issue Type", "Message", "Recommended Fix"],
            use_container_width=True, hide_index=True, height=500
        )
        
        # Real-time update logic
        if not edited_df.equals(st.session_state.results_df):
            st.session_state.results_df = edited_df
            st.rerun()

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            edited_df.to_excel(writer, index=False, sheet_name='Validation Results')
        st.download_button("📥 Download Report (.xlsx)", data=output.getvalue(), file_name="Validation_Report.xlsx", mime="application/vnd.ms-excel")
    else:
        st.success("✨ Zero issues detected! Your BoM is perfectly clean.")

def page_communications():
    st.markdown('<div class="main-title"><h1>📧 Communications</h1></div>', unsafe_allow_html=True)
    
    if st.session_state.get('results_df') is None:
        st.warning("⚠️ Please go to 'Upload & Validate' and run the validation first.")
        return
        
    active_issues = st.session_state.results_df[~st.session_state.results_df['Resolved']]
    if active_issues.empty:
        st.success("✨ No active issues to communicate!")
        return
        
    st.markdown("### Engineer Queues")
    # Will now correctly group by Engineer Email (e.g. john.smith@...)
    for eng, group in active_issues.groupby('Engineer ID'):
        if eng == 'Unassigned' or eng == 'None': continue
        
        with st.expander(f"👤 {eng} — {len(group)} Actions"):
            st.dataframe(group[['Severity', 'Part Number', 'Issue Type']], use_container_width=True, hide_index=True)
            
            body = f"Hello,\r\n\r\nAction is required for {len(group)} parts in the latest BoM validation.\r\n\r\n"
            for _, r in group.iterrows():
                body += f"- {r['Part Number']} ({r['Feature Code']}): {r['Message']}\r\n"
            body += "\r\nThank you,\r\nBoM Validation Team"
            
            mailto = f"mailto:{eng}?subject=BoM Action Required&body={urllib.parse.quote(body)}"
            
            btn_html = f"""<a href="{mailto}" style="text-decoration: none;">
            <button style="background: linear-gradient(135deg, {Config.COLORS['primary_red']} 0%, {Config.COLORS['dark_red']} 100%); color: white; border: none; padding: 10px 20px; border-radius: 8px; font-weight: 600; cursor: pointer; margin-top: 10px; margin-bottom: 10px; font-family: 'Inter', sans-serif;">📨 Draft Email in Outlook</button>
            </a>"""
            st.markdown(btn_html, unsafe_allow_html=True)

def page_nightletter():
    st.markdown('<div class="main-title"><h1>🌙 Executive Nightletter</h1></div>', unsafe_allow_html=True)
    
    if st.session_state.get('results_df') is None:
        st.warning("⚠️ Please go to 'Upload & Validate' and run the validation first.")
        return
        
    stats = calculate_stats(st.session_state.results_df, st.session_state.total_parts)
    date_str = datetime.now().strftime("%B %d, %Y")
    bom_name = st.session_state.get('bom_filename', 'Unknown BoM')
    
    # Assess Status
    if stats['risk_score'] < 20: 
        status_text, status_color = "🟢 ON TRACK", Config.COLORS['green']
    elif stats['risk_score'] < 50: 
        status_text, status_color = "🟡 AT RISK", Config.COLORS['yellow']
    else: 
        status_text, status_color = "🔴 CRITICAL", Config.COLORS['primary_red']

    # --- Restored Premium Visual HTML Preview ---
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
<h1 style="margin: 5px 0 0 0; color: {Config.COLORS['text_main']}; font-size: 2.5rem;">{stats['risk_score']}<span style="font-size: 1rem; color: {Config.COLORS['text_muted']};">/100</span></h1>
</div>
<div style="flex: 1; background-color: {Config.COLORS['card_bg']}; padding: 1.5rem; border-radius: 8px; border-top: 4px solid {Config.COLORS['blue_1']}; text-align: center;">
<h4 style="margin: 0; color: {Config.COLORS['text_muted']};">Parts Evaluated</h4>
<h1 style="margin: 5px 0 0 0; color: {Config.COLORS['text_main']}; font-size: 2.5rem;">{stats['total_parts']}</h1>
</div>
<div style="flex: 1; background-color: {Config.COLORS['card_bg']}; padding: 1.5rem; border-radius: 8px; border-top: 4px solid {Config.COLORS['primary_red']}; text-align: center;">
<h4 style="margin: 0; color: {Config.COLORS['text_muted']};">Active Issues</h4>
<h1 style="margin: 5px 0 0 0; color: {Config.COLORS['text_main']}; font-size: 2.5rem;">{stats['critical'] + stats['errors'] + stats['warnings']}</h1>
</div>
</div>
<div style="display: flex; gap: 40px;">
<div style="flex: 1;">
<h4 style="color: {Config.COLORS['text_muted']}; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 5px;">🚨 Issue Breakdown</h4>
<ul style="list-style-type: none; padding-left: 0;">
<li style="margin-bottom: 5px;"><strong style="color: {Config.COLORS['primary_red']};">CRITICAL:</strong> {stats['critical']}</li>
<li style="margin-bottom: 5px;"><strong style="color: {Config.COLORS['orange']};">ERRORS:</strong> {stats['errors']}</li>
<li style="margin-bottom: 5px;"><strong style="color: {Config.COLORS['yellow']};">WARNINGS:</strong> {stats['warnings']}</li>
</ul>
</div>
</div>
<p style="margin-top: 2rem; font-size: 0.85rem; color: {Config.COLORS['text_muted']}; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 10px;">
Engineers have been notified via the automated queue system to correct their respective items.<br>
<em>— Quick Release_ Validation System</em>
</p>
</div>"""
    st.markdown(html_preview, unsafe_allow_html=True)

    # --- Plain Text Payload for Email ---
    nightletter_body = f"""=========================================================
          🌙 EXECUTIVE NIGHTLETTER - BoM VALIDATION
=========================================================
Date: {date_str}
File Analyzed: {bom_name}
Status: {status_text}

📊 KEY METRICS
---------------------------------------------------------
• Buildability Risk Score : {stats['risk_score']}/100
• Total Parts Evaluated   : {stats['total_parts']}
• Active Issues Detected  : {stats['critical'] + stats['errors'] + stats['warnings']}

🚨 ISSUE BREAKDOWN
---------------------------------------------------------
• [CRITICAL] : {stats['critical']}
• [ERROR]    : {stats['errors']}
• [WARNING]  : {stats['warnings']}

=========================================================
Generated by Quick Release_ Validation System
=========================================================\r\n"""

    mailto_link = f"mailto:management_team@quickrelease.co.uk?subject=Daily BoM Nightletter - {date_str}&body={urllib.parse.quote(nightletter_body)}"
    
    btn_html = f"""<a href="{mailto_link}" style="text-decoration: none;">
<button style="background: linear-gradient(135deg, {Config.COLORS['green']} 0%, #3A6825 100%); color: white; border: none; padding: 15px 30px; border-radius: 8px; font-weight: 600; font-size: 1.1rem; cursor: pointer; box-shadow: 0 4px 6px rgba(0,0,0,0.3); width: 100%; font-family: 'Inter', sans-serif;">🚀 Send Plain-Text Nightletter via Outlook</button>
</a>"""
    st.markdown(btn_html, unsafe_allow_html=True)

# ============================================================================
# 6. MAIN APP ROUTING (Native Multi-Page)
# ============================================================================

def main():
    st.set_page_config(page_title="Feature Validator | QR_", page_icon="📋", layout="wide")
    init_db()
    apply_custom_theme()
    show_logo()
    
    if "results_df" not in st.session_state: st.session_state.results_df = None
    if "total_parts" not in st.session_state: st.session_state.total_parts = 0

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
