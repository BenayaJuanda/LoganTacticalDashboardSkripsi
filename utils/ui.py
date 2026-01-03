import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
from io import BytesIO     

PRIMARY_BG = "#ffffff"
CARD_BG = "#CED2DC"
ACCENT = "#22c55e"
TEXT = "#000000"

def inject_css():
    st.markdown(f"""
    <style>
      .ltd-header {{ display:flex; align-items:center; gap:14px; padding:10px 6px 4px; }}
      .ltd-logo {{ width:36px; height:36px; object-fit:contain; }}
      .ltd-title {{ font-size:20px; font-weight:600; color:{TEXT}; margin:0; line-height:1.2; }}
      .ltd-sub {{ font-size:12px; color:#6b7280; margin:0; }}
      .ltd-card {{ background:{CARD_BG}; border:1px solid rgba(0,0,0,0.06); border-radius:12px; padding:14px; }}

      .kpi-grid {{ display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:14px; margin:6px 0 12px; }}
      .kpi-card {{ background:{CARD_BG}; border-radius:14px; padding:16px; border:1px solid rgba(0,0,0,0.06); box-shadow:0 2px 10px rgba(0,0,0,0.05); }}
      .kpi-title {{ color:#374151; font-size:12px; margin:0 0 6px; }}
      .kpi-value {{ color:{TEXT}; font-size:22px; font-weight:700; letter-spacing:.2px; }}

      .alert {{ border-radius:12px; padding:14px; border:1px solid rgba(0,0,0,0.06); margin:8px 0; }}
      .alert.info {{ background:#eef2ff; border-color:#dbeafe; }}
      .alert.success {{ background:#ecfdf5; border-color:#d1fae5; }}
      .alert.warn {{ background:#fffbeb; border-color:#fef3c7; }}
      .alert h4 {{ margin:0 0 6px; font-size:14px; color:{TEXT}; }}
      .alert ul {{ margin:0; padding-left:18px; color:#374151; }}

      .stApp {{ background:{PRIMARY_BG} !important; }}
    </style>
    """, unsafe_allow_html=True)

def _load_logo_inline(path: Path):
    try:
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    except:
        return None

def render_header(title="Logan Tactical Dashboard", subtitle="Sales Forecasting & Insights Platform", logo_path=None):
    inject_css()
    if logo_path is None:
        logo_path = Path(__file__).resolve().parent.parent / "image_source" / "Logo_logan_tactical.png"
    else:
        logo_path = Path(logo_path)
    logo_inline = _load_logo_inline(logo_path)
    logo_html = f'<img class="ltd-logo" src="{logo_inline}" />' if logo_inline else ""
    st.markdown(f"""
    <div class="ltd-header">
      {logo_html}
      <div>
        <p class="ltd-title">{title}</p>
        <p class="ltd-sub">{subtitle}</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

def sidebar_brand(logo_path="image_source/Logo_logan_tactical.png", version="v1.0.0"):
    p = Path(logo_path)
    if p.exists():
        st.sidebar.image(str(p), use_container_width=True)
    st.sidebar.markdown(f"**Logan Tactical Dashboard**  \n`{version}`")

def render_kpi_cards(items):
    inject_css()
    html_parts = ['<div class="kpi-grid">']
    for title, value, icon in items:
        html_parts.append(f"""
        <div class="kpi-card">
          <p class="kpi-title">{icon} {title}</p>
          <div class="kpi-value">{value}</div>
        </div>
        """)
    html_parts.append("</div>")
    st.markdown("\n".join(html_parts), unsafe_allow_html=True)

def insight_box(title, lines, tone="info"):
    inject_css()
    t = "info" if tone not in {"info","success","warn"} else tone
    items = "".join([f"<li>{l}</li>" for l in lines])
    st.markdown(f"""
    <div class="alert {t}">
      <h4>{title}</h4>
      <ul>{items}</ul>
    </div>
    """, unsafe_allow_html=True)

def export_chart_as_png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf