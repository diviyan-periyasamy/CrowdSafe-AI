import sys, os
sys.path.append(os.path.dirname(__file__))

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import csv
import io
from datetime import datetime
from PIL import Image
from collections import deque

from detector      import (detect_people, get_density_label,
                            generate_heatmap, detect_flow_direction,
                            detect_stampede)
from risk_engine   import (calculate_grid_risk, calculate_risk_score,
                            GRID_COLS, GRID_ROWS)
from alert_engine  import generate_alerts

# Page config
st.set_page_config(
    page_title="CrowdSafe AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family:'Inter',sans-serif !important; }
[data-testid="stAppViewContainer"] { background:#f8fafc; }
[data-testid="stSidebar"]          { background:#0f172a; }
.block-container                   { padding:0 2rem 2rem !important; }
#MainMenu,footer,header            { visibility:hidden; }
[data-testid="stDecoration"]       { display:none; }
[data-testid="stSidebar"] *        { color:#e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label   { font-size:13px !important; color:#94a3b8 !important; }
[data-testid="stSidebar"] .stCheckbox label{ font-size:13px !important; color:#94a3b8 !important; }

.topbar {
    background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 50%,#0f172a 100%);
    padding:28px 36px; margin:-1rem -2rem 1.5rem;
    display:flex; align-items:center; justify-content:space-between;
    border-bottom:1px solid rgba(255,255,255,0.06);
}
.topbar-left  { display:flex; align-items:center; gap:18px; }
.topbar-icon  {
    width:52px;height:52px;
    background:linear-gradient(135deg,#3b82f6,#8b5cf6);
    border-radius:14px; display:flex; align-items:center;
    justify-content:center; font-size:24px;
    box-shadow:0 6px 20px rgba(59,130,246,0.35);
}
.topbar-title { font-size:24px;font-weight:800;color:#f8fafc;margin:0; }
.topbar-sub   { font-size:12px;color:#64748b;margin:3px 0 0; }
.live-badge {
    background:rgba(34,197,94,0.15); border:1px solid rgba(34,197,94,0.3);
    color:#4ade80; padding:5px 14px; border-radius:30px;
    font-size:12px; font-weight:600; display:flex; align-items:center; gap:6px;
}
.dot { width:6px;height:6px;background:#4ade80;border-radius:50%;
       animation:blink 1.2s infinite; }
@keyframes blink { 0%,100%{opacity:1}50%{opacity:0.3} }

.stat-chip { display:inline-block;padding:3px 10px;border-radius:20px;
             font-size:11px;font-weight:600;margin-top:6px; }
.chip-safe     { background:#dcfce7;color:#16a34a; }
.chip-moderate { background:#fef9c3;color:#ca8a04; }
.chip-high     { background:#fee2e2;color:#dc2626; }
.chip-critical { background:#fecaca;color:#b91c1c;
                 animation:pulse-c 1s infinite; }
@keyframes pulse-c {
    0%,100%{box-shadow:0 0 0 0 rgba(185,28,28,0.3)}
    50%{box-shadow:0 0 0 5px rgba(185,28,28,0)}
}
.alert-ok     { background:#f0fdf4;border-left:3px solid #22c55e;
                border-radius:6px;padding:9px 13px;color:#16a34a;
                font-size:12px;margin-top:6px; }
.alert-warn   { background:#fffbeb;border-left:3px solid #f59e0b;
                border-radius:6px;padding:9px 13px;color:#92400e;
                font-size:12px;margin-top:5px; }
.alert-danger { background:#fef2f2;border-left:3px solid #ef4444;
                border-radius:6px;padding:9px 13px;color:#b91c1c;
                font-size:12px;margin-top:5px; }

.stDownloadButton button {
    background:#0f172a !important; color:#f8fafc !important;
    border:none !important; border-radius:8px !important;
    font-weight:600 !important; width:100% !important;
    padding:9px 16px !important;
}
[data-testid="stMetric"] { display:none; }
[data-testid="stImage"] img { border-radius:10px; width:100%; }
</style>
""", unsafe_allow_html=True)

# Top bar
st.markdown("""
<div class="topbar">
  <div class="topbar-left">
    <div class="topbar-icon">🛡️</div>
    <div>
      <div class="topbar-title">CrowdSafe AI</div>
      <div class="topbar-sub">
        Intelligent Crowd Monitoring & Risk Assessment — YOLOv8 Deep Learning
      </div>
    </div>
  </div>
  <div class="live-badge"><div class="dot"></div>System Online</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:10px 0 18px'>
        <div style='font-size:26px'>🛡️</div>
        <div style='font-size:15px;font-weight:700;color:white'>CrowdSafe AI</div>
        <div style='font-size:10px;color:rgba(255,255,255,0.35);margin-top:2px'>
            v2.0 · Deep Learning Edition
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:10px;color:#475569;text-transform:uppercase;"
                "letter-spacing:1.4px;margin-bottom:8px'>📡 Input Source</div>",
                unsafe_allow_html=True)
    source = st.radio("", [
        "Upload Video",
        "Webcam",
        "Mobile Camera",
        "Upload Image"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div style='font-size:10px;color:#475569;text-transform:uppercase;"
                "letter-spacing:1.4px;margin-bottom:8px'>🔧 Features</div>",
                unsafe_allow_html=True)
    show_heatmap  = st.checkbox("🌡️  Crowd Heatmap",       value=True)
    show_flow     = st.checkbox("🌊  Flow Direction",       value=True)
    show_stampede = st.checkbox("🌪️  Stampede Detection",   value=True)
    show_chart    = st.checkbox("📈  Live Trend Chart",      value=True)

import os
_active_model = "best.pt (Fine-tuned)" if os.path.exists("models/best.pt") else "yolov8n.pt (Default)"

st.markdown(f"""
<div style='font-size:11px;color:#8b949e;line-height:2.2'>
🤖 <span style='color:#94a3b8'>Model:</span> {_active_model}<br>
⚙️ <span style='color:#94a3b8'>Mode:</span> CPU Inference<br>
📦 <span style='color:#94a3b8'>Library:</span> Ultralytics<br>
🗺️ <span style='color:#94a3b8'>Risk:</span> 4×3 Grid Zone System<br>
🌪️ <span style='color:#94a3b8'>Safety:</span> Stampede Detection<br>
🖥️ <span style='color:#94a3b8'>UI:</span> Streamlit
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
for k, v in [("alert_log",[]),
              ("count_history", deque(maxlen=60)),
              ("stats_log",[]),
              ("stampede_info", {})]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────
def chip(category):
    cls   = {"Safe":"chip-safe","Moderate":"chip-moderate",
             "High":"chip-high","Critical":"chip-critical"}.get(category,"chip-safe")
    icons = {"Safe":"✅","Moderate":"🟡","High":"🔴","Critical":"🚨"}
    return f'<span class="stat-chip {cls}">{icons.get(category,"")} {category}</span>'


def process_frame(frame):
    h, w = frame.shape[:2]

    annotated, count, boxes = detect_people(frame)
    density                 = get_density_label(count)

    zone_data = []
    if boxes:
        annotated, zone_data, category, _ = calculate_grid_risk(annotated, boxes)
        score, _, _ = calculate_risk_score(count, density, h*w)
    else:
        score, category, _ = calculate_risk_score(count, density, h*w)

    if show_heatmap and boxes:
        annotated = generate_heatmap(annotated, boxes)

    # Stampede detection
    stampede_info = {"level":"Normal","score":0,"speed":0,
                     "uniformity":0,"density":0,"direction":"—","message":""}
    if show_stampede:
        annotated, stampede_info = detect_stampede(annotated, boxes, count)
        st.session_state.stampede_info = stampede_info

    # Flow direction
    flow_label = ""
    if show_flow:
        annotated, flow_label = detect_flow_direction(annotated)

    # Logging
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.count_history.append(count)
    st.session_state.stats_log.append({
        "Time":ts,"Count":count,"Density":density,
        "Risk Score":score,"Risk Level":category,
        "Flow":flow_label,
        "Stampede Level":stampede_info["level"],
        "Stampede Score":stampede_info["score"]
    })

    # Alerts
    alerts = generate_alerts(category, count, density)
    lvl    = stampede_info["level"]
    if lvl == "STAMPEDE":
        alerts.append(f"🚨 [{ts}] STAMPEDE DETECTED! {stampede_info['message']}")
    elif lvl == "Danger":
        alerts.append(f"⚠️ [{ts}] DANGER: {stampede_info['message']}")
    elif lvl == "Warning":
        alerts.append(f"🟡 [{ts}] WARNING: {stampede_info['message']}")

    for a in alerts:
        if a not in st.session_state.alert_log:
            st.session_state.alert_log.append(a)

    return annotated, count, density, score, category, alerts, flow_label, zone_data, stampede_info


def render_stats(count, density, score, category, alerts,
                 flow_label, zone_data=[], stampede_info={}):

    # Overall summary 
    st.markdown(f"""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
                padding:14px 16px;margin-bottom:12px;">
      <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;
                  letter-spacing:1.2px;margin-bottom:10px;">📊 Overall Frame Summary</div>
      <div style="display:flex;gap:8px;">
        <div style="flex:1;text-align:center;background:white;border:1px solid #e2e8f0;
                    border-radius:10px;padding:10px 4px;">
          <div style="font-size:26px;font-weight:800;color:#0f172a">{count}</div>
          <div style="font-size:9px;color:#94a3b8;margin-top:2px">TOTAL PEOPLE<br>IN FRAME</div>
        </div>
        <div style="flex:1;text-align:center;background:white;border:1px solid #e2e8f0;
                    border-radius:10px;padding:10px 4px;">
          <div style="font-size:17px;font-weight:800;color:#0f172a;padding-top:5px">{density}</div>
          <div style="font-size:9px;color:#94a3b8;margin-top:2px">OVERALL<br>DENSITY</div>
        </div>
        <div style="flex:1;text-align:center;background:white;border:1px solid #e2e8f0;
                    border-radius:10px;padding:10px 4px;">
          <div style="font-size:26px;font-weight:800;color:#0f172a">
            {score}<span style="font-size:12px;color:#94a3b8">/100</span></div>
          <div style="font-size:9px;color:#94a3b8;margin-top:2px">RISK<br>SCORE</div>
          {chip(category)}
        </div>
      </div>
      <div style="font-size:10px;color:#94a3b8;margin-top:8px;text-align:center;
                  background:#f1f5f9;border-radius:6px;padding:4px;">
        ℹ️ Values represent the <b>entire video frame</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Zone grid
    if zone_data:
        st.markdown("""
        <div style="font-size:10px;color:#64748b;text-transform:uppercase;
                    letter-spacing:1.2px;margin-bottom:6px;
                    border-bottom:1px solid #e2e8f0;padding-bottom:5px;">
            🗺️ Zone-by-Zone Grid Breakdown
        </div>
        <div style="font-size:11px;color:#94a3b8;margin-bottom:8px;">
            Each cell = one zone — shows <b>people count + risk</b> for that area
        </div>
        """, unsafe_allow_html=True)

        risk_colors = {
            "Safe":     ("#dcfce7","#16a34a","✅"),
            "Moderate": ("#fef9c3","#ca8a04","🟡"),
            "High":     ("#fee2e2","#dc2626","🔴"),
            "Critical": ("#fecaca","#b91c1c","🚨"),
        }

        rows_dict = {}
        for z in zone_data:
            rows_dict.setdefault(z["row"], []).append(z)

        for row_idx in sorted(rows_dict.keys()):
            zones = sorted(rows_dict[row_idx], key=lambda z: z["col"])
            cols  = st.columns(len(zones))
            for zone, col in zip(zones, cols):
                bg, tc, icon = risk_colors.get(zone["risk"],("#f1f5f9","#475569","⬜"))
                with col:
                    st.markdown(f"""
                    <div style="background:{bg};border:1px solid {tc}44;border-radius:8px;
                                padding:7px 4px;text-align:center;margin-bottom:5px;">
                        <div style="font-size:19px;font-weight:800;color:{tc}">{zone['count']}</div>
                        <div style="font-size:9px;color:{tc};font-weight:600">people</div>
                        <div style="font-size:9px;color:{tc};margin-top:1px">{icon} {zone['risk']}</div>
                        <div style="font-size:8px;color:#94a3b8;margin-top:1px">
                            R{zone['row']+1}·C{zone['col']+1}</div>
                    </div>
                    """, unsafe_allow_html=True)

        safe_c = sum(1 for z in zone_data if z["risk"]=="Safe")
        mod_c  = sum(1 for z in zone_data if z["risk"]=="Moderate")
        hi_c   = sum(1 for z in zone_data if z["risk"]=="High")
        cr_c   = sum(1 for z in zone_data if z["risk"]=="Critical")

        st.markdown(f"""
        <div style="display:flex;gap:5px;margin:6px 0 12px">
          <div style="flex:1;background:#dcfce7;border-radius:8px;padding:7px 4px;text-align:center">
            <div style="font-size:17px;font-weight:700;color:#16a34a">{safe_c}</div>
            <div style="font-size:8px;color:#16a34a;font-weight:600">Safe<br>zones</div>
          </div>
          <div style="flex:1;background:#fef9c3;border-radius:8px;padding:7px 4px;text-align:center">
            <div style="font-size:17px;font-weight:700;color:#ca8a04">{mod_c}</div>
            <div style="font-size:8px;color:#ca8a04;font-weight:600">Moderate<br>zones</div>
          </div>
          <div style="flex:1;background:#fee2e2;border-radius:8px;padding:7px 4px;text-align:center">
            <div style="font-size:17px;font-weight:700;color:#dc2626">{hi_c}</div>
            <div style="font-size:8px;color:#dc2626;font-weight:600">High<br>zones</div>
          </div>
          <div style="flex:1;background:#fecaca;border-radius:8px;padding:7px 4px;text-align:center">
            <div style="font-size:17px;font-weight:700;color:#b91c1c">{cr_c}</div>
            <div style="font-size:8px;color:#b91c1c;font-weight:600">Critical<br>zones</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Stampede panel
    if show_stampede and stampede_info:
        lvl = stampede_info.get("level","Normal")
        clr = {"Normal":  ("#f0fdf4","#16a34a","#dcfce7"),
               "Warning": ("#fffbeb","#ca8a04","#fef9c3"),
               "Danger":  ("#fff7ed","#ea580c","#ffedd5"),
               "STAMPEDE":("#fef2f2","#b91c1c","#fecaca")}.get(lvl,("#f0fdf4","#16a34a","#dcfce7"))
        tc, bc, bg = clr

        st.markdown(f"""
        <div style="background:{tc};border:1px solid {bc}33;border-radius:12px;
                    padding:13px 15px;margin-bottom:12px;">
          <div style="font-size:10px;color:{bc};font-weight:700;text-transform:uppercase;
                      letter-spacing:1px;margin-bottom:8px;">🌪️ Stampede & Abnormal Movement</div>
          <div style="display:flex;gap:7px;margin-bottom:8px;">
            <div style="flex:1;background:{bg};border-radius:8px;padding:8px;text-align:center">
              <div style="font-size:17px;font-weight:800;color:{bc}">{lvl}</div>
              <div style="font-size:9px;color:{bc};margin-top:2px">RISK LEVEL</div>
            </div>
            <div style="flex:1;background:{bg};border-radius:8px;padding:8px;text-align:center">
              <div style="font-size:17px;font-weight:800;color:{bc}">
                {stampede_info.get('score',0)}<span style="font-size:10px">/100</span></div>
              <div style="font-size:9px;color:{bc};margin-top:2px">STAMPEDE SCORE</div>
            </div>
          </div>
          <div style="display:flex;gap:5px;">
            <div style="flex:1;background:white;border-radius:6px;padding:6px;
                        text-align:center;border:1px solid {bc}22">
              <div style="font-size:12px;font-weight:700;color:{bc}">
                {stampede_info.get('speed',0)}</div>
              <div style="font-size:8px;color:#94a3b8">Speed</div>
            </div>
            <div style="flex:1;background:white;border-radius:6px;padding:6px;
                        text-align:center;border:1px solid {bc}22">
              <div style="font-size:12px;font-weight:700;color:{bc}">
                {stampede_info.get('uniformity',0)}%</div>
              <div style="font-size:8px;color:#94a3b8">Uniformity</div>
            </div>
            <div style="flex:1;background:white;border-radius:6px;padding:6px;
                        text-align:center;border:1px solid {bc}22">
              <div style="font-size:12px;font-weight:700;color:{bc}">
                {stampede_info.get('direction','—')}</div>
              <div style="font-size:8px;color:#94a3b8">Direction</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Flow direction
    if show_flow and flow_label:
        st.markdown(f"""
        <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
                    padding:9px 13px;display:flex;align-items:center;gap:8px;
                    margin-bottom:10px;">
          <span style="font-size:18px">🌊</span>
          <div>
            <div style="font-size:9px;color:#1d4ed8;font-weight:700;
                        text-transform:uppercase;letter-spacing:0.8px">Crowd Movement</div>
            <div style="font-size:13px;color:#1d4ed8;font-weight:700">{flow_label}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Alert status
    st.markdown("""
    <div style="font-size:10px;color:#64748b;text-transform:uppercase;
                letter-spacing:1.2px;margin-bottom:6px;
                border-bottom:1px solid #e2e8f0;padding-bottom:5px;">
        🚨 Alert Status
    </div>
    """, unsafe_allow_html=True)

    if alerts:
        for a in alerts:
            cls = "alert-danger" if "CRITICAL" in a or "STAMPEDE" in a else "alert-warn"
            st.markdown(f'<div class="{cls}">{a}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-ok">✅ All clear — situation normal</div>',
                    unsafe_allow_html=True)


def render_export():
    st.markdown("""
    <div style="font-size:10px;color:#64748b;text-transform:uppercase;
                letter-spacing:1.2px;margin:14px 0 8px;
                border-bottom:1px solid #e2e8f0;padding-bottom:5px;">
        📤 Export Reports
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.session_state.stats_log:
            buf = io.StringIO()
            w   = csv.DictWriter(buf, fieldnames=st.session_state.stats_log[0].keys())
            w.writeheader()
            w.writerows(st.session_state.stats_log)
            st.download_button("⬇️ CSV Report", buf.getvalue(),
                file_name=f"crowdsafe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv")
        else:
            st.caption("No data yet")
    with c2:
        if st.session_state.alert_log:
            st.download_button("⬇️ Alert Log",
                "\n".join(st.session_state.alert_log),
                file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain")
        else:
            st.caption("No alerts yet")

    if st.session_state.alert_log:
        st.markdown("""
        <div style="font-size:10px;color:#64748b;text-transform:uppercase;
                    letter-spacing:1.2px;margin:12px 0 6px;
                    border-bottom:1px solid #e2e8f0;padding-bottom:5px;">
            🔔 Alert History
        </div>
        """, unsafe_allow_html=True)
        for a in reversed(st.session_state.alert_log[-10:]):
            cls = "alert-danger" if "CRITICAL" in a or "STAMPEDE" in a else "alert-warn"
            st.markdown(f'<div class="{cls}">{a}</div>', unsafe_allow_html=True)


def run_feed(cap):
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""<div style="font-size:10px;color:#64748b;text-transform:uppercase;
                    letter-spacing:1.2px;margin-bottom:6px;">📡 Live Feed</div>""",
                    unsafe_allow_html=True)
        frame_ph = st.empty()

    with col2:
        stats_ph = st.empty()

    chart_ph = st.empty()
    st.sidebar.markdown("---")
    stop = st.sidebar.button("⏹️ Stop Feed", type="primary", use_container_width=True)

    while not stop:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Feed lost.")
            break

        (annotated, count, density, score, category,
         alerts, flow, zone_data, stampede_info) = process_frame(frame)

        frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                       channels="RGB", use_container_width=True)

        with stats_ph.container():
            render_stats(count, density, score, category,
                         alerts, flow, zone_data, stampede_info)

        if show_chart and len(st.session_state.count_history) > 1:
            with chart_ph.container():
                st.markdown("""<div style="font-size:10px;color:#64748b;
                            text-transform:uppercase;letter-spacing:1.2px;
                            margin-bottom:4px;">📈 People Count Over Time</div>""",
                            unsafe_allow_html=True)
                st.line_chart(list(st.session_state.count_history),
                              use_container_width=True)
        time.sleep(0.05)

    cap.release()
    render_export()



# INPUT MODES

col1, col2 = st.columns([3, 2])

if source == "Upload Video":
    uploaded = st.sidebar.file_uploader("Choose video", type=["mp4","avi","mov"])
    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded.read())
        run_feed(cv2.VideoCapture(tmp.name))

elif source == "Webcam":
    if st.sidebar.button("▶️ Start Webcam", use_container_width=True):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Webcam not found.")
        else:
            run_feed(cap)

elif source == "Mobile Camera":
    method = st.sidebar.radio("Method", ["DroidCam Client","Direct IP"])
    if method == "DroidCam Client":
        idx = st.sidebar.selectbox("Camera Index", [1,2,0,3])
        if st.sidebar.button("▶️ Start", use_container_width=True):
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                st.error("❌ DroidCam not found.")
            else:
                run_feed(cap)
    else:
        ip   = st.sidebar.text_input("Phone IP", placeholder="192.168.x.x")
        port = st.sidebar.text_input("Port", value="4747")
        if st.sidebar.button("▶️ Connect", use_container_width=True):
            if not ip:
                st.warning("Enter IP address.")
            else:
                cap = cv2.VideoCapture(f"http://{ip}:{port}/video")
                if not cap.isOpened():
                    st.error("❌ Cannot connect.")
                else:
                    st.success(f"✅ Connected to {ip}:{port}")
                    run_feed(cap)

elif source == "Upload Image":
    img_file = st.sidebar.file_uploader("Choose image", type=["jpg","jpeg","png"])
    if img_file:
        frame = cv2.cvtColor(np.array(Image.open(img_file)), cv2.COLOR_RGB2BGR)
        (annotated, count, density, score, category,
         alerts, flow, zone_data, stampede_info) = process_frame(frame)

        with col1:
            st.markdown("""<div style="font-size:10px;color:#64748b;
                        text-transform:uppercase;letter-spacing:1.2px;
                        margin-bottom:6px;">🖼️ Analyzed Image</div>""",
                        unsafe_allow_html=True)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     use_container_width=True)
        with col2:
            render_stats(count, density, score, category,
                         alerts, flow, zone_data, stampede_info)
        render_export()