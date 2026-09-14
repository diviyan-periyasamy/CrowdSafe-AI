from datetime import datetime

def generate_alerts(risk_category, person_count, density_label):
    alerts    = []
    timestamp = datetime.now().strftime("%H:%M:%S")

    if risk_category == "Critical":
        alerts.append(f"🚨 [{timestamp}] CRITICAL: Extreme overcrowding! ({person_count} people)")
        alerts.append(f"🚨 [{timestamp}] CRITICAL: Immediate action required!")
    elif risk_category == "High":
        alerts.append(f"⚠️ [{timestamp}] HIGH RISK: Dangerous crowd density ({person_count} people)")
    elif risk_category == "Moderate":
        alerts.append(f"🟡 [{timestamp}] MODERATE: Crowd growing ({person_count} people)")

    if density_label == "High" and risk_category not in ["Critical", "High"]:
        alerts.append(f"⚠️ [{timestamp}] WARNING: High density area detected")

    return alerts