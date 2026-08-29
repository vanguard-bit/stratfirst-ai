import json
from pathlib import Path

import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]


st.set_page_config(page_title="NSE Trader", layout="wide")
st.title("NSE Paper Trading — Dashboard")

registry_path = ROOT / "data" / "fees" / "registry.json"
if registry_path.exists():
    reg = json.loads(registry_path.read_text())
    st.subheader("Fee registry")
    st.caption(f"Updated: {reg.get('updated_at')} | Broker: {reg.get('broker_profile')}")
    for seg_key, seg in reg.get("segments", {}).items():
        with st.expander(seg_key):
            st.dataframe(seg.get("components", []), use_container_width=True)
else:
    st.warning("No fee registry. Run: python main.py refresh-fees")

with (ROOT / "config" / "strategies.yaml").open() as fh:
    strategies = yaml.safe_load(fh)
st.subheader("Strategy universe")
st.write(f"{len(strategies.get('strategies', {}))} strategies across {len(strategies.get('clusters', {}))} clusters")
