import streamlit as st
import time

class PipelineTracker:
    def __init__(self, box, bar):
        self.box = box
        self.bar = bar
    def log(self, msg, pct):
        self.box.caption(f"[{time.strftime('%H:%M:%S')}] {msg}")
        self.bar.progress(pct)
