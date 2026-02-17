import streamlit as st
import time

class PipelineTracker:
    def __init__(self, box, bar):
        self.box = box
        self.bar = bar
    def log(self, msg, pct):
        self.box.code(f"[{time.strftime('%H:%M:%S')}] {msg}", language="bash")
        self.bar.progress(pct, text=msg)
