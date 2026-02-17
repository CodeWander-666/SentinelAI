import streamlit as st
import time

class PipelineTracker:
    """
    Manages Real-Time Dashboard Feedback.
    """
    def __init__(self, log_container, progress_bar):
        self.log_container = log_container
        self.progress_bar = progress_bar
        self.logs = []

    def log(self, message, percent):
        """Updates the visual log and progress bar."""
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.logs.append(entry)
        
        # Show last 8 lines for cleaner UI
        self.log_container.code("\n".join(self.logs[-8:]), language="bash")
        self.progress_bar.progress(percent, text=message)
