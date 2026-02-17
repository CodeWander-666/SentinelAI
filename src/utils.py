import streamlit as st
import time

class PipelineTracker:
    """
    Real-time feedback engine. 
    Updates the dashboard log and progress bar instantly.
    """
    def __init__(self, container, progress_bar):
        self.container = container
        self.progress_bar = progress_bar
        self.logs = []
        self.step = 0

    def log(self, message, progress_percent):
        """
        Updates the UI with a new log message and progress state.
        """
        # Add timestamp-like prefix
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        
        self.logs.append(entry)
        # Keep only the last 10 lines to keep the UI clean
        display_logs = "\n".join(self.logs[-10:])
        
        # Update UI components
        self.container.code(display_logs, language="bash")
        self.progress_bar.progress(progress_percent, text=message)
