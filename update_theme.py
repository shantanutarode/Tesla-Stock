import re

# Read the file
with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace chart themes
content = content.replace("template='plotly_dark'", "template='plotly_white'")
content = content.replace("paper_bgcolor='rgba(30, 41, 59, 0.8)'", "paper_bgcolor='white'")
content = content.replace("plot_bgcolor='rgba(30, 41, 59, 0.5)'", "plot_bgcolor='#f8f9fa'")
content = content.replace("font=dict(color='white')", "font=dict(color='#1a1a1a')")

# Write back
with open('streamlit_app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Chart themes updated to light theme!")
