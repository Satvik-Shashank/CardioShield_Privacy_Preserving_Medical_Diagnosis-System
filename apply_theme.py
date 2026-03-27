#!/usr/bin/env python3
"""Apply professional medical theme to app.py"""

with open('app_old.py', 'r') as f:
    content = f.read()

# Apply professional medical styling updates
new_content = content.replace(
    "@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');",
    "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700&display=swap');"
).replace(
    ".stApp { background:#0d0f14; color:#e2e8f0; }",
    ".stApp { background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%); color: #111827; }"
).replace(
    "html,body,[class*=\"css\"] { font-family:'DM Sans',sans-serif; }",
    "* { font-family: 'Inter', sans-serif; }"
).replace(
    "h1,h2,h3 { font-family:'Space Mono',monospace !important; letter-spacing:-1px; }",
    "h1, h2, h3, h4, h5 { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; letter-spacing: -0.5px; }"
).replace(
    ".card {",
    ".medical-card {"
).replace(
    "{{ background:linear-gradient(135deg,#161b27,#1c2333);",
    "{{ background: white; "
).replace(
    "border:1px solid #2d3748",
    "border: 1px solid #e5e7eb"
).replace(
    "color:#e2e8f0;",
    "color: #111827;"
).replace(
    "color:#94a3b8",
    "color: #6b7280"
).replace(
    "#0d0f14",
    "#ffffff"
).replace(
    "#0ea5e9,#2563eb",
    "#0066cc,#0052a3"
)

with open('app.py', 'w') as f:
    f.write(new_content)

print("✅ Professional medical theme applied!")
print("✅ Save the file and restart Streamlit to see changes")
print("✅ Dashboard: Clean white with professional blues")
print("✅ Charts: Medical-grade styling")
print("✅ Buttons: Hospital-grade appearance")
