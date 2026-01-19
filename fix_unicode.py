#!/usr/bin/env python
"""Fix unicode symbols in main.py for Windows compatibility"""

with open('main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace unicode symbols with ASCII equivalents
replacements = {
    '✓': '[OK]',
    '✗': '[ERROR]',
    '⚠': '[WARNING]',
    '⏭': '[SKIP]'
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed unicode symbols in main.py")
