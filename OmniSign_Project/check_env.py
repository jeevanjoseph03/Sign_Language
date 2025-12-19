# check_env.py
# Simple import test for common dependencies used by OmniSign_Project
import sys

packages = [
    ('numpy', 'numpy'),
    ('cv2', 'cv2'),
    ('mediapipe', 'mediapipe'),
    ('matplotlib', 'matplotlib')
]

for name, mod in packages:
    try:
        m = __import__(mod)
        v = getattr(m, '__version__', 'unknown')
        print(f"OK: {name} imported (version={v})")
    except Exception as e:
        print(f"ERROR: failed to import {name}: {e}")

print('Python executable:', sys.executable)
print('Python version:', sys.version.splitlines()[0])
