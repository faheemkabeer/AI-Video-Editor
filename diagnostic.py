import os

print("--- Diagnostic for os module ---")
print(f"Type of os module: {type(os)}")
print(f"Has os.listdir: {hasattr(os, 'listdir')}")
if hasattr(os, 'listdir'):
    print(f"Type of os.listdir: {type(os.listdir)}")

print("\n--- Diagnostic for os.path module ---")
print(f"Type of os.path module: {type(os.path)}")
print(f"Has os.path.listdir: {hasattr(os.path, 'listdir')}")


import sys
if 'posixpath' in sys.modules:
    print(f"os.path is posixpath: {os.path is sys.modules['posixpath']}")
