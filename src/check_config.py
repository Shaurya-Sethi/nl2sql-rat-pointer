import os

print("CWD:", os.getcwd())
print("config.yaml exists (relative):", os.path.exists("config.yaml"))
print("config.yaml exists (absolute):", os.path.exists("/home/jupyter/Transqlate/src/config.yaml"))
with open("config.yaml") as f:
    print("First line:", f.readline())
