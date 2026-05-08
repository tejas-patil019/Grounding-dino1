from groundingdino.util.inference import load_model
import torch
import sys

model = load_model(
    "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "weights/groundingdino_swint_ogc.pth"
)
model = model.to("cpu")

print("MODEL READY", flush=True)

# ⛔ BLOCK + LISTEN
while True:
    cmd = sys.stdin.readline().strip()
    if not cmd:
        continue

    if cmd == "ping":
        print("pong", flush=True)

    if cmd == "exit":
        print("shutting down", flush=True)
        break

