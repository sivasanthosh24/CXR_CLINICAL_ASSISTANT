from pathlib import Path
import torch

checkpoint_path = (
    Path("models")
    / "module1_disease"
    / "checkpoints"
    / "lateral_resnet50_weighted_best.pth"
)

print("=" * 70)
print("LATERAL CHECKPOINT INSPECTION")
print("=" * 70)

print("Checkpoint:", checkpoint_path.resolve())
print("Exists:", checkpoint_path.is_file())

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu"
)

print("\nCheckpoint type:")
print(type(checkpoint))

if isinstance(checkpoint, dict):
    print("\nTop-level checkpoint keys:")
    for key in checkpoint.keys():
        print(" -", key)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        selected_key = "model_state_dict"
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        selected_key = "state_dict"
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
        selected_key = "model"
    else:
        state_dict = checkpoint
        selected_key = "checkpoint itself"

    print("\nSelected state dictionary:", selected_key)
    print("Number of parameters:", len(state_dict))

    print("\nFirst 40 parameter keys:")
    for index, key in enumerate(state_dict.keys()):
        print(f"{index + 1:02d}. {key}")
        if index >= 39:
            break

    print("\nLast 10 parameter keys:")
    keys = list(state_dict.keys())
    for key in keys[-10:]:
        print(key)

else:
    print("Checkpoint is not a dictionary.")

print("\nInspection completed.")
