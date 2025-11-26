import torch

try:
    state_dict = torch.load(r"C:\Users\sinha\Documents\Research Paper Projects\Sentament Analysis\sentiment_model_weights.pth", map_location="cpu")
    print("Keys and shapes:")
    for key, value in state_dict.items():
        print(f"{key}: {value.shape}")
except Exception as e:
    print(e)
