# Adding the trained checkpoint

The original model file is pending upload. The notebook expects `voxceleb_cnn_lite_gpu.pth` in the repository root unless `CONFIG['model_save_path']` is changed.

The saved checkpoint contains:

- `epoch` (zero-based; 44 corresponds to the recorded best epoch 45)
- `model_state_dict`
- `optimizer_state_dict`
- `val_acc` and `val_loss`
- `class_to_idx`
- `config`

The model has 34,448,893 parameters, so the checkpoint with optimizer state can be large. Use a GitHub Release asset for the model file, then add its download link and SHA-256 checksum here. This avoids storing large binary history in Git. Checkpoint extensions are ignored by `.gitignore`.

For inference, run the imports and `VGGStyleCNN` definition from the notebook, then load your own trusted checkpoint:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(
    "voxceleb_cnn_lite_gpu.pth",
    map_location=device,
    weights_only=True,
)
class_to_idx = checkpoint["class_to_idx"]
model = VGGStyleCNN(
    num_classes=len(class_to_idx),
    input_channels=1,
    dropout=checkpoint["config"]["dropout"],
).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

Use the checkpoint's exact `class_to_idx` mapping and the training spectrogram extraction pipeline. The notebook's `predict_sample` accepts an already-extracted `.npy` spectrogram. It does not accept raw audio.

The original notebook's inference demonstration also depends on dataset paths and label variables from earlier cells. Training must not be rerun just to load an existing checkpoint.
