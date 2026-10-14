# Spectrogram inputs

The notebook expects pre-extracted NumPy arrays grouped by speaker. Its default directory is spelled `Spectograms`.

Example paths:

- `Spectograms/id10001/video_id/00001.npy`
- `Spectograms/id10001/video_id/00002.npy`
- `Spectograms/id10002/video_id/00001.npy`

The loader recursively discovers `.npy` files inside selected speaker folders. It selects the first 50% of sorted speaker directories (605 of 1,211 in the saved run) and assigns integer labels in that order.

Provide 2D numeric spectrogram arrays. The loader also reshapes 1D arrays to a truncated square, but that fallback does not substitute for documented audio preprocessing. Each array is normalized per sample and resized to 256 × 256 using bilinear interpolation.

The original audio preprocessing pipeline is not supplied. Before releasing it, document sample rate, window/hop sizes, FFT size, linear or mel scale, log scaling, clip duration, padding, and segmentation. New inputs must use the same extraction pipeline as training.

Data and arrays are excluded from Git. Obtain VoxCeleb from its official source and follow its access terms. For stronger evaluation, split by video/session before creating clips so related recordings do not cross partitions.
