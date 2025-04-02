# Hyperparameter Configuration
CFG = {
    "data_root": "/home/xyc/ML/data/AniPersonaCaps",  # Corrected to the AniPersonaCaps directory
    "metadata_path": "/home/xyc/ML/data/AniPersonaCaps/metadata.jsonl",
    "batch_size": 64,
    "lr": 3e-5,
    "epochs": 8,
    "image_size": 224,
    "text_max_len": 77,
    "num_workers": 8,
    "max_workers": 16,
    "train_folders": 24,  # The first 24 folders as the training set
    "val_folders": 6      # The last 6 folders as the validation set
}