root_dir = "../data/plant_dataset_original/plant_diseases_images"

batch_size = 32
epochs = 6
seed = 255

focal_loss = {
    'alpha': 0.50,
    'gamma': 1.75
}

adamax_lr = 0.001
adamax_weight_decay = 0.001
