from ray import tune

### Hyperparameters for BINARY classifiers
search_space_resnext_binary = {
    "embedding": tune.choice([32, 64, 128, 256, 512]),
    "cardinality": tune.choice([4, 8, 16, 32]),
    "num_filters": tune.choice([64, 128, 256, 512]),
    # "width": tune.choice([4, 8, 16, 32]), Do not use width itself to better control the overall number of filters (width = num_filters / cardinality)
    "depth": tune.choice([1, 2, 3]),
    "mlp-head": tune.choice([0, 64, 256]),
    "lr": tune.choice([0.01, 0.005, 0.001, 0.0005, 0.0001]),
    "batch_size": tune.choice([64, 128, 256, 512])
}

search_space_convnext_binary = {
    "embedding": tune.choice([32, 64, 128, 256, 512]),
    "num_filters": tune.choice([64, 128, 256, 512]),
    "depth": tune.choice([1, 2, 3]),
    "mlp-head": tune.choice([0, 64, 256]),
    "lr": tune.choice([0.01, 0.005, 0.001, 0.0005, 0.0001]),
    "batch_size": tune.choice([64, 128, 256, 512])
}

search_space_transformer_binary = {
    "embedding": tune.choice([32, 64, 128, 256, 512]),
    "depth": tune.choice([1, 2, 3]),
    "num_heads": tune.choice([2, 4, 6, 8]),
    "dense_dim": tune.choice([32, 64, 128, 256, 512]),
    "mlp-head": tune.choice([0, 64, 256]),
    "lr": tune.choice([0.01, 0.005, 0.001, 0.0005, 0.0001]),
    "batch_size": tune.choice([64, 128, 256, 512])
}

search_space_rwkv_binary = {
    "embedding": tune.choice([32, 64, 128]),
    "depth": tune.choice([1, 2, 3, 4, 5]),
    "lr": tune.choice([0.001, 0.0005, 0.0001]),
    "batch_size": tune.choice([64, 128, 256])
}

### Hyperparameters for MULTICLASS classifiers
search_space_resnext_mcc = {
    "embedding": tune.choice([32, 64, 128, 256, 512]),
    "cardinality": tune.choice([4, 8, 16, 32]),
    "num_filters": tune.choice([64, 128, 256, 512]),
    "depth": tune.randint(1, 16),
    "mlp-head": tune.choice([0, 64, 128, 512, 1024]),
    "lr": tune.choice([0.01, 0.005, 0.001, 0.0005, 0.0001]),
    "batch_size": tune.choice([32, 64, 128, 256])
}

search_space_convnext_mcc = {
    "embedding": tune.choice([32, 64, 128, 256, 512]),
    "num_filters": tune.choice([64, 128, 256, 512]),
    "depth": tune.randint(1, 16),
    "mlp-head": tune.choice([0, 64, 128, 512, 1024]),
    "lr": tune.choice([0.01, 0.005, 0.001, 0.0005, 0.0001]),
    "batch_size": tune.choice([32, 64, 128, 256])
}

search_space_transformer_mcc = {
    "embedding": tune.choice([32, 64, 128, 256, 512]),
    "depth": tune.randint(1, 7),  # Depth in the range [2,6]
    "num_heads": tune.randint(2, 11),  # Heads is in the range [2,10]
    "dense_dim": tune.choice([32, 64, 128, 256, 512]),
    "mlp-head": tune.choice([0, 64, 128, 512, 1024]),
    "lr": tune.choice([0.005, 0.001, 0.0005, 0.0001]),
    "batch_size": tune.choice([32, 64, 128, 256])
}

search_space_rwkv_mcc = {
    "embedding": tune.choice([32, 64, 128, 256, 512]),
    "depth": tune.randint(1, 7),
    "batch_size": tune.choice([32, 64, 128, 256]),
    "lr": tune.choice([0.01, 0.005, 0.001, 0.0001]),
}
