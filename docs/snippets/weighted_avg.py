# docs/snippets/weighted_avg.py
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count # if the ammount of variables in each class is different, we need to adjust the weight
    samples_weight = np.array([weight[int(t)] for t in y_train])


    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # Dataloaders for python

    train_loader = DataLoader(ChurnDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False,drop_last=True)
    # instead of shuffle=True, we use the sampler to ensure balanced sampling, set to True if results are worse
    """drop_last=True fixes the Expected more than 1 value per channel when training, got input size torch.Size([1, 64])"""
    val_loader = DataLoader(ChurnDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(ChurnDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)