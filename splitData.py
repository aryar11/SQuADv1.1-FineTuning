from datasets import load_dataset
def load_and_split_data():
    squad = load_dataset("squad")

    print("One data sample:")
    import json
    example = squad['train'][1000]
    print(json.dumps(example, indent=2, ensure_ascii=False))

    # Load train/val dataset
    # train_data = squad['train'].select(range(5000))
    # test_ds = squad['validation'].select(range(1000))  # test set

    train_data = squad['train']
    test_ds = squad['validation']   # test set

    # Split training data into train and val
    train_ds, val_ds = train_test_split(train_data.to_list(), test_size=0.1, random_state=42)


    # Check context length - now not truncate
    context_lengths = [len(word_tokenize(example["context"])) for example in squad['train']]
    context_lengths_series = pd.Series(context_lengths)
    print("\nStats for training data:")
    print(context_lengths_series.describe(percentiles=[.5, .9, .95, .99]))

    return train_ds, val_ds, test_ds
