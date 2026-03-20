from datasets import load_dataset 

from src.config import SEED

def load_imdb_dataset(data_fraction=1.0):
    dataset = load_dataset('imdb')
    split_dataset = dataset['train'].train_test_split(
        test_size=0.2,
        seed=SEED 
    )
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    test_dataset = dataset['test']

    if data_fraction < 1.0:
        train_size = int(len(train_dataset) * data_fraction)
        val_size = int(len(val_dataset) * data_fraction)

        train_dataset = train_dataset.shuffle(seed=SEED).select(range(train_size))
        val_dataset = val_dataset.shuffle(seed=SEED).select(range(val_size))
    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
