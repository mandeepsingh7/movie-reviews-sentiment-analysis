from datasets import load_dataset
import pandas as pd 
import sys
from pathlib import Path 
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import TEST_DATA_PATH

def main():
    dataset = load_dataset('imdb', split='test')

    df = pd.DataFrame({
        'text': dataset['text'],
        'label': dataset['label']
    })

    TEST_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TEST_DATA_PATH, index=False)
    print('Test data downloaded and saved to', TEST_DATA_PATH)

if __name__ == '__main__':
    main()
