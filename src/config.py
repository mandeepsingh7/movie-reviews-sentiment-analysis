from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DATA_PATH = BASE_DIR / 'data' / 'imdb_test.csv'

SEED = 42 

# Model
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 2 
MAX_LENGTH = 256

# Training
TRAIN_BATCH_SIZE = 16 
EVAL_BATCH_SIZE = 16 

TRANSFORMERS_MODELS_DIR = BASE_DIR / "models" / "transformers"


# from pathlib import Path

# # Base paths
# BASE_DIR = Path(__file__).resolve().parent.parent.parent

# # Data
# DATASET_NAME = "imdb"

# # Model
# MODEL_NAME = "distilbert-base-uncased"
# NUM_LABELS = 2
# MAX_LENGTH = 256

# # Training
# OUTPUT_DIR = BASE_DIR / "models/transformers/distilbert"
# EPOCHS = 3
# LR = 2e-5
# TRAIN_BATCH_SIZE = 16
# EVAL_BATCH_SIZE = 16
# WEIGHT_DECAY = 0.01

# # Misc
# SEED = 42

