
LOAD_FROM_CACHE = True # Set to True to load pre-processed data and columns
OVERWRITE = False   # Set to True to overwrite cache files if LOAD_FROM_CACHE is False


ALPHA = 0
BETA = .9
START_SEASON = 2020
END_SEASON = 2024

MIN_PA = 300
CHUNK_SIZE=4

OUTPUT_DATA_FILE = f'predictions_{START_SEASON}_{END_SEASON}.csv' # File to save the final projection results
INPUT_DATA_FILE = '../data.csv'  # Path to the raw input data