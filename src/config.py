import numpy as np

TARGET_COL = "default"
POSITIVE_CLASS = 1
RANDOM_STATE = 42
TEST_SIZE = 0.2

THRESHOLDS = list(np.round(np.arange(0.10, 0.91, 0.05), 2))

FIGURES_DIR = "reports/figures"
