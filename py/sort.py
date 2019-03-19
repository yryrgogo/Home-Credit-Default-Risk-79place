import sys
import pandas as pd
import numpy as np

feim_path = sys.argv[1]
feim = pd.read_csv(feim_path)
feim['rank'] = np.arange(len(feim)) + 1

feim.to_csv(feim_path, index=False)
