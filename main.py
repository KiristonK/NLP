import os
import nltk
import numpy as np
import pandas as pd
import pandas as py

from data import DataWorker

dw = DataWorker('complaints_processed.csv')

print(dw.get_words_bag())
