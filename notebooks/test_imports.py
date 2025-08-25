import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, concatenate, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

print("All imports successful!")
print("Numpy working:", hasattr(np, 'nan'))
print("Pandas working:", hasattr(pd, 'DataFrame'))
