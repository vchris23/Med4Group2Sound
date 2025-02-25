import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import numpy as np
from sklearn.datasets import fetch_california_housing

cali = fetch_california_housing()
scaler = StandardScaler()
normalizer = Normalizer()
cali_data = scaler.fit_transform(cali.data)
cali_data = normalizer.fit_transform(cali_data)
print(cali_data)
cali_data_train, cali_data_test, cali_label_train, cali_label_test = train_test_split(cali_data, cali.target, test_size = 0.20, random_state=42)
