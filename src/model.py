from sklearn.metrics import log_loss
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

def build_model():
    # Расчёт весов классов: здоровые (0) и больные (1)
    # neg, pos = np.bincount(y_train)
    # scale = neg / pos
    # print(f"[INFO] scale_pos_weight = {scale:.2f}")

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(eval_metric='logloss', random_state=42))
    ])
    return model
