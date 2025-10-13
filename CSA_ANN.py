import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
df = pd.read_csv("C:/Users/Mitali/OneDrive/Desktop/Python WorkSpace/CSAT/CSAT_clean_data.csv")
# ===============================
# 🔢 STEP 4: ENCODE CATEGORICAL COLUMNS
# ===============================
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ===============================
# 🎯 STEP 5: SPLIT FEATURES & TARGET
# ===============================
X = df.drop('CSAT Score', axis=1)
y = df['CSAT Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# ⚖️ STEP 6: NORMALIZE FEATURES
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# 🧩 STEP 7: BUILD ANN MODEL
# ===============================
from tensorflow.keras.optimizers import Adamax
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='linear') # Output layer for regression
])

# Compile the model
model.compile(optimizer=Adamax(learning_rate=0.001), loss='mse', metrics=['mae'])

model.summary()

# ===============================
# 🚀 STEP 8: TRAIN MODEL
# ===============================
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=300,
    batch_size=32,
    verbose=1
)
# ===============================
# 📈 STEP 9: EVALUATE MODEL
# ===============================
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test MSE: {loss:.4f}")
print(f"✅ Test MAE: {mae:.4f}")
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()
#model.save("csat_ann_model.h5")
#print("✅ Model saved as csat_ann_model.h5")