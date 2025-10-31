import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as snsgpus = tf.config.list_physical_devices('GPU')
if gpus:
    try: # allow memory growth
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print("GPU config error:", e)
# ==========================
# 1️⃣ Load and Preprocess Data
# ==========================
def load_dataset(directory):
    X, y = [], []
    classes = sorted(os.listdir(directory))
    for label in classes:
        class_dir = os.path.join(directory, label)
        if not os.path.isdir(class_dir):
            continue
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = load_img(img_path, target_size=(64, 64))
            img = img_to_array(img) / 255.0
            X.append(img)
            y.append(label)
    return np.array(X), np.array(y), classes


train_dir = "Brain_Tumor_data/Training"
X, y, class_names = load_dataset(train_dir)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded, num_classes=len(class_names))

print(f"[INFO] Loaded {len(X)} images across {len(class_names)} classes: {class_names}")



# ==========================
# 2️⃣ Define CNN Model (with your updated architecture)
# ==========================
def build_cnn(input_shape=(64,64,3), num_classes=4):
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Dense Layers
        Flatten(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# ==========================
# ⏱️ Custom Callback for Timing
# ==========================
class TimingCallback(Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        print(f"⏱️ Time for Epoch {epoch + 1}: {epoch_time:.2f} seconds")
        


# ==========================
# 3️⃣ K-Fold Cross Validation with Timing
# ==========================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
fold_accuracies = []
fold_times = []
epoch_time_records = []

os.makedirs("logs", exist_ok=True)

for train_index, val_index in kf.split(X):
    print(f"\n[INFO] Training Fold {fold_no}...")
    fold_start_time = time.time()

    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y_cat[train_index], y_cat[val_index]

    model = build_cnn((64, 64, 3), len(class_names))

    timing_callback = TimingCallback()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=16,
        verbose=1,
        callbacks=[timing_callback]
    )

    fold_time = time.time() - fold_start_time
    fold_times.append(fold_time)

    print(f"🕒 Total Time for Fold {fold_no}: {fold_time:.2f} seconds")
    
    # Store epoch times
    for epoch, t in enumerate(timing_callback.epoch_times, start=1):
        epoch_time_records.append([fold_no, epoch, t])
        
    acc = np.max(history.history['val_accuracy'])
    fold_accuracies.append(acc)
    model.save(f"classical_cnn_SGD_fold{fold_no}.keras")

    print(f"[INFO] Fold {fold_no} Accuracy: {acc:.4f}")
    fold_no += 1

mean_acc = np.mean(fold_accuracies)
mean_time = np.mean(fold_times)

print(f"\n✅ Mean K-Fold Validation Accuracy: {mean_acc:.4f}")
print(f"⏰ Average Time per Fold: {mean_time:.2f} seconds")

# ==========================
# 📊 Save Timing Data & Visualize
# ==========================
csv_path = "logs/epoch_training_times.csv"
with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Fold", "Epoch", "Time (seconds)"])
    writer.writerows(epoch_time_records)

print(f"[INFO] Training time data saved at: {csv_path}")

# Create visualization of epoch times
epoch_time_records = np.array(epoch_time_records)
folds = epoch_time_records[:, 0].astype(int)
epochs = epoch_time_records[:, 1].astype(int)
times = epoch_time_records[:, 2].astype(float)

plt.figure(figsize=(10, 6))
sns.lineplot(x=epochs, y=times, hue=folds, marker="o", palette="viridis")
plt.title("Training Time per Epoch across Folds", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Time (seconds)", fontsize=12)
plt.legend(title="Fold", loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("logs/epoch_time_plot.png")
plt.show()

# ==========================
# ✅ Final Stats
# ==========================
mean_acc = np.mean(fold_accuracies)
mean_time = np.mean(fold_times)

print(f"\n✅ Mean K-Fold Validation Accuracy: {mean_acc:.4f}")
print(f"⏰ Average Time per Fold: {mean_time:.2f} seconds")

model.save("classical_cnn_SGD_complete.keras")
print("[INFO] Final model saved successfully!")
# ==========================
# 4️⃣ Evaluate and Visualize (using test set)
# ==========================

model = load_model("classical_cnn_SGD_complete.keras")

TEST_DIR = "Brain_Tumor_data/Testing"
def load_test_dataset(directory, img_size=(64, 64)):
    X, y = [], []
    classes = sorted(os.listdir(directory))
    for label in classes:
        class_dir = os.path.join(directory, label)
        if not os.path.isdir(class_dir):
            continue
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = load_img(img_path, target_size=img_size)
            img = img_to_array(img) / 255.0
            X.append(img)
            y.append(label)
    return np.array(X), np.array(y)

X_test, y_test = load_test_dataset(TEST_DIR)
y_test_enc = encoder.transform(y_test)
y_test_cat = to_categorical(y_test_enc, num_classes=len(class_names))

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=1)
print(f"\n🎯 Test Accuracy: {test_acc:.4f}")

# Predictions
y_pred_probs = model.predict(X_test, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
print("\n📊 Classification Report:")
print(classification_report(y_test_enc, y_pred))
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test_enc, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test_enc, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('ML-CNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('mlcnn_confusion_matrix.png', dpi=300)
plt.show()

# ==========================
# 5️⃣ Save Results
# ==========================
from datetime import datetime
import json


results = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'model_config': {'img_size': (64, 64, 3), 'num_classes': len(class_names)},
    'kfold_results': {'fold_scores': [float(s) for s in fold_accuracies],
                      'mean_accuracy': float(np.mean(fold_accuracies)),
                      'std_accuracy': float(np.std(fold_accuracies))},
    'test_accuracy': float(test_acc)
}

with open('mlcnn_results.json', 'w') as f:
    json.dump(results, f, indent=4)
print("💾 Saved: mlcnn_results.json")

model.summary()