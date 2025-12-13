import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try: # allow memory growth
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print("GPU config error:", e)
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import pennylane as qml
import os
import time
import cv2
import csv

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# -------------------------
# Configuration
# -------------------------
IMG_SIZE = 64
BATCH_SIZE = 16
N_QUBITS = 4
N_LAYERS = 2
EPOCHS = 30
K_FOLDS = 5
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Quantum/noise/shots settings
SHOTS = 1000                # Number of circuit samples per forward pass (lower -> more statistical noise)
NOISE_PROB = 0.02           # Depolarizing noise probability (0.0 = no noise, 0.02 = 2% depolarizing)
USE_NOISE = True            # Toggle to enable/disable noise injection


print("="*70)
print("QUANTUM-INSPIRED CNN FOR BRAIN TUMOR CLASSIFICATION (UPDATED)")
print("="*70)
print(f"\nConfiguration:")
print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Quantum Qubits: {N_QUBITS}")
print(f"  Quantum Layers: {N_LAYERS}")
print(f"  K-Folds: {K_FOLDS}")
print(f"  Classes: {CLASS_NAMES}")
print(f"  Shots: {SHOTS}  (fewer shots => higher sampling noise)")
print(f"  Depolarizing Noise Enabled: {USE_NOISE}, Noise prob: {NOISE_PROB}")
print("="*70)

@register_keras_serializable(package="custom", name="AdamSGDHybrid")
class AdamSGDHybrid(tf.keras.optimizers.Optimizer):

    def __init__(
        self,
        learning_rate=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        momentum=0.9,
        epsilon=1e-7,
        switch_epoch=10,
        name="AdamSGDHybrid",
        **kwargs
    ):
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.momentum = momentum
        self.epsilon = epsilon
        self.switch_epoch = switch_epoch

        self._current_epoch = tf.Variable(
            0, trainable=False, dtype=tf.int32
        )

        self._m = {}
        self._v = {}
        self._velocity = {}

    def increment_epoch(self):
        self._current_epoch.assign_add(1)

    def build(self, var_list):
        for var in var_list:
            key = var.name
            self._m[key] = self.add_variable_from_reference(var, "m")
            self._v[key] = self.add_variable_from_reference(var, "v")
            self._velocity[key] = self.add_variable_from_reference(var, "velocity")

    def update_step(self, grad, var):
        lr = self.learning_rate
        key = var.name

        m = self._m[key]
        v = self._v[key]
        velocity = self._velocity[key]

        if self._current_epoch < self.switch_epoch:
            # Adam phase
            m.assign(self.beta_1 * m + (1.0 - self.beta_1) * grad)
            v.assign(self.beta_2 * v + (1.0 - self.beta_2) * tf.square(grad))

            m_hat = m / (1.0 - self.beta_1)
            v_hat = v / (1.0 - self.beta_2)

            var.assign_sub(lr * m_hat / (tf.sqrt(v_hat) + self.epsilon))
        else:
            # SGD + momentum phase
            velocity.assign(self.momentum * velocity + grad)
            var.assign_sub(lr * velocity)

    def get_config(self):
        config = super().get_config()
        config.update({
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "momentum": self.momentum,
            "epsilon": self.epsilon,
            "switch_epoch": self.switch_epoch,
        })
        return config
@register_keras_serializable(package="custom", name="OptimizerEpochTracker")
class OptimizerEpochTracker(Callback):

    def on_epoch_end(self, epoch, logs=None):
        opt = self.model.optimizer
        if hasattr(opt, "increment_epoch"):
            opt.increment_epoch()
    def get_config(self):
        return {}


# -------------------------
# Quantum circuit (with noise)
# -------------------------
def quantum_circuit(inputs, weights):
    """Quantum circuit with data encoding, variational layers and optional depolarizing noise."""
    # Encode classical data into quantum states
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)

    # Optionally add depolarizing noise after encoding (simulates polarization/depolarizing noise)
    if USE_NOISE and NOISE_PROB > 0:
        for i in range(N_QUBITS):
            qml.DepolarizingChannel(NOISE_PROB, wires=i)

    # Variational quantum layers with entanglement
    for layer in range(N_LAYERS):
        # parameterized single-qubit rotations
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)

        # entangling CNOT chain
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])

        # Optionally add depolarizing noise between layers (models decoherence during circuit)
        if USE_NOISE and NOISE_PROB > 0:
            for i in range(N_QUBITS):
                qml.DepolarizingChannel(NOISE_PROB, wires=i)

    # Return expectation values (with finite shots these are estimated)
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
try:
    dev = qml.device("default.qubit", wires=N_QUBITS)
    qnode = qml.QNode(quantum_circuit, dev, interface='autograd')
    qnode = qml.transforms.set_shots(SHOTS)(qnode)
except Exception:
    # fallback for older versions
    dev = qml.device("default.qubit", wires=N_QUBITS, shots=SHOTS)
    qnode = qml.QNode(quantum_circuit, dev, interface='autograd')

# -------------------------
# Quantum-inspired Keras layer (classical differentiable approximation that can call qnode)
# -------------------------
@register_keras_serializable(package="custom", name="QuantumInspiredLayer")
class QuantumInspiredLayer(layers.Layer):
    def __init__(self, n_qubits, n_layers, **kwargs):
        super(QuantumInspiredLayer, self).__init__(**kwargs)
        self.n_qubits = n_qubits
        self.n_layers = n_layers

    def build(self, input_shape):
        self.rotation_weights = self.add_weight(
            name="rotation_weights",
            shape=(self.n_layers, self.n_qubits, 2),
            initializer=keras.initializers.RandomUniform(0, 2*np.pi),
            trainable=True
        )
        self.interaction_weights = self.add_weight(
            name="interaction_weights",
            shape=(self.n_layers, self.n_qubits, self.n_qubits),
            initializer=keras.initializers.GlorotUniform(),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        """
        Primary forward pass — classical differentiable transformation inspired by quantum ops.
        Optionally demonstrates true quantum circuit outputs via qnode for small batches.
        """
        x = inputs
        for layer in range(self.n_layers):
            ry_transform = tf.cos(self.rotation_weights[layer, :, 0]) * x + \
                          tf.sin(self.rotation_weights[layer, :, 0]) * tf.roll(x, shift=1, axis=-1)

            rz_transform = tf.cos(self.rotation_weights[layer, :, 1]) * ry_transform - \
                          tf.sin(self.rotation_weights[layer, :, 1]) * tf.roll(ry_transform, shift=-1, axis=-1)

            interaction = tf.matmul(
                tf.expand_dims(rz_transform, axis=-1),
                tf.expand_dims(rz_transform, axis=-2)
            )

            interaction_effect = tf.reduce_sum(
                interaction * self.interaction_weights[layer],
                axis=-1
            )

            x = tf.tanh(rz_transform + 0.1 * interaction_effect)

        # x shape may be (batch, n_qubits) — keep as-is for downstream dense layer
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"n_qubits": self.n_qubits, "n_layers": self.n_layers})
        return config

    def demonstrate_quantum_circuit(self, sample_input):
        if not tf.executing_eagerly():
            raise RuntimeError("Quantum demonstration requires eager mode")
        weights_np = tf.keras.backend.get_value(self.rotation_weights)
        return np.array(qnode(sample_input, weights_np))


# -------------------------
# Model creation (uses QuantumInspiredLayer)
# -------------------------
def create_quantum_cnn_model():
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        # Classical conv blocks
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),

        # Project to N_QUBITS for the quantum-inspired layer input
        layers.Dense(N_QUBITS, activation='tanh'),

        # Use the defined QuantumInspiredLayer
        QuantumInspiredLayer(N_QUBITS, N_LAYERS, name='quantum_layer'),

        # Output
        layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])
    return model

# -------------------------
# Data loading
# -------------------------
def load_data(data_dir, img_size=IMG_SIZE):
    images = []
    labels = []
    for idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found, skipping...")
            continue
        print(f"Loading {class_name}...", end=" ")
        count = 0
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (img_size, img_size))
                    img = img / 255.0
                    images.append(img)
                    labels.append(idx)
                    count += 1
            except Exception:
                continue
        print(f"{count} images loaded")
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

def visualize_quantum_layer(model, test_images, save_path='quantum_layer_viz.png'):
    try:
        quantum_model = keras.Model(inputs=model.input, outputs=model.get_layer('quantum_layer').output)
    except Exception:
        print("Quantum layer not found for visualization")
        return
    n_samples = min(4, len(test_images))
    indices = np.random.choice(len(test_images), n_samples, replace=False)
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, 2)
    for idx, img_idx in enumerate(indices):
        img = test_images[img_idx:img_idx+1]
        quantum_output = quantum_model.predict(img, verbose=0)[0]
        axes[idx, 0].imshow(img[0])
        axes[idx, 0].set_title('Input Image', fontsize=10)
        axes[idx, 0].axis('off')
        axes[idx, 1].bar(range(N_QUBITS), quantum_output)
        axes[idx, 1].set_title('Quantum Layer Activations', fontsize=10)
        axes[idx, 1].set_xlabel('Qubit')
        axes[idx, 1].set_ylabel('Activation')
        axes[idx, 1].set_ylim([-1, 1])
        axes[idx, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Quantum layer visualization saved to {save_path}")
    plt.close()
    
def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()
    
def plot_training_history(histories, save_path='training_history.png'):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for fold_idx, history in enumerate(histories):
        axes[0].plot(history['accuracy'], label=f'Fold {fold_idx+1} Train', alpha=0.6)
        axes[0].plot(history['val_accuracy'], label=f'Fold {fold_idx+1} Val', alpha=0.6, linestyle='--')
    axes[0].set_title('Model Accuracy Across Folds')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    for fold_idx, history in enumerate(histories):
        axes[1].plot(history['loss'], label=f'Fold {fold_idx+1} Train', alpha=0.6)
        axes[1].plot(history['val_loss'], label=f'Fold {fold_idx+1} Val', alpha=0.6, linestyle='--')
    axes[1].set_title('Model Loss Across Folds')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history saved to {save_path}")
    plt.close()
    
    
@register_keras_serializable(package="custom", name="TimingCallback")
class TimingCallback(Callback):
    def __init__(self, fold_no, epoch_time_records):
        super().__init__()
        self.fold_no = fold_no
        self.epoch_times = []
        self.epoch_time_records = epoch_time_records

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        t = time.time() - self._epoch_start
        self.epoch_times.append(t)
        # safely extract learning rate (Keras 3 uses .learning_rate)
        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        except:
            lr = None
        # record: fold, epoch (1-based), time, train_acc, val_acc, train_loss, val_loss, lr
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        self.epoch_time_records.append([self.fold_no, epoch+1, t, train_acc, val_acc, train_loss, val_loss, lr])
        print(f"⏱️ [Fold {self.fold_no}] Epoch {epoch+1} time: {t:.2f}s | train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        
    def get_config(self):
        return {
        "fold_no": self.fold_no,
        # epoch_time_records is mutable and not serializable, so skip it
        }

# -------------------------
# Main training/evaluation pipeline
# -------------------------
from tensorflow.keras.utils import to_categorical

def main():
    train_dir = '/mnt/c/Users/hassa/OneDrive/Desktop/QML/Brain_Tumor_data/Training'
    test_dir = '/mnt/c/Users/hassa/OneDrive/Desktop/QML/Brain_Tumor_data/Testing'

    if not os.path.exists(train_dir):
        print(f"\nError: Training directory '{train_dir}' not found!")
        return

    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    X_train, y_train = load_data(train_dir)
    y_train = to_categorical(y_train, num_classes=len(CLASS_NAMES))

    print(f"\nTraining data: {X_train.shape}, Labels: {y_train.shape}")

    X_test, y_test = None, None
    if os.path.exists(test_dir):
        X_test, y_test = load_data(test_dir)
        if y_test is not None:
            y_test = to_categorical(y_test, num_classes=len(CLASS_NAMES))


        print(f"Testing data: {X_test.shape}, Labels: {y_test.shape}")
    else:
        print(f"\nWarning: Test directory '{test_dir}' not found. Skipping test evaluation.")

    # K-Fold Cross Validation
    print("\n" + "="*70)
    print("K-FOLD CROSS VALIDATION TRAINING")
    print("="*70)

    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_scores = []
    histories = []
    best_model = None
    best_val_acc = 0.0
    epoch_time_records = []  # collect [fold, epoch, time, train_acc, val_acc, train_loss, val_loss, lr]
    os.makedirs("logs", exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        fold_no = fold + 1
        print(f"\n{'='*70}")
        print(f"FOLD {fold_no}/{K_FOLDS}")
        print('='*70)

        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        print(f"Train samples: {len(X_fold_train)}, Val samples: {len(X_fold_val)}")

        model = create_quantum_cnn_model()
        
        hybrid_optimizer = AdamSGDHybrid(
            learning_rate=1e-3,
            switch_epoch=10
        )
        # Compile with custom optimizer
        model.compile(
            optimizer=hybrid_optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        ]


        timing_cb = TimingCallback(fold_no, epoch_time_records)
        callbacks.append(timing_cb)

        fold_start_time = time.time()
        try:
            history = model.fit(
                X_fold_train, y_fold_train,
                validation_data=(X_fold_val, y_fold_val),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=callbacks,
                verbose=1
            )
            fold_training_time = time.time() - fold_start_time

            val_loss, val_acc = model.evaluate(X_fold_val, y_fold_val, verbose=0)
            fold_scores.append(val_acc)
            histories.append(history.history)

            print(f"\nFold {fold_no} Results:")
            print(f"  Validation Accuracy: {val_acc:.4f}")
            print(f"  Validation Loss: {val_loss:.4f}")
            print(f"  Fold Training Time: {fold_training_time:.2f} seconds")

            # Save model for this fold
            model.save(f"quantum_cnn_fold{fold_no}.keras")
            print(f"[INFO] Saved fold model: quantum_cnn_fold{fold_no}.keras")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model
                print(f"  *** New best model! ***")
            tf.keras.backend.clear_session()
        except Exception as e:
            print(f"\nError in fold {fold_no}: {str(e)}")
            continue

    if len(fold_scores) == 0:
        print("\nNo folds completed successfully.")
        return

    # Save epoch timing CSV
    csv_path = "logs/qcnn_epoch_times.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Fold", "Epoch", "TimeSec", "TrainAcc", "ValAcc", "TrainLoss", "ValLoss", "LR"])
        writer.writerows(epoch_time_records)
    print(f"[INFO] Epoch timing data saved to: {csv_path}")

    # Plot epoch times
    if len(epoch_time_records) > 0:
        arr = np.array(epoch_time_records)
        folds_plot = arr[:,0].astype(int)
        epochs_plot = arr[:,1].astype(int)
        times_plot = arr[:,2].astype(float)

        plt.figure(figsize=(10,6))
        sns.lineplot(x=epochs_plot, y=times_plot, hue=folds_plot, marker="o")
        plt.title("Training Time per Epoch across Folds")
        plt.xlabel("Epoch")
        plt.ylabel("Time (seconds)")
        plt.legend(title="Fold", loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = "logs/qcnn_epoch_time_plot.png"
        plt.savefig(plot_path)
        plt.show()
        plt.close()
        print(f"[INFO] Epoch time plot saved to: {plot_path}")

    # CV summary
    print("\n" + "="*70)
    print("CROSS-VALIDATION RESULTS")
    print("="*70)
    print(f"Fold Accuracies: {[f'{acc:.4f}' for acc in fold_scores]}")
    print(f"Mean CV Accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
    print(f"Best Fold Accuracy: {max(fold_scores):.4f}")

    # Plot training history
    if len(histories) > 0:
        plot_training_history(histories)

    # Test evaluation using best_model
    if X_test is not None and y_test is not None and best_model is not None:
        print("\n" + "="*70)
        print("TEST SET EVALUATION")
        print("="*70)
        test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        y_pred = best_model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        # Classification report & metrics
        print("\n" + "-"*70)
        print("CLASSIFICATION REPORT")
        print("-"*70)
        print(classification_report(y_true_classes, y_pred_classes, target_names=CLASS_NAMES))
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_classes, y_pred_classes, average=None, labels=range(len(CLASS_NAMES))
        )
        print("\n" + "-"*70)
        print("PER-CLASS PERFORMANCE ANALYSIS")
        print("-"*70)
        for i, class_name in enumerate(CLASS_NAMES):
            print(f"\n{class_name.upper()}:")
            print(f"  Precision: {precision[i]:.4f}")
            print(f"  Recall:    {recall[i]:.4f}")
            print(f"  F1-Score:  {f1[i]:.4f}")
            print(f"  Support:   {support[i]}")

        # Visualizations
        plot_confusion_matrix(y_true_classes, y_pred_classes)
        # visualize_gradcam(best_model, X_test, y_test)
        visualize_quantum_layer(best_model, X_test)

    # Save best model
    if best_model is not None:
        best_model.save('quantum_cnn_brain_tumor_model.keras')
        print("\n" + "="*70)
        print("Model saved as 'quantum_cnn_brain_tumor_model.keras'")
        print("="*70)

    print("\n✓ Training and evaluation complete!")
    print("\nGenerated files:")
    print("  1. quantum_cnn_brain_tumor_model.keras - Trained best model")
    print("  2. quantum_cnn_fold{n}.keras - Per-fold saved models")
    print("  3. logs/qcnn_epoch_times.csv - Epoch timing & metrics")
    print("  4. logs/qcnn_epoch_time_plot.png - Epoch time plot")
    print("  5. training_history.png - Training curves")
    print("  6. confusion_matrix.png - Confusion matrix")
    print("  7. gradcam_results.png - Grad-CAM visualizations")
    print("  8. quantum_layer_viz.png - Quantum layer activations")
    
if __name__ == "__main__":
    main()
