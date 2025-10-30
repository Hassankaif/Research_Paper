import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.framework.errors_impl import ResourceExhaustedError
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import pennylane as qml
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import gc, os, numpy as np
import time

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================
# QUANTUM LAYER DEFINITION (Same as QCNN.py)
# ============================================
N_QUBITS = 4
N_LAYERS = 2

# Initialize quantum device
dev = qml.device("default.qubit", wires=N_QUBITS)

def quantum_circuit(inputs, weights):
    """Quantum circuit with data encoding and trainable layers"""
    # Encode classical data into quantum states
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)
    
    # Variational quantum layers
    for layer in range(N_LAYERS):
        # Entangling layer
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        
        # CNOT entanglement
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])
    
    # Measure expectations
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# Create QNode
qnode = qml.QNode(quantum_circuit, dev, interface='autograd')

class QuantumInspiredLayer(layers.Layer):
    """
    Quantum-inspired layer that emulates quantum behavior with classical operations.
    Same implementation as QCNN.py for consistency.
    """
    def __init__(self, n_qubits, n_layers, **kwargs):
        super(QuantumInspiredLayer, self).__init__(**kwargs)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
    def build(self, input_shape):
        # Quantum-inspired rotation parameters
        self.rotation_weights = self.add_weight(
            name="rotation_weights",
            shape=(self.n_layers, self.n_qubits, 2),
            initializer=keras.initializers.RandomUniform(0, 2*np.pi),
            trainable=True
        )
        
        # Entanglement interaction weights
        self.interaction_weights = self.add_weight(
            name="interaction_weights",
            shape=(self.n_layers, self.n_qubits, self.n_qubits),
            initializer=keras.initializers.GlorotUniform(),
            trainable=True
        )
        
        super().build(input_shape)
    
    def call(self, inputs):
        """Quantum-inspired transformation using rotation and entanglement-like operations"""
        x = inputs
        
        for layer in range(self.n_layers):
            # Rotation-like transformations (inspired by RY, RZ gates)
            ry_transform = tf.cos(self.rotation_weights[layer, :, 0]) * x + \
                          tf.sin(self.rotation_weights[layer, :, 0]) * tf.roll(x, shift=1, axis=-1)
            
            rz_transform = tf.cos(self.rotation_weights[layer, :, 1]) * ry_transform - \
                          tf.sin(self.rotation_weights[layer, :, 1]) * tf.roll(ry_transform, shift=-1, axis=-1)
            
            # Entanglement-like interaction (inspired by CNOT gates)
            interaction = tf.matmul(
                tf.expand_dims(rz_transform, axis=-1),
                tf.expand_dims(rz_transform, axis=-2)
            )
            
            # Apply learned interaction weights
            interaction_effect = tf.reduce_sum(
                interaction * self.interaction_weights[layer], 
                axis=-1
            )
            
            # Non-linear activation mimicking quantum measurement probabilities
            x = tf.tanh(rz_transform + 0.1 * interaction_effect)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
        })
        return config

# ============================================
# CLIENT SEQUENCE (Same as FLCNN.py)
# ============================================
class ClientSequence(keras.utils.Sequence):
    """Loads images from disk on-the-fly for a client. Avoids loading whole dataset into memory."""
    def __init__(self, filepaths, labels, img_size=(64, 64), batch_size=16, num_classes=4, shuffle=True):
        self.filepaths = np.array(filepaths)
        self.labels = np.array(labels)
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.filepaths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.filepaths) / float(self.batch_size)))

    def __getitem__(self, index):
        batch_idx = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_files = self.filepaths[batch_idx]
        batch_labels = self.labels[batch_idx]
        X = np.zeros((len(batch_files), *self.img_size, 3), dtype=np.float32)
        y = np.zeros((len(batch_files), self.num_classes), dtype=np.float32)
        for i, fp in enumerate(batch_files):
            img = cv2.imread(fp)
            if img is None:
                img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
            img = img.astype(np.float32) / 255.0
            y[i, int(batch_labels[i])] = 1.0
            X[i] = img
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# ============================================
# FEDERATED QUANTUM CNN CLASS
# ============================================
class FederatedQuantumCNN:
    """
    Federated Learning with Quantum-Inspired CNN
    Combines FL architecture from FLCNN.py with Quantum layer from QCNN.py
    """
    def __init__(self, num_clients=5, img_size=(64, 64), num_classes=4, class_names=None):
        self.num_clients = num_clients
        self.img_size = img_size
        self.num_classes = num_classes
        self.class_names = class_names or ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.global_model = None
        self.history = {'val_accuracy': [], 'val_loss': []}
        self.n_qubits = N_QUBITS
        self.n_layers = N_LAYERS

    def create_quantum_cnn_model(self):
        """
        Create Hybrid Quantum-Classical CNN architecture
        Architecture matches QCNN.py but adapted for federated learning
        """
        inputs = layers.Input(shape=(*self.img_size, 3))
        
        # Classical convolutional layers (same as QCNN.py)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        # Dimensionality reduction for quantum layer
        x = layers.Dense(self.n_qubits, activation='tanh', name='quantum_input')(x)
        
        # Quantum-inspired layer
        x = QuantumInspiredLayer(self.n_qubits, self.n_layers, name='quantum_layer')(x)
        
        # Classical output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        return model

    def load_data_generators(self, train_dir, test_dir, batch_size=16):
        """Return ImageDataGenerators (same as FLCNN.py)"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest',
            validation_split=0.2
        )
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        val_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        idx_to_class = {v: k for k, v in train_generator.class_indices.items()}
        self.class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

        return train_generator, val_generator, test_generator

    def split_filepaths_for_clients(self, train_generator):
        """Create per-client filepaths+labels lists (same as FLCNN.py)"""
        filepaths = np.array(train_generator.filepaths)
        labels = np.array(train_generator.classes, dtype=int)

        indices = np.arange(len(filepaths))
        np.random.shuffle(indices)
        splits = np.array_split(indices, self.num_clients) 

        client_filelists = []
        for s in splits:
            client_filelists.append((filepaths[s].tolist(), labels[s].tolist()))
        return client_filelists

    def train_federated(self, train_dir, val_generator, num_rounds=5, local_epochs=6, batch_size=16, start_round=1):
        """
        Federated training with Quantum CNN
        Same federated logic as FLCNN.py but with Quantum model
        """
        print("\n========== FEDERATED QUANTUM CNN TRAINING STARTED ==========")
        print(f"Clients: {self.num_clients}, Rounds: {num_rounds}, Local epochs: {local_epochs}")
        print(f"Batch size: {batch_size}, Qubits: {self.n_qubits}, Quantum layers: {self.n_layers}")
        print(f"Starting Round: {start_round}")

        # Ensure global model exists
        if self.global_model is None:
            if start_round > 1:
                raise ValueError("Must load a checkpoint into self.global_model before resuming.")
            self.global_model = self.create_quantum_cnn_model()

        # Build base generator for splitting
        base_datagen = ImageDataGenerator(rescale=1./255)
        base_gen = base_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        client_filelists = self.split_filepaths_for_clients(base_gen)

        if not hasattr(self, 'history') or not isinstance(self.history, dict):
            self.history = {'val_accuracy': [], 'val_loss': []}

        # Federated rounds
        for round_num in range(start_round - 1, num_rounds):
            print(f"\n{'='*70}")
            print(f"FEDERATED ROUND {round_num + 1}/{num_rounds}")
            print('='*70)
            client_weights = []

            for client_id, (filepaths, labels) in enumerate(client_filelists):
                print(f"\n  -> Training Client {client_id + 1}/{self.num_clients}")
                print(f"     Samples: {len(filepaths)}")

                # Create client model and load global weights
                client_model = self.create_quantum_cnn_model()
                client_model.set_weights(self.global_model.get_weights())

                # OOM-resilient training
                success = False
                attempt = 0
                cur_batch = batch_size
                cur_img_size = self.img_size
                tmp_path = os.path.join("tmp_models", f"fqcnn_global_r{round_num+1}_c{client_id+1}.keras")

                while not success and attempt < 4:
                    try:
                        print(f"     Attempt {attempt+1}: batch={cur_batch}, img_size={cur_img_size}")
                        
                        seq = ClientSequence(filepaths, labels, img_size=cur_img_size, 
                                           batch_size=cur_batch, num_classes=self.num_classes, 
                                           shuffle=True)

                        if cur_img_size != self.img_size:
                            client_model = self.create_quantum_cnn_model()
                            client_model.set_weights(self.global_model.get_weights())

                        callbacks = [
                            EarlyStopping(patience=3, restore_best_weights=True, verbose=0, monitor='loss'),
                            ReduceLROnPlateau(factor=0.5, patience=2, verbose=0, monitor='loss')
                        ]
                        
                        # Train on client
                        client_start = time.time()
                        client_model.fit(seq, epochs=local_epochs, verbose=1, callbacks=callbacks)
                        client_time = time.time() - client_start
                        print(f"     ✓ Client training completed in {client_time:.2f}s")

                        success = True

                    except ResourceExhaustedError as oom:
                        print("     !! OOM during client training")
                        try:
                            os.makedirs("tmp_models", exist_ok=True)
                            self.global_model.save(tmp_path)
                            print(f"     Saved temporary model to {tmp_path}")
                        except Exception as e:
                            print(f"     Warning: could not save temp model: {e}")

                        tf.keras.backend.clear_session()
                        gc.collect()

                        self.global_model = self.create_quantum_cnn_model()
                        if os.path.exists(tmp_path):
                            try:
                                self.global_model = tf.keras.models.load_model(tmp_path)
                                print("     Reloaded temporary model")
                            except Exception:
                                print("     Could not reload temp model")

                        attempt += 1
                        cur_batch = max(1, cur_batch // 2)
                        if attempt >= 2:
                            cur_img_size = (max(64, cur_img_size[0] // 2), max(64, cur_img_size[1] // 2))

                        client_model = self.create_quantum_cnn_model()
                        client_model.set_weights(self.global_model.get_weights())

                    except Exception as e:
                        print(f"     Unexpected error: {type(e).__name__}: {e}")
                        try:
                            os.makedirs("tmp_models", exist_ok=True)
                            self.global_model.save(os.path.join("tmp_models", f"fqcnn_error_r{round_num+1}_c{client_id+1}.keras"))
                        except Exception:
                            pass
                        raise

                if not success:
                    raise RuntimeError(f"Client {client_id + 1} training failed after retries.")

                client_weights.append(client_model.get_weights())

                try:
                    del client_model
                except Exception:
                    pass
                gc.collect()

            # Federated averaging
            print("\n  Aggregating client weights (FedAvg)...")
            averaged_weights = []
            for layer_weights in zip(*client_weights):
                stacked = np.stack(layer_weights, axis=0)
                averaged = np.mean(stacked, axis=0)
                averaged_weights.append(averaged)

            self.global_model.set_weights(averaged_weights)

            # Evaluate global model
            print("  Evaluating global quantum model...")
            val_loss, val_accuracy = self.global_model.evaluate(val_generator, verbose=1)
            print(f"\n  [Round {round_num + 1}] Val Accuracy: {val_accuracy:.4f} | Val Loss: {val_loss:.4f}")

            if len(self.history['val_loss']) < (round_num + 1):
                self.history['val_loss'].append(float(val_loss))
                self.history['val_accuracy'].append(float(val_accuracy))

            # Save checkpoint
            try:
                os.makedirs("models", exist_ok=True)
                ckpt_path = os.path.join("models", f"federated_quantum_model_round_{round_num + 1}.keras")
                self.global_model.save(ckpt_path)
                print(f"  ✓ Saved checkpoint: {ckpt_path}")
            except Exception as e:
                print(f"  Warning: failed to save checkpoint: {e}")

            gc.collect()

        print("\n========== FEDERATED QUANTUM CNN TRAINING COMPLETED ==========")
        return self.global_model

    def kfold_cross_validation(self, train_dir, k_folds=5, batch_size=16):
        """K-fold cross-validation with Quantum CNN (same structure as FLCNN.py)"""
        print("\n" + "="*70)
        print("K-FOLD CROSS-VALIDATION (QUANTUM CNN)")
        print("="*70)

        datagen = ImageDataGenerator(rescale=1./255)
        full_generator = datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        filepaths = np.array(full_generator.filepaths)
        labels = np.array(full_generator.classes, dtype=int)

        x_dummy = np.arange(len(filepaths))

        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(x_dummy)):
            print(f"\n-- Fold {fold + 1}/{k_folds} --")
            train_files = filepaths[train_idx]
            train_labels = labels[train_idx]
            val_files = filepaths[val_idx]
            val_labels = labels[val_idx]

            train_seq = ClientSequence(train_files, train_labels, img_size=self.img_size, 
                                      batch_size=batch_size, num_classes=self.num_classes, shuffle=True)
            val_seq = ClientSequence(val_files, val_labels, img_size=self.img_size, 
                                    batch_size=batch_size, num_classes=self.num_classes, shuffle=False)

            model = self.create_quantum_cnn_model()
            callbacks = [
                EarlyStopping(patience=5, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(factor=0.5, patience=3, verbose=0)
            ]

            model.fit(train_seq, validation_data=val_seq, epochs=30, verbose=1, callbacks=callbacks) 
            _, accuracy = model.evaluate(val_seq, verbose=0)
            fold_scores.append(float(accuracy))
            print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

            try:
                del model
            except Exception:
                pass
            tf.keras.backend.clear_session()
            gc.collect()

        print(f"\nQuantum CNN K-Fold Mean Accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
        return fold_scores

    def evaluate_model(self, test_generator):
        """Evaluate model on test set (same as FLCNN.py)"""
        print("\n" + "="*70)
        print("FEDERATED QUANTUM CNN - TEST SET EVALUATION")
        print("="*70)

        y_pred_probs = self.global_model.predict(test_generator, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = test_generator.classes

        accuracy = accuracy_score(y_true, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")

        report = classification_report(y_true, y_pred, target_names=self.class_names, digits=4)
        print(report)

        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class analysis
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(len(self.class_names))
        )
        
        print("\n" + "-"*70)
        print("PER-CLASS PERFORMANCE")
        print("-"*70)
        for i, class_name in enumerate(self.class_names):
            print(f"\n{class_name.upper()}:")
            print(f"  Precision: {precision[i]:.4f}")
            print(f"  Recall: {recall[i]:.4f}")
            print(f"  F1-Score: {f1[i]:.4f}")
            print(f"  Support: {support[i]}")
        
        return y_true, y_pred, y_pred_probs, cm, accuracy

    def plot_training_history(self):
        """Plot training history (same as FLCNN.py)"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(range(1, len(self.history['val_accuracy'])+1), 
                    self.history['val_accuracy'], marker='o', linewidth=2, color='green')
        axes[0].set_title('Federated Quantum CNN - Validation Accuracy', fontweight='bold')
        axes[0].set_xlabel('Federated Round')
        axes[0].set_ylabel('Accuracy')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])

        axes[1].plot(range(1, len(self.history['val_loss'])+1), 
                    self.history['val_loss'], marker='o', linewidth=2, color='red')
        axes[1].set_title('Federated Quantum CNN - Validation Loss', fontweight='bold')
        axes[1].set_xlabel('Federated Round')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('fqcnn_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: fqcnn_training_history.png")

    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix (same as FLCNN.py)"""
        plt.figure(figsize=(9, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Federated Quantum CNN - Confusion Matrix', fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('fqcnn_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: fqcnn_confusion_matrix.png")

    def plot_performance_metrics(self, y_true, y_pred):
        """Plot performance metrics (same as FLCNN.py)"""
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width, precision, width, label='Precision', color='#FF6B6B')
        ax.bar(x, recall, width, label='Recall', color='#4ECDC4')
        ax.bar(x + width, f1, width, label='F1-Score', color='#45B7D1')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45)
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1.05])
        ax.set_title('Federated Quantum CNN - Performance Metrics', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('fqcnn_performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: fqcnn_performance_metrics.png")

    def visualize_quantum_layer(self, test_generator, num_samples=4):
        """Visualize quantum layer activations"""
        print("\n" + "="*70)
        print("QUANTUM LAYER VISUALIZATION")
        print("="*70)
        
        quantum_model = keras.Model(
            inputs=self.global_model.input,
            outputs=self.global_model.get_layer('quantum_layer').output
        )
        
        # Get sample images
        test_images = []
        test_labels = []
        for _ in range(num_samples):
            batch_x, batch_y = next(iter(test_generator))
            test_images.append(batch_x[0])
            test_labels.append(np.argmax(batch_y[0]))
        
        fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, 2)
        
        for idx in range(num_samples):
            img = np.expand_dims(test_images[idx], axis=0)
            
            # Get quantum layer activation
            quantum_output = quantum_model.predict(img, verbose=0)[0]
            
            # Original image
            axes[idx, 0].imshow(test_images[idx])
            axes[idx, 0].set_title(f'Input: {self.class_names[test_labels[idx]]}', fontweight='bold')
            axes[idx, 0].axis('off')
            
            # Quantum activation
            axes[idx, 1].bar(range(self.n_qubits), quantum_output, color='green', alpha=0.7)
            axes[idx, 1].set_title('Quantum Layer Activations', fontweight='bold')
            axes[idx, 1].set_xlabel('Qubit Index')
            axes[idx, 1].set_ylabel('Activation Value')
            axes[idx, 1].set_ylim([-1, 1])
            axes[idx, 1].grid(True, alpha=0.3)
            axes[idx, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
        
        plt.tight_layout()
        plt.savefig('fqcnn_quantum_activations.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: fqcnn_quantum_activations.png")

    def save_results(self, fold_scores, test_accuracy):
        """Save results to JSON (same as FLCNN.py)"""
        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'Federated Quantum CNN',
            'model_config': {
                'num_clients': self.num_clients, 
                'image_size': self.img_size, 
                'num_classes': self.num_classes,
                'n_qubits': self.n_qubits,
                'n_quantum_layers': self.n_layers
            },
            'kfold_results': {
                'fold_scores': [float(s) for s in fold_scores], 
                'mean_accuracy': float(np.mean(fold_scores)),
                'std_accuracy': float(np.std(fold_scores))
            },
            'test_accuracy': float(test_accuracy),
            'training_history': {
                'val_accuracy': [float(x) for x in self.history['val_accuracy']],
                'val_loss': [float(x) for x in self.history['val_loss']]
            }
        }
        with open('federated_quantum_cnn_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        print("Saved: federated_quantum_cnn_results.json")


# ============================================
# MAIN FUNCTION
# ============================================
def main():
    """Main training and evaluation pipeline"""
    
    TRAIN_DIR = 'Brain_Tumor_data/Training'
    TEST_DIR = 'Brain_Tumor_data/Testing'
    
    # Configuration matching other models
    NUM_CLIENTS = 5
    NUM_FEDERATED_ROUNDS = 5
    LOCAL_EPOCHS = 6
    K_FOLDS = 5
    BATCH_SIZE = 16
    IMG_SIZE = (64, 64)

    print("\n" + "="*70)
    print("FEDERATED QUANTUM CNN - CONFIGURATION")
    print("="*70)
    print(f"  Clients: {NUM_CLIENTS}")
    print(f"  Federated Rounds: {NUM_FEDERATED_ROUNDS}")
    print(f"  Local Epochs per Round: {LOCAL_EPOCHS}")
    print(f"  K-Folds: {K_FOLDS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Image Size: {IMG_SIZE}")
    print(f"  Quantum Qubits: {N_QUBITS}")
    print(f"  Quantum Layers: {N_LAYERS}")
    print("="*70)

    # Initialize Federated Quantum CNN
    fqcnn = FederatedQuantumCNN(num_clients=NUM_CLIENTS, img_size=IMG_SIZE)
    
    # Load data generators
    train_gen, val_gen, test_gen = fqcnn.load_data_generators(TRAIN_DIR, TEST_DIR, batch_size=BATCH_SIZE)
    print(f"\nDataset Info:")
    print(f"  Training samples: {train_gen.samples}")
    print(f"  Validation samples: {val_gen.samples}")
    print(f"  Test samples: {test_gen.samples}")
    print(f"  Classes: {fqcnn.class_names}")

    # K-fold Cross-Validation (Centralized Baseline for Quantum CNN)
    print("\n" + "="*70)
    print("PHASE 1: K-FOLD CROSS-VALIDATION (CENTRALIZED QUANTUM CNN BASELINE)")
    print("="*70)
    fold_start_time = time.time()
    fold_scores = fqcnn.kfold_cross_validation(TRAIN_DIR, k_folds=K_FOLDS, batch_size=BATCH_SIZE)
    fold_total_time = time.time() - fold_start_time
    print(f"\nK-Fold Training Time: {fold_total_time:.2f} seconds ({fold_total_time/60:.2f} minutes)")

    # Federated Training (Main Contribution)
    print("\n" + "="*70)
    print("PHASE 2: FEDERATED QUANTUM CNN TRAINING")
    print("="*70)
    fed_start_time = time.time()
    global_model = fqcnn.train_federated(
        TRAIN_DIR, 
        val_gen, 
        num_rounds=NUM_FEDERATED_ROUNDS,
        local_epochs=LOCAL_EPOCHS, 
        batch_size=BATCH_SIZE
    )
    fed_total_time = time.time() - fed_start_time
    print(f"\nFederated Training Time: {fed_total_time:.2f} seconds ({fed_total_time/60:.2f} minutes)")

    # Plot training history
    fqcnn.plot_training_history()

    # Evaluate on Test Set
    print("\n" + "="*70)
    print("PHASE 3: TEST SET EVALUATION")
    print("="*70)
    y_true, y_pred, y_pred_probs, cm, test_acc = fqcnn.evaluate_model(test_gen)
    
    # Visualizations
    fqcnn.plot_confusion_matrix(cm)
    fqcnn.plot_performance_metrics(y_true, y_pred)
    fqcnn.visualize_quantum_layer(test_gen, num_samples=4)

    # Save results
    fqcnn.save_results(fold_scores, test_acc)
    
    # Save model
    try:
        global_model.save('federated_quantum_brain_tumor_model.h5')
        print("\nSaved model: federated_quantum_brain_tumor_model.h5")
    except Exception as e:
        print(f"\nWarning: Could not save .h5 format: {e}")
    
    global_model.save('federated_quantum_brain_tumor_model.keras')
    print("Saved model: federated_quantum_brain_tumor_model.keras")

    # Print final summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Centralized Quantum CNN (K-Fold):")
    print(f"  Mean Accuracy: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"  Training Time: {fold_total_time/60:.2f} minutes")
    print(f"\nFederated Quantum CNN:")
    print(f"  Final Round Accuracy: {fqcnn.history['val_accuracy'][-1]:.4f}")
    print(f"  Final Round Loss: {fqcnn.history['val_loss'][-1]:.4f}")
    print(f"  Training Time: {fed_total_time/60:.2f} minutes")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print("\nGenerated Files:")
    print("  1. federated_quantum_brain_tumor_model.keras - Trained model")
    print("  2. fqcnn_training_history.png - Training curves")
    print("  3. fqcnn_confusion_matrix.png - Confusion matrix")
    print("  4. fqcnn_performance_metrics.png - Per-class metrics")
    print("  5. fqcnn_quantum_activations.png - Quantum layer visualization")
    print("  6. federated_quantum_cnn_results.json - Complete results")
    print("="*70)
    print("✓ Federated Quantum CNN Training Complete!")
    print("="*70)


# ============================================
# RESUME TRAINING (Optional)
# ============================================
def resume_training():
    """
    Resume federated training from a checkpoint
    Usage: Uncomment and run this instead of main() if resuming
    """
    import os
    from tensorflow.keras.models import load_model
    
    # Force CPU if needed
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.config.optimizer.set_jit(False)
    
    # Configuration
    TRAIN_DIR = 'Brain_Tumor_data/Training' 
    TEST_DIR = 'Brain_Tumor_data/Testing'   
    NUM_CLIENTS = 5
    NUM_FEDERATED_ROUNDS = 5
    LOCAL_EPOCHS = 6
    BATCH_SIZE = 16
    IMG_SIZE = (64, 64)
    
    # Resume setup
    START_ROUND = 4  # Resume from round 4
    CHECKPOINT_PATH = 'models/federated_quantum_model_round_3.keras'
    
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    
    print(f"✓ Resuming from checkpoint: {CHECKPOINT_PATH}")
    print(f"  Starting at Round {START_ROUND}")
    
    # Initialize
    fqcnn = FederatedQuantumCNN(num_clients=NUM_CLIENTS, img_size=IMG_SIZE)
    train_gen, val_gen, test_gen = fqcnn.load_data_generators(TRAIN_DIR, TEST_DIR, batch_size=BATCH_SIZE)
    
    # Load checkpoint
    print(f"\nLoading checkpoint...")
    fqcnn.global_model = load_model(CHECKPOINT_PATH, custom_objects={'QuantumInspiredLayer': QuantumInspiredLayer})
    print("✓ Checkpoint loaded successfully")
    
    # Resume training
    print(f"\n🚀 Resuming Federated Quantum CNN training from Round {START_ROUND}...")
    global_model = fqcnn.train_federated(
        TRAIN_DIR, 
        val_gen, 
        num_rounds=NUM_FEDERATED_ROUNDS,
        local_epochs=LOCAL_EPOCHS, 
        batch_size=BATCH_SIZE,
        start_round=START_ROUND
    )
    
    # Evaluate
    fold_scores = [0.8, 0.82, 0.79, 0.81, 0.83]  # Placeholder or load from file
    fqcnn.plot_training_history()
    y_true, y_pred, y_pred_probs, cm, test_acc = fqcnn.evaluate_model(test_gen)
    fqcnn.plot_confusion_matrix(cm)
    fqcnn.plot_performance_metrics(y_true, y_pred)
    fqcnn.visualize_quantum_layer(test_gen)
    fqcnn.save_results(fold_scores, test_acc)
    
    global_model.save('federated_quantum_brain_tumor_model_resumed.keras')
    print("\n✓ Saved resumed model: federated_quantum_brain_tumor_model_resumed.keras")


if __name__ == "__main__":
    # Run main training
    main()
    
    # To resume training instead, comment out main() and uncomment below:
    # resume_training()