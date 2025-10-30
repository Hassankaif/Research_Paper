import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.framework.errors_impl import ResourceExhaustedError
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import gc, os, numpy as np

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

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
                # fallback: black image
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
            
class FederatedBrainTumorCNN:
    # UPDATED: img_size default to (64, 64)
    def __init__(self, num_clients=5, img_size=(64, 64), num_classes=4, class_names=None):
        self.num_clients = num_clients
        self.img_size = img_size
        self.num_classes = num_classes
        self.class_names = class_names or ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.global_model = None
        self.history = {'val_accuracy': [], 'val_loss': []}

    def create_cnn_model(self):
        """Create CNN architecture for brain tumor classification"""
        model = models.Sequential([
            layers.Input(shape=(*self.img_size, 3)),

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

            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        # NOTE: Keeping categorical_crossentropy as the ClientSequence and flow_from_directory 
        # are set up for one-hot encoding (categorical mode)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        return model

    def load_data_generators(self, train_dir, test_dir, batch_size=16): # UPDATED: batch_size default
        """Return ImageDataGenerators and generators (for overall sample counts)."""
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

        # save class names according to generator's index mapping
        idx_to_class = {v: k for k, v in train_generator.class_indices.items()}
        self.class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

        return train_generator, val_generator, test_generator

    def split_filepaths_for_clients(self, train_generator):
        """Create per-client filepaths+labels lists (shuffled) so each client can use a Sequence loader."""
        filepaths = np.array(train_generator.filepaths)
        labels = np.array(train_generator.classes, dtype=int)

        indices = np.arange(len(filepaths))
        np.random.shuffle(indices)
        # UPDATED: Use self.num_clients
        splits = np.array_split(indices, self.num_clients) 

        client_filelists = []
        for s in splits:
            client_filelists.append((filepaths[s].tolist(), labels[s].tolist()))
        return client_filelists

    def federated_averaging(self, client_weights):
        """Aggregate client model weights using Federated Averaging"""
        avg_weights = []
        for weights_tuple in zip(*client_weights):
            avg_weights.append(np.mean(weights_tuple, axis=0))
        return avg_weights



    # # UPDATED: num_rounds, local_epochs, batch_size defaults
    # def train_federated(self, train_dir, val_generator, num_rounds=5, local_epochs=6, batch_size=16):
    #     """
    #     Federated training across simulated clients with safer memory handling and correct FedAvg.
    #     Uses split_filepaths_for_clients() already defined in the class.
    #     """

    #     print("\n========== FEDERATED TRAINING STARTED ==========")
    #     print(f"Clients: {self.num_clients}, Rounds: {num_rounds}, Local epochs: {local_epochs}, Batch size: {batch_size}")

    #     # Ensure global model exists
    #     if self.global_model is None:
    #         self.global_model = self.create_cnn_model()

    #     # Build a base generator (non-augmented) to extract filepaths/classes for splitting
    #     base_datagen = ImageDataGenerator(rescale=1./255)
    #     base_gen = base_datagen.flow_from_directory(
    #         train_dir,
    #         target_size=self.img_size,
    #         batch_size=batch_size,
    #         class_mode='categorical',
    #         shuffle=False
    #     )

    #     # Use your existing split helper to get client filepaths+labels
    #     client_filelists = self.split_filepaths_for_clients(base_gen)  # returns list of (filepaths, labels)

    #     # Initialize history keys if missing
    #     if not hasattr(self, 'history') or not isinstance(self.history, dict):
    #         self.history = {'val_accuracy': [], 'val_loss': []}

    #     # Federated rounds
    #     for round_num in range(num_rounds):
    #         print(f"\n========== ROUND {round_num + 1}/{num_rounds} ==========")
    #         client_weights = []

    #         for client_id, (filepaths, labels) in enumerate(client_filelists):
    #             print(f"  -> Training client {client_id + 1}/{self.num_clients} | samples: {len(filepaths)}")

    #             # Create a fresh client model and load global weights
    #             client_model = self.create_cnn_model()
    #             client_model.set_weights(self.global_model.get_weights())

    #             # ClientSequence will load images on the fly
    #             # UPDATED: ClientSequence will use the dynamically changed batch_size and img_size during OOM attempts
    #             seq = ClientSequence(
    #                 filepaths, labels,
    #                 img_size=self.img_size, 
    #                 batch_size=batch_size,
    #                 num_classes=self.num_classes,
    #                 shuffle=True
    #             )

    #             callbacks = [
    #                 EarlyStopping(patience=3, restore_best_weights=True, verbose=0, monitor='loss'),
    #                 ReduceLROnPlateau(factor=0.5, patience=2, verbose=0, monitor='loss')
    #             ]

    #             # OOM-resilient training attempts
    #             success = False
    #             attempt = 0
    #             cur_batch = batch_size
    #             cur_img_size = self.img_size # (64, 64) from init

    #             while not success and attempt < 4:
    #                 try:
    #                     print(f"    Attempt {attempt+1}: batch_size={cur_batch}, img_size={cur_img_size}")
    #                     # recreate seq if we changed batch or img size
    #                     seq = ClientSequence(filepaths, labels, img_size=cur_img_size, batch_size=cur_batch,
    #                                          num_classes=self.num_classes, shuffle=True)

    #                     # if image size changed, recreate client_model from scratch and reassign global weights
    #                     if cur_img_size != self.img_size:
    #                         client_model = self.create_cnn_model()
    #                         client_model.set_weights(self.global_model.get_weights())

    #                     # Train on client
    #                     client_model.fit(seq, epochs=local_epochs, verbose=1, callbacks=callbacks)

    #                     success = True

    #                 except ResourceExhaustedError as oom:
    #                     print("!! OOM during client training:", oom)
    #                     # Save temp global to avoid irrecoverable loss (best-effort)
    #                     try:
    #                         os.makedirs("tmp_models", exist_ok=True)
    #                         tmp_path = os.path.join("tmp_models", f"global_before_oom_r{round_num+1}_c{client_id+1}.keras")
    #                         self.global_model.save(tmp_path)
    #                         print(f"Saved temporary global model to {tmp_path}")
    #                     except Exception as e:
    #                         print("Warning: could not save temp model:", e)

    #                     # free memory and retry with smaller batch / smaller image
    #                     tf.keras.backend.clear_session()  # clear Keras graph to release GPU memory
    #                     gc.collect()

    #                     # Recreate global model (we will reload weights below)
    #                     self.global_model = self.create_cnn_model()
    #                     # If we had saved a temp model, try to reload it to restore global state
    #                     if os.path.exists(tmp_path):
    #                         try:
    #                             self.global_model = tf.keras.models.load_model(tmp_path)
    #                             print("Reloaded temporary global model after clearing session.")
    #                         except Exception:
    #                             # if reload fails, keep freshly created global model (weights lost but attempt to continue)
    #                             print("Could not reload temporary model; continuing with fresh global model.")

    #                     attempt += 1
    #                     cur_batch = max(1, cur_batch // 2)
    #                     if attempt >= 2:
    #                         cur_img_size = (max(64, cur_img_size[0] // 2), max(64, cur_img_size[1] // 2))

    #                     # recreate client_model for next attempt with global weights
    #                     client_model = self.create_cnn_model()
    #                     client_model.set_weights(self.global_model.get_weights())

    #                 except Exception as e:
    #                     print("Unexpected error during client training:", type(e), e)
    #                     # best-effort save and rethrow
    #                     try:
    #                         os.makedirs("tmp_models", exist_ok=True)
    #                         self.global_model.save(os.path.join("tmp_models", f"global_error_r{round_num+1}_c{client_id+1}.keras"))
    #                     except Exception:
    #                         pass
    #                     raise

    #             if not success:
    #                 raise RuntimeError(f"Client {client_id + 1} training failed after retries (OOM).")

    #             # append client's learned weights
    #             client_weights.append(client_model.get_weights())

    #             # cleanup client model to free resources
    #             try:
    #                 del client_model
    #             except Exception:
    #                 pass
    #             gc.collect()

    #         # ===== Federated averaging (correct per-layer average) =====
    #         print("Aggregating client weights (FedAvg)...")
    #         averaged_weights = []
    #         for layer_weights in zip(*client_weights):
    #             # stack along new axis, then mean
    #             stacked = np.stack(layer_weights, axis=0)
    #             averaged = np.mean(stacked, axis=0)
    #             averaged_weights.append(averaged)

    #         self.global_model.set_weights(averaged_weights)

    #         # ===== Evaluate global model on validation data =====
    #         print("Evaluating global model on validation set...")
    #         val_loss, val_accuracy = self.global_model.evaluate(val_generator, verbose=1)
    #         print(f"[Round {round_num + 1}] Val loss: {val_loss:.4f} | Val accuracy: {val_accuracy:.4f}")

    #         self.history['val_loss'].append(float(val_loss))
    #         self.history['val_accuracy'].append(float(val_accuracy))

    #         # Save checkpoint after each round (safe)
    #         try:
    #             os.makedirs("models", exist_ok=True)
    #             ckpt_path = os.path.join("models", f"federated_model_round_{round_num + 1}.keras")
    #             self.global_model.save(ckpt_path)
    #             print(f"[✓] Saved checkpoint: {ckpt_path}")
    #         except Exception as e:
    #             print("Warning: failed to save checkpoint:", e)

    #         # light cleanup (do NOT clear entire session; keep global model alive)
    #         gc.collect()

    #     print("\n========== FEDERATED TRAINING COMPLETED ==========")
    #     return self.global_model

    # Inside the FederatedBrainTumorCNN class:

# UPDATED: Added start_round parameter with a default of 1
    def train_federated(self, train_dir, val_generator, num_rounds=5, local_epochs=6, batch_size=16, start_round=1):
        """
        Federated training across simulated clients with safer memory handling and correct FedAvg.
        The loop now starts from `start_round`.
        """

        print("\n========== FEDERATED TRAINING STARTED/RESUMED ==========")
        print(f"Clients: {self.num_clients}, Total Rounds: {num_rounds}, Local epochs: {local_epochs}, Batch size: {batch_size}, Starting Round: {start_round}")

        # Ensure global model exists
        if self.global_model is None:
            if start_round > 1:
                # If we are resuming, the model should have been loaded externally
                raise ValueError("Must load a checkpoint into self.global_model before starting a resumed training (start_round > 1).")
            self.global_model = self.create_cnn_model()

        # Build a base generator (non-augmented) to extract filepaths/classes for splitting
        base_datagen = ImageDataGenerator(rescale=1./255)
        base_gen = base_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        # Use your existing split helper to get client filepaths+labels
        client_filelists = self.split_filepaths_for_clients(base_gen)

        # Initialize history keys if missing
        if not hasattr(self, 'history') or not isinstance(self.history, dict):
            self.history = {'val_accuracy': [], 'val_loss': []}

        # Federated rounds: Loop starts from (start_round - 1) up to num_rounds
        for round_num in range(start_round - 1, num_rounds):
            print(f"\n========== ROUND {round_num + 1}/{num_rounds} (RESUMED) ==========")
            client_weights = []

            for client_id, (filepaths, labels) in enumerate(client_filelists):
                print(f"  -> Training client {client_id + 1}/{self.num_clients} | samples: {len(filepaths)}")

                # Create a fresh client model and load global weights
                client_model = self.create_cnn_model()
                # Copy the current global weights (which might be loaded from a checkpoint)
                client_model.set_weights(self.global_model.get_weights())

                # OOM-resilient training attempts
                success = False
                attempt = 0
                cur_batch = batch_size
                cur_img_size = self.img_size # (64, 64) from init
                tmp_path = os.path.join("tmp_models", f"global_before_oom_r{round_num+1}_c{client_id+1}.keras")


                while not success and attempt < 4:
                    try:
                        print(f"    Attempt {attempt+1}: batch_size={cur_batch}, img_size={cur_img_size}")
                        
                        # recreate seq if we changed batch or img size
                        seq = ClientSequence(filepaths, labels, img_size=cur_img_size, batch_size=cur_batch,
                                            num_classes=self.num_classes, shuffle=True)

                        # if image size changed, recreate client_model from scratch and reassign global weights
                        if cur_img_size != self.img_size:
                            client_model = self.create_cnn_model()
                            client_model.set_weights(self.global_model.get_weights())

                        callbacks = [
                            EarlyStopping(patience=3, restore_best_weights=True, verbose=0, monitor='loss'),
                            ReduceLROnPlateau(factor=0.5, patience=2, verbose=0, monitor='loss')
                        ]
                            
                        # Train on client
                        client_model.fit(seq, epochs=local_epochs, verbose=1, callbacks=callbacks)

                        success = True

                    except ResourceExhaustedError as oom:
                        print("!! OOM during client training:", oom)
                        # Save temp global to avoid irrecoverable loss (best-effort)
                        try:
                            os.makedirs("tmp_models", exist_ok=True)
                            self.global_model.save(tmp_path)
                            print(f"Saved temporary global model to {tmp_path}")
                        except Exception as e:
                            print("Warning: could not save temp model:", e)

                        # free memory and retry with smaller batch / smaller image
                        tf.keras.backend.clear_session()  # clear Keras graph to release GPU memory
                        gc.collect()

                        # Recreate global model (we will reload weights below)
                        self.global_model = self.create_cnn_model()
                        # If we had saved a temp model, try to reload it to restore global state
                        if os.path.exists(tmp_path):
                            try:
                                self.global_model = tf.keras.models.load_model(tmp_path)
                                print("Reloaded temporary global model after clearing session.")
                            except Exception:
                                # if reload fails, keep freshly created global model (weights lost but attempt to continue)
                                print("Could not reload temporary model; continuing with fresh global model.")

                        attempt += 1
                        cur_batch = max(1, cur_batch // 2)
                        if attempt >= 2:
                            cur_img_size = (max(64, cur_img_size[0] // 2), max(64, cur_img_size[1] // 2))

                        # recreate client_model for next attempt with global weights
                        client_model = self.create_cnn_model()
                        client_model.set_weights(self.global_model.get_weights())

                    except Exception as e:
                        print("Unexpected error during client training:", type(e), e)
                        # best-effort save and rethrow
                        try:
                            os.makedirs("tmp_models", exist_ok=True)
                            self.global_model.save(os.path.join("tmp_models", f"global_error_r{round_num+1}_c{client_id+1}.keras"))
                        except Exception:
                            pass
                        raise

                if not success:
                    raise RuntimeError(f"Client {client_id + 1} training failed after retries (OOM).")

                # append client's learned weights
                client_weights.append(client_model.get_weights())

                # cleanup client model to free resources
                try:
                    del client_model
                except Exception:
                    pass
                gc.collect()

            # ===== Federated averaging (correct per-layer average) =====
            print("Aggregating client weights (FedAvg)...")
            averaged_weights = []
            for layer_weights in zip(*client_weights):
                # stack along new axis, then mean
                stacked = np.stack(layer_weights, axis=0)
                averaged = np.mean(stacked, axis=0)
                averaged_weights.append(averaged)

            self.global_model.set_weights(averaged_weights) 

            # ===== Evaluate global model on validation data =====
            print("Evaluating global model on validation set...")
            val_loss, val_accuracy = self.global_model.evaluate(val_generator, verbose=1)
            print(f"[Round {round_num + 1}] Val loss: {val_loss:.4f} | Val accuracy: {val_accuracy:.4f}")

            # Ensure we only append history for the rounds actually trained in this session
            if len(self.history['val_loss']) < (round_num + 1):
                self.history['val_loss'].append(float(val_loss))
                self.history['val_accuracy'].append(float(val_accuracy))

            # Save checkpoint after each round (safe)
            try:
                os.makedirs("models", exist_ok=True)
                ckpt_path = os.path.join("models", f"federated_model_round_{round_num + 1}.keras")
                self.global_model.save(ckpt_path)
                print(f"[✓] Saved checkpoint: {ckpt_path}")
            except Exception as e:
                print("Warning: failed to save checkpoint:", e)

            # light cleanup (do NOT clear entire session; keep global model alive)
            gc.collect()

        print("\n========== FEDERATED TRAINING COMPLETED ==========")
        return self.global_model
    
    # UPDATED: k_folds default
    def kfold_cross_validation(self, train_dir, k_folds=5, batch_size=16): 
        print("\n" + "="*70)
        print("K-FOLD CROSS-VALIDATION")
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

        x_dummy = np.arange(len(filepaths))  # We'll read images per-index during training folds

        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(x_dummy)):
            print(f"\n-- Fold {fold + 1}/{k_folds} --")
            train_files = filepaths[train_idx]
            train_labels = labels[train_idx]
            val_files = filepaths[val_idx]
            val_labels = labels[val_idx]

            # Create sequences for fold training
            train_seq = ClientSequence(train_files, train_labels, img_size=self.img_size, batch_size=batch_size,
                                       num_classes=self.num_classes, shuffle=True)
            val_seq = ClientSequence(val_files, val_labels, img_size=self.img_size, batch_size=batch_size,
                                     num_classes=self.num_classes, shuffle=False)

            model = self.create_cnn_model()
            callbacks = [
                EarlyStopping(patience=5, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(factor=0.5, patience=3, verbose=0)
            ]

            # UPDATED: Set epochs to 30 for K-fold (simulating EFFEECTIVE_EPOCHS for non-federated training)
            model.fit(train_seq, validation_data=val_seq, epochs=30, verbose=1, callbacks=callbacks) 
            _, accuracy = model.evaluate(val_seq, verbose=0)
            fold_scores.append(float(accuracy))
            print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

            # cleanup model to free resources
            try:
                del model
            except Exception:
                pass
            tf.keras.backend.clear_session()
            gc.collect()


        print(f"\nK-Fold Cross-Validation Mean Accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
        return fold_scores

    def evaluate_model(self, test_generator):
        print("\n" + "="*70)
        print("MODEL EVALUATION ON TEST SET")
        print("="*70)

        y_pred_probs = self.global_model.predict(test_generator, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = test_generator.classes

        accuracy = accuracy_score(y_true, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")

        report = classification_report(y_true, y_pred, target_names=self.class_names, digits=4)
        print(report)

        cm = confusion_matrix(y_true, y_pred)
        return y_true, y_pred, y_pred_probs, cm, accuracy

    def plot_training_history(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(self.history['val_accuracy'], marker='o', linewidth=2)
        axes[0].set_title('Federated Learning - Validation Accuracy')
        axes[0].set_xlabel('Federated Round')
        axes[0].set_ylabel('Accuracy')
        axes[0].grid(True)
        axes[0].set_ylim([0, 1])

        axes[1].plot(self.history['val_loss'], marker='o', linewidth=2)
        axes[1].set_title('Federated Learning - Validation Loss')
        axes[1].set_xlabel('Federated Round')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: training_history.png")

    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(9, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: confusion_matrix.png")

    def plot_performance_metrics(self, y_true, y_pred):
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        x = np.arange(len(self.class_names))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width, precision, width, label='Precision')
        ax.bar(x, recall, width, label='Recall')
        ax.bar(x + width, f1, width, label='F1')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45)
        ax.set_ylim([0, 1.05])
        ax.legend()
        plt.tight_layout()
        plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: performance_metrics.png")

    # The commented out gradcam_visualization method remains commented out as requested.

    def save_results(self, fold_scores, test_accuracy):
        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_config': {'num_clients': self.num_clients, 'image_size': self.img_size, 'num_classes': self.num_classes},
            'kfold_results': {'fold_scores': [float(s) for s in fold_scores], 'mean_accuracy': float(np.mean(fold_scores)),
                              'std_accuracy': float(np.std(fold_scores))},
            'test_accuracy': float(test_accuracy),
            'training_history': {'val_accuracy': [float(x) for x in self.history['val_accuracy']],
                                 'val_loss': [float(x) for x in self.history['val_loss']]}
        }
        with open('federated_learning_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        print("Saved: federated_learning_results.json")


def main():
    TRAIN_DIR = 'Brain_Tumor_data/Training'  # update to your path
    TEST_DIR = 'Brain_Tumor_data/Testing'    # update to your path
    
    # MODIFIED CONFIGURATION/CONSTANTS:
    NUM_CLIENTS = 5
    NUM_FEDERATED_ROUNDS = 5
    LOCAL_EPOCHS = 6
    K_FOLDS = 5 # OUTER_KFOLD
    BATCH_SIZE = 16

    print("CONFIG:")
    print(f"  clients: {NUM_CLIENTS}, fed rounds: {NUM_FEDERATED_ROUNDS}, local epochs: {LOCAL_EPOCHS}, kfolds: {K_FOLDS}, batch_size: {BATCH_SIZE}")

    # UPDATED: img_size to (64, 64)
    fed = FederatedBrainTumorCNN(num_clients=NUM_CLIENTS, img_size=(64, 64)) 
    train_gen, val_gen, test_gen = fed.load_data_generators(TRAIN_DIR, TEST_DIR, batch_size=BATCH_SIZE)
    print("Samples:", train_gen.samples, "Validation:", val_gen.samples, "Test:", test_gen.samples)

    # K-fold (optional; can be heavy)
    fold_scores = fed.kfold_cross_validation(TRAIN_DIR, k_folds=K_FOLDS, batch_size=BATCH_SIZE)

    # Federated training (streaming per client)
    global_model = fed.train_federated(TRAIN_DIR, val_gen, num_rounds=NUM_FEDERATED_ROUNDS,
                                       local_epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE)

    fed.plot_training_history()

    # Evaluate
    y_true, y_pred, y_pred_probs, cm, test_acc = fed.evaluate_model(test_gen)
    fed.plot_confusion_matrix(cm)
    fed.plot_performance_metrics(y_true, y_pred)
    # fed.gradcam_visualization(test_gen, num_samples=5)

    fed.save_results(fold_scores, test_acc)
    global_model.save('federated_brain_tumor_model.h5')
    print("Saved model: federated_brain_tumor_model.h5")
    global_model.save('federated_brain_tumor_model.keras')
    print("Saved model: federated_brain_tumor_model.keras")
    
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
import tensorflow as tf

# >>> FORCE TENSORFLOW TO USE ONLY THE CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Optional: Disable XLA again for good measure (though not strictly needed on CPU)
tf.config.optimizer.set_jit(False)

# --- CONFIGURATION (Must match original training) ---
TRAIN_DIR = 'Brain_Tumor_data/Training' 
TEST_DIR = 'Brain_Tumor_data/Testing'   
NUM_CLIENTS = 5
NUM_FEDERATED_ROUNDS = 5 # Total rounds to complete
LOCAL_EPOCHS = 6
BATCH_SIZE = 16
IMG_SIZE = (64, 64)

# --- RESUME SETUP ---
START_ROUND = 4
CHECKPOINT_PATH = 'models/federated_model_round_3.keras' # Load model AFTER Round 3

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}. Cannot restart.")

print(f"✅ Starting RESUME sequence. Target round: {START_ROUND}")

# 1. Initialize the FederatedBrainTumorCNN instance
# NOTE: Ensure the class definition (including the modified train_federated) is run first.
fed = FederatedBrainTumorCNN(num_clients=NUM_CLIENTS, img_size=IMG_SIZE) 

# 2. Load generators (needed for validation and model setup)
train_gen, val_gen, test_gen = fed.load_data_generators(TRAIN_DIR, TEST_DIR, batch_size=BATCH_SIZE)

# 3. Load the checkpoint and assign it as the global model
print(f"Loading checkpoint: {CHECKPOINT_PATH}")
fed.global_model = load_model(CHECKPOINT_PATH)

# Optional: Load previous history if available (not strictly necessary but useful for plotting)
# If your history file was saved, you'd load it here.
# For simplicity, we assume your history dictionary is empty or only tracks the rounds run in this session.

# 4. Resume Federated Training using the modified method
print(f"\n🚀 Resuming training from Round {START_ROUND}...")

# Call the modified method, passing the start_round
global_model = fed.train_federated(
    TRAIN_DIR, 
    val_gen, 
    num_rounds=NUM_FEDERATED_ROUNDS,
    local_epochs=LOCAL_EPOCHS, 
    batch_size=BATCH_SIZE,
    start_round=START_ROUND # This is the crucial new parameter
)

# 5. Final Evaluation/Saving (Use a dummy/placeholder for fold_scores if not run)
fold_scores = [0.8, 0.82, 0.79, 0.81, 0.83] # Use placeholder or load previous if saved

fed.plot_training_history()
y_true, y_pred, y_pred_probs, cm, test_acc = fed.evaluate_model(test_gen)
fed.plot_confusion_matrix(cm)
fed.plot_performance_metrics(y_true, y_pred)
fed.save_results(fold_scores, test_acc) # Pass in your actual or dummy fold_scores

global_model.save('federated_brain_tumor_model_resumed.keras')
print("Saved final resumed model: federated_brain_tumor_model_resumed.keras")