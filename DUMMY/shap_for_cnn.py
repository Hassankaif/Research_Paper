import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap 
import traceback

# =======================================================
# 1️⃣ Setup Paths and Load Model
# =======================================================
MODEL_PATH = "classical_cnn_complete.keras"
TEST_DIR = "Brain_tumor_data/Testing"

try:
    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model_to_explain = load_model(MODEL_PATH)
    print(" Model loaded successfully.")
except Exception as e:
    print(f"[FATAL ERROR] Could not load model. Ensure the file exists and is in the correct format. Error: {e}")
    exit()

# =======================================================
# 2️⃣ Load Test Dataset
# =======================================================
print(f"[INFO] Loading test data from: {TEST_DIR}")
datagen = ImageDataGenerator(rescale=1./255)
test_data = datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

class_labels = list(test_data.class_indices.keys())
print(f"[INFO] Found {len(test_data.filenames)} test images in {len(class_labels)} classes: {class_labels}")

# =======================================================
# 3️⃣ SHAP Visualization (Model Interpretation)
# =======================================================
print("\n Starting SHAP Interpretation...")
num_samples = 50  # Number of background images for explainer
num_images_to_explain = 3 # Number of test images to visualize

# --- Prepare data for SHAP Explainer (Background and Explain Images) ---

# 3.1 Collect X_background
test_data.reset()
X_background = []
# Collect background data from the generator
print(f"[INFO] Collecting {num_samples} background samples...")
for i in range(int(np.ceil(num_samples / test_data.batch_size))):
    X_batch, _ = next(test_data)
    X_background.extend(X_batch)
    if len(X_background) >= num_samples:
        X_background = np.array(X_background[:num_samples])
        break

# 3.2 Prepare images to explain (X_explain)
test_data.reset()
X_explain, y_cat_explain = next(test_data)
X_explain = X_explain[:num_images_to_explain]
y_true_labels = np.argmax(y_cat_explain[:num_images_to_explain], axis=1)

# --- SHAP Calculation and Plotting ---
try:
    # 3.3 Use GradientExplainer
    print("Creating SHAP GradientExplainer...")
    # NOTE: GradientExplainer requires a list of tensors for the background, not a single numpy array
    X_background_tensor = tf.convert_to_tensor(X_background, dtype=tf.float32)
    explainer = shap.GradientExplainer(model_to_explain, X_background_tensor)
    
    print(f"Calculating SHAP values for {num_images_to_explain} images (This may take time)...")
    shap_values = explainer.shap_values(X_explain)
    
    # 3.4 Visualize SHAP results
    for i in range(num_images_to_explain):
        true_label = class_labels[y_true_labels[i]]
        predicted_probs = model_to_explain.predict(np.expand_dims(X_explain[i], axis=0), verbose=0)
        predicted_index = np.argmax(predicted_probs)
        predicted_label = class_labels[predicted_index]

        print(f"\n{'='*60}")
        print(f"Image {i+1} | True: {true_label} | Predicted: {predicted_label}")
        print(f"Prediction confidence: {predicted_probs[0][predicted_index]:.3f}")
        
        # Get SHAP values for the predicted class
        # shap_values is a list where each element corresponds to a class output
        if isinstance(shap_values, list):
            shap_img = shap_values[predicted_index][i]
        else:
            # Fallback, though uncommon for multi-class CNNs
            shap_img = shap_values[i]
        
        # Sum across color channels to get a single importance map
        if len(shap_img.shape) == 3 and shap_img.shape[-1] in [1, 3]:
            # Use absolute values to show importance magnitude regardless of positive/negative contribution
            shap_img_sum = np.sum(np.abs(shap_img), axis=-1)
        else:
            shap_img_sum = np.abs(shap_img)
        
        # --- Create Visualization ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Original image
        axes[0].imshow(X_explain[i].squeeze(), cmap='gray' if X_explain[i].shape[-1] == 1 else None)
        axes[0].set_title(f"Original Image\nTrue: {true_label}", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Absolute SHAP values (importance magnitude)
        im1 = axes[1].imshow(shap_img_sum, cmap='hot', interpolation='bilinear')
        axes[1].set_title(f"SHAP Importance\n(Absolute Values)", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # SHAP overlay on original
        axes[2].imshow(X_explain[i].squeeze(), cmap='gray' if X_explain[i].shape[-1] == 1 else None, alpha=0.6)
        
        # Normalize SHAP values for better color mapping visualization
        shap_normalized = (shap_img_sum - np.min(shap_img_sum)) / (np.max(shap_img_sum) - np.min(shap_img_sum) + 1e-8)
        im2 = axes[2].imshow(shap_normalized, cmap='jet', alpha=0.5, interpolation='bilinear')
        axes[2].set_title(f"SHAP Overlay\nPredicted: {predicted_label}", fontsize=12, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.suptitle(f"SHAP Analysis: {predicted_label} (True: {true_label}) | Confidence: {predicted_probs[0][predicted_index]:.2%}", 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

except Exception as e:
    print(f"\n[ERROR] SHAP analysis failed. Error: {e}")
    traceback.print_exc()
    print(" The code includes a fallback to Grad-CAM in the original logic, which can be uncommented/used if SHAP continues to fail due to version conflicts.")