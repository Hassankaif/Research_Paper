# main.py

import tensorflow as tf
from tensorflow.keras.models import load_model
from qml_layers import QuantumInspiredLayer, create_quantum_cnn_model

# Load saved model with custom layer
custom_objects = {'QuantumInspiredLayer': QuantumInspiredLayer}
model = load_model("quantum_cnn_brain_tumor_model.keras", custom_objects=custom_objects)

print("✅ Loaded model from file:")
model.summary()

# Optionally rebuild model from code
reconstructed_model = create_quantum_cnn_model()
print("\n🔧 Reconstructed model from code:")
reconstructed_model.summary()
