import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np # <-- Added numpy import

# Changed to the backward-compatible serialization decorator
@keras.utils.register_keras_serializable() 
class QuantumInspiredLayer(layers.Layer):
    """
    Quantum-inspired layer that emulates quantum behavior with classical operations.
    The implementation is inspired by quantum circuit structures (rotations and entanglement) 
    but uses standard TensorFlow operations.
    """
    def __init__(self, n_qubits, n_layers, **kwargs):
        super(QuantumInspiredLayer, self).__init__(**kwargs)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
    def build(self, input_shape):
        # Quantum-inspired rotation parameters (shape: [n_layers, n_qubits, 2] for RY and RZ)
        self.rotation_weights = self.add_weight(
            name="rotation_weights",
            shape=(self.n_layers, self.n_qubits, 2),
            initializer=keras.initializers.RandomUniform(0, 2 * np.pi), # <-- np is now available
            trainable=True
        )
        
        # Entanglement interaction weights (shape: [n_layers, n_qubits, n_qubits])
        self.interaction_weights = self.add_weight(
            name="interaction_weights",
            shape=(self.n_layers, self.n_qubits, self.n_qubits),
            initializer=keras.initializers.GlorotUniform(),
            trainable=True
        )
        
        super().build(input_shape)
    
    def call(self, inputs):
        """Quantum-inspired transformation using rotation and entanglement-like operations"""
        # Ensure inputs has rank 2: [batch_size, n_qubits]
        x = inputs
        
        for layer in range(self.n_layers):
            # --- 1. Rotation-like transformations (Inspired by single-qubit gates) ---
            
            # RY-like operation (combines input with a cyclic shift using trig functions)
            ry_transform = tf.cos(self.rotation_weights[layer, :, 0]) * x + \
                           tf.sin(self.rotation_weights[layer, :, 0]) * tf.roll(x, shift=1, axis=-1)
            
            # RZ-like operation (another rotation applied to the result)
            rz_transform = tf.cos(self.rotation_weights[layer, :, 1]) * ry_transform - \
                           tf.sin(self.rotation_weights[layer, :, 1]) * tf.roll(ry_transform, shift=-1, axis=-1)
            
            # --- 2. Entanglement-like interaction (Inspired by CNOT/Controlled gates) ---
            
            # Create an outer product (tensor product approximation) [batch, n_q, n_q]
            # (Note: This implements a quadratic interaction between all features/qubits)
            interaction = tf.matmul(
                tf.expand_dims(rz_transform, axis=-1),
                tf.expand_dims(rz_transform, axis=-2)
            )
            
            # Apply learned interaction weights and reduce to get a single effect per qubit
            # The result is [batch, n_qubits]
            interaction_effect = tf.reduce_sum(
                interaction * self.interaction_weights[layer], 
                axis=-1
            )
            
            # --- 3. Update State with Non-linear Activation ---
            # Non-linear activation mimicking quantum measurement probabilities
            # The 0.1 factor scales the interaction effect's contribution
            x = tf.tanh(rz_transform + 0.1 * interaction_effect)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
        })
        return config
    
    # Required for the utils.register_keras_serializable decorator (Keras 2 compatibility)
    @classmethod
    def from_config(cls, config):
        return cls(**config)
