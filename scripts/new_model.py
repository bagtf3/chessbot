import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

BIG_NEG = -1e9
EPS = 1e-9


class MaskedPolicyModel(tf.keras.Model):
    def __init__(self, core):
        super().__init__()
        
        self.core = core
        self.policy_loss_tracker = tf.keras.metrics.Mean(name="policy_loss")
        self.value_loss_tracker = tf.keras.metrics.Mean(name="value_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.policy_loss_tracker,
                self.value_loss_tracker]

    def compile(self, optimizer, policy_loss_weight=1.0, value_loss_weight=1.0,
                **kwargs):
        super().compile(**kwargs)
        self.optimizer = tf.keras.optimizers.get(optimizer)
        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight

    def train_step(self, data):
        (boards, legal_mask), y = data
        labels = y["policy"]
        values = y["value"]

        with tf.GradientTape() as tape:
            logits, v_pred = self.core([boards, legal_mask], training=True)
            B = tf.shape(logits)[0]
            logits = tf.reshape(logits, [B, -1])
            labels = tf.reshape(labels, [B, -1])
            mask = tf.cast(tf.reshape(legal_mask, [B, -1]), logits.dtype)

            masked_logits = tf.where(mask > 0.5,
                                     logits,
                                     tf.ones_like(logits) * BIG_NEG)
            logp = tf.nn.log_softmax(masked_logits, axis=1)

            label_sums = tf.reduce_sum(labels, axis=1, keepdims=True)
            labels_norm = labels / (label_sums + EPS)

            per_sample_ce = -tf.reduce_sum(labels_norm * logp, axis=1)
            valid = tf.squeeze(label_sums > EPS, axis=1)
            policy_loss = tf.reduce_sum(
                tf.where(valid, per_sample_ce, tf.zeros_like(per_sample_ce))
            ) / (tf.reduce_sum(tf.cast(valid, tf.float32)) + EPS)

            value_loss = tf.reduce_mean(tf.square(v_pred - values))
            total_loss = (self.policy_loss_weight * policy_loss +
                          self.value_loss_weight * value_loss)

        grads = tape.gradient(total_loss, self.core.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.core.trainable_variables))

        self.policy_loss_tracker.update_state(policy_loss)
        self.value_loss_tracker.update_state(value_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {"loss": self.total_loss_tracker.result(),
                "policy_loss": self.policy_loss_tracker.result(),
                "value_loss": self.value_loss_tracker.result()}

    def test_step(self, data):
        (boards, legal_mask), y = data
        labels = y["policy"]
        values = y["value"]

        logits, v_pred = self.core([boards, legal_mask], training=False)
        B = tf.shape(logits)[0]
        logits = tf.reshape(logits, [B, -1])
        labels = tf.reshape(labels, [B, -1])
        mask = tf.cast(tf.reshape(legal_mask, [B, -1]), logits.dtype)

        masked_logits = tf.where(mask > 0.5,
                                 logits,
                                 tf.ones_like(logits) * BIG_NEG)
        logp = tf.nn.log_softmax(masked_logits, axis=1)

        label_sums = tf.reduce_sum(labels, axis=1, keepdims=True)
        labels_norm = labels / (label_sums + EPS)
        per_sample_ce = -tf.reduce_sum(labels_norm * logp, axis=1)
        valid = tf.squeeze(label_sums > EPS, axis=1)
        policy_loss = tf.reduce_sum(
            tf.where(valid, per_sample_ce, tf.zeros_like(per_sample_ce))
        ) / (tf.reduce_sum(tf.cast(valid, tf.float32)) + EPS)

        value_loss = tf.reduce_mean(tf.square(v_pred - values))
        total_loss = (self.policy_loss_weight * policy_loss +
                      self.value_loss_weight * value_loss)

        self.policy_loss_tracker.update_state(policy_loss)
        self.value_loss_tracker.update_state(value_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {"loss": self.total_loss_tracker.result(),
                "policy_loss": self.policy_loss_tracker.result(),
                "value_loss": self.value_loss_tracker.result()}
    
    def call(self, inputs, training):
        return self.core(inputs, training=training)
    
    def save(self, path):
        self.core.save(path)

    @classmethod
    def from_saved(cls, path):
        """
        Load a MaskedPolicyModel previously saved with save(path).
        Returns a built subclass instance (not compiled).
        """
        # load the functional core (works for SavedModel dir or .h5 file)
        core_loaded = tf.keras.models.load_model(path)
        inst = cls(core_loaded)
        # try to build so model.summary() is usable immediately
        try:
            inst.build(input_shape=[(None, 8, 8, 29), (None, 4096)])
        except Exception:
            # if build fails (nonstandard shapes), ignore — user can call build/call
            pass
        return inst

    @classmethod
    def build_core_model_stm_pov(self):
        board_in = Input(shape=(8, 8, 29), name="board")
        legal_in = Input(shape=(4096,), name="legal_mask")

        x = layers.Conv2D(256, 3, padding="same", name="conv0")(board_in)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        # Residual-ish trunk: 12 blocks of 256 filters (wide & deep)
        for i in range(10):
            r = layers.Conv2D(256, 3, padding="same", name=f"b{i}_c1")(x)
            r = layers.BatchNormalization()(r)
            r = layers.LeakyReLU()(r)
            r = layers.Conv2D(256, 3, padding="same", name=f"b{i}_c2")(r)
            r = layers.BatchNormalization()(r)
            x = layers.Add()([x, r])
            x = layers.LeakyReLU()(x)

        trunk = x  # shape (B, 8, 8, 256)

        # Policy head: 1x1 conv -> 64 channels -> flatten to 4096 logits
        p = layers.Conv2D(64, 1, padding="same", name="policy_conv1x1")(trunk)
        policy_logits = layers.Reshape((4096,), name="policy_logits")(p)

        # Value head: GAP -> MLP -> tanh in [-1,1]
        v = layers.GlobalAveragePooling2D(name="gap")(trunk)
        v = layers.Dense(1024, activation="relu", name="v_fc1")(v)
        v = layers.Dense(256, activation="relu", name="v_fc2")(v)
        v_out = layers.Dense(1, activation="tanh", name="value_out")(v)

        core = Model(inputs=[board_in, legal_in],
                     outputs=[policy_logits, v_out],
                     name="core_v12m")
        return core



