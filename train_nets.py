import os
import datetime
import numpy as np
import cupy as cp
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, LearningRateScheduler,
    EarlyStopping, TerminateOnNaN, TensorBoard, CSVLogger
)
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from RKmcp import make_model_inception
from make_data_train import NDCMultiPairGenerator
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.layers import (
    Input, Reshape, Permute, Conv1D, BatchNormalization, Activation, Add,
    Flatten, Dense, GlobalAveragePooling2D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model


def integer_to_binary_array(int_val, num_bits):
    return cp.array([int(i) for i in bin(int_val)[2:].zfill(num_bits)], dtype=cp.uint8).reshape(1, num_bits)

def cyclic_lr(num_epochs, high_lr, low_lr):
    return lambda i: low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr)

def save_model_architecture(model, filepath):
    """Save model architecture to JSON file"""
    config_path = filepath.replace('.weights.h5', '_architecture.json')
    with open(config_path, 'w') as f:
        f.write(model.to_json())
    return config_path

def load_model_from_weights(weights_path, architecture_path, custom_objects=None):
    """Load model from weights and architecture files"""
    from tensorflow.keras.models import model_from_json
    
    # Load architecture
    with open(architecture_path, 'r') as f:
        model = model_from_json(f.read(), custom_objects=custom_objects)
    
    # Load weights
    model.load_weights(weights_path)
    return model

lr_scheduler = LearningRateScheduler(cyclic_lr(num_epochs=10, high_lr=0.002, low_lr=0.0001))

# Ensure checkpoints directory exists
os.makedirs('checkpoints', exist_ok=True)

# Use weights-only checkpoint for smaller file size
checkpoint_cb = ModelCheckpoint(
    filepath=os.path.join('checkpoints', 'best_model.weights.h5'),
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True,  # Changed to True
    verbose=1
)

earlystop_cb = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True,
    verbose=1
)

terminate_nan_cb = TerminateOnNaN()

log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=0)
csv_logger_cb = CSVLogger(os.path.join(log_dir, "training.csv"))

callbacks = [
    checkpoint_cb,
    lr_scheduler,
    earlystop_cb,
    terminate_nan_cb,
    tensorboard_cb,
    csv_logger_cb
]

def update_checkpoint_in_callbacks(
    callbacks,
    rounds,
    cipher_name: str = "present80",
    run_id: str | None = None,
    save_dir: str = 'checkpoints',
    add_last_checkpoint: bool = True,
    save_weights_only: bool = True  # NEW: option to save weights only
):
    from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger

    # Prepare directories
    ckpt_dir = os.path.join(save_dir, cipher_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Best model per cipher and rounds - weights only for smaller files
    if save_weights_only:
        best_path = os.path.join(ckpt_dir, f"{cipher_name}_best_{rounds}r.weights.h5")
    else:
        best_path = os.path.join(ckpt_dir, f"{cipher_name}_best_{rounds}r.keras")
    
    new_best_ckpt = ModelCheckpoint(
        filepath=best_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=save_weights_only,
        verbose=1
    )

    # Replace existing 'best' checkpoint if present; otherwise append
    replaced = False
    for i, cb in enumerate(callbacks):
        if isinstance(cb, ModelCheckpoint) and getattr(cb, 'save_best_only', False):
            callbacks[i] = new_best_ckpt
            replaced = True
            break
    if not replaced:
        callbacks.append(new_best_ckpt)

    # Optionally add a 'last' checkpoint (weights only) saved every epoch for resume
    if add_last_checkpoint:
        last_path = os.path.join(ckpt_dir, f"{cipher_name}_last_{rounds}r.weights.h5")
        last_ckpt = ModelCheckpoint(
            filepath=last_path,
            save_best_only=False,
            save_weights_only=True,  # Always weights-only for frequent saves
            save_freq='epoch',
            verbose=0
        )
        callbacks.append(last_ckpt)

    # Configure per-run logs (TensorBoard + CSV)
    if run_id is None:
        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_root = os.path.join("logs", cipher_name, run_id)
    os.makedirs(log_root, exist_ok=True)

    # Replace/append TensorBoard
    tb_replaced = False
    for i, cb in enumerate(callbacks):
        if isinstance(cb, TensorBoard):
            callbacks[i] = TensorBoard(log_dir=log_root, histogram_freq=0)
            tb_replaced = True
            break
    if not tb_replaced:
        callbacks.append(TensorBoard(log_dir=log_root, histogram_freq=0))

    # Replace/append CSVLogger
    csv_replaced = False
    for i, cb in enumerate(callbacks):
        if isinstance(cb, CSVLogger):
            callbacks[i] = CSVLogger(os.path.join(log_root, f"training_{rounds}r.csv"))
            csv_replaced = True
            break
    if not csv_replaced:
        callbacks.append(CSVLogger(os.path.join(log_root, f"training_{rounds}r.csv")))

    return callbacks



def select_best_delta_key(
    encryption_function, input_difference,
    plain_bits, key_bits, n_round, pairs,
    n_samples=100_000, batch_size=5000, use_gpu=True
):
    delta_plain = integer_to_binary_array(input_difference, plain_bits)
    best_score = -1.0
    best_bit = -1
    all_scores = {}

    print("Searching for best delta_key (Hamming weight = 1):")

    for bit in range(key_bits):
        delta_key = np.zeros(key_bits, dtype=cp.uint8)
        delta_key[bit] = 1

        gen = NDCMultiPairGenerator(
            encryption_function=encryption_function,
            plain_bits=plain_bits, key_bits=key_bits, nr=n_round,
            delta_state=delta_plain,
            delta_key=delta_key,
            pairs=pairs,
            n_samples=n_samples, batch_size=batch_size,
            use_gpu=use_gpu, to_float32=True
        )

        X_val, Y_val = gen[0]

        try:
            pca = PCA(n_components=2).fit_transform(X_val)
            score = silhouette_score(pca, Y_val)
        except Exception:
            score = -1.0
        all_scores[bit] = score

        if score > best_score:
            best_score = score
            best_bit = bit

    print(f"\nBest delta_key bit: {best_bit} with score = {best_score:.5f}")
    return best_bit, best_score, all_scores


def train_by_chunks(
    model,
    encrypt,
    plain_bits,
    key_bits,
    n_round,
    pairs,
    delta_plain,
    delta_key,
    total_samples,
    chunk_size,
    batch_size,
    epochs,
    val_samples=1_000_000,
    callbacks=None,
):
    """
    Chunk-based training with GLOBAL-EPOCH callbacks compatibility.
    """

    n_chunks = (total_samples + chunk_size - 1) // chunk_size
    callbacks = callbacks or []

    # --- build fixed validation generator ---
    val_gen = NDCMultiPairGenerator(
        encrypt,
        plain_bits,
        key_bits,
        n_round,
        delta_state=delta_plain,
        delta_key=delta_key,
        n_samples=val_samples,
        batch_size=batch_size,
        pairs=pairs,
    )

    # --- callback lifecycle: on_train_begin ---
    for cb in callbacks:
        cb.set_model(model)
        cb.on_train_begin()

    history = {"loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        print(f"\n========== Global Epoch {epoch+1}/{epochs} ==========")

        for cb in callbacks:
            if isinstance(cb, LearningRateScheduler):
                lr = cb.schedule(epoch)
                if hasattr(model.optimizer.learning_rate, "assign"):
                    model.optimizer.learning_rate.assign(lr)
                else:
                    model.optimizer.learning_rate = lr
                print(f"[LR] set to {lr:.6e}")

        # ---- training chunks ----
        for c in range(n_chunks):
            print(f"--- Chunk {c+1}/{n_chunks} ---")

            gen_chunk = NDCMultiPairGenerator(
                encrypt,
                plain_bits,
                key_bits,
                n_round,
                delta_state=delta_plain,
                delta_key=delta_key,
                n_samples=chunk_size,
                batch_size=batch_size,
                pairs=pairs,
            )

            h = model.fit(gen_chunk, epochs=1, verbose=1)

            # accumulate training loss
            if "loss" in h.history:
                history["loss"].extend(h.history["loss"])

        # ---- validation ----
        val_loss, val_acc = model.evaluate(val_gen, verbose=0)
        print(f"[Validation] loss={val_loss:.5f}, acc={val_acc:.5f}")

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logs = {
            "loss": history["loss"][-1],
            "val_loss": val_loss,
            "val_acc": val_acc,
        }

        # ---- callbacks: on_epoch_end ----
        stop_training = False
        for cb in callbacks:
            cb.on_epoch_end(epoch, logs)
            if getattr(model, "stop_training", False):
                stop_training = True

        if stop_training:
            print("⛔ Early stopping triggered by callback")
            break

    # --- callbacks: on_train_end ---
    for cb in callbacks:
        cb.on_train_end()

    return history
