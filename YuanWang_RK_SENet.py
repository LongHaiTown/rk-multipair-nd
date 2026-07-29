"""Yuan-Wang style residual SENet baseline for the current RK multi-pair codebase.

Expected generator output
-------------------------
Each sample is a flat vector containing ``pairs`` triples
``[DeltaC || C || C*]``.  Therefore::

    input_dim = pairs * 3 * plain_bits

The flat vector is converted inside the model as follows::

    (batch, pairs * 3 * plain_bits)
      -> (batch, pairs, 3, plain_bits)
      -> (batch, pairs, plain_bits, 3)
      -> (batch, pairs * plain_bits, 3)

This preserves the batch axis and lets Conv1D operate along the aggregated
pair/bit axis with the three ciphertext representations as channels.
"""

from __future__ import annotations

from typing import Optional

from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    Input,
    Multiply,
    Permute,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def _positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def senet_module(
    input_tensor,
    reduction_ratio: int = 8,
    min_hidden: int = 8,
    reg_param: float = 1e-5,
    name_prefix: str = "senet",
):
    """Squeeze-and-Excitation reweighting for a 1D feature map."""
    channels = input_tensor.shape[-1]
    if channels is None:
        raise ValueError("SENet requires a statically known channel dimension.")
    channels = int(channels)

    reduction_ratio = _positive_int("reduction_ratio", reduction_ratio)
    min_hidden = _positive_int("min_hidden", min_hidden)
    hidden = max(min_hidden, channels // reduction_ratio)

    se = GlobalAveragePooling1D(name=f"{name_prefix}_gap")(input_tensor)
    se = Dense(
        hidden,
        activation="relu",
        kernel_regularizer=l2(reg_param),
        name=f"{name_prefix}_reduce",
    )(se)
    se = Dense(
        channels,
        activation="sigmoid",
        kernel_regularizer=l2(reg_param),
        name=f"{name_prefix}_expand",
    )(se)
    se = Reshape((1, channels), name=f"{name_prefix}_reshape")(se)
    return Multiply(name=f"{name_prefix}_scale")([input_tensor, se])


def module1_initial_embedding(
    x,
    num_filters: int = 64,
    dense_width: int = 128,
    reg_param: float = 1e-5,
):
    """Initial pointwise embedding used by the Yuan-Wang style baseline."""
    x = Conv1D(
        num_filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(reg_param),
        name="module1_conv1x1",
    )(x)
    x = BatchNormalization(name="module1_conv_bn")(x)
    x = Activation("relu", name="module1_conv_relu")(x)

    # Dense on rank-3 tensors acts independently at every sequence position.
    x = Dense(
        dense_width,
        use_bias=False,
        kernel_regularizer=l2(reg_param),
        name="module1_dense_1",
    )(x)
    x = BatchNormalization(name="module1_dense_1_bn")(x)
    x = Activation("relu", name="module1_dense_1_relu")(x)

    x = Dense(
        dense_width,
        use_bias=False,
        kernel_regularizer=l2(reg_param),
        name="module1_dense_2",
    )(x)
    x = BatchNormalization(name="module1_dense_2_bn")(x)
    x = Activation("relu", name="module1_dense_2_relu")(x)
    return x


def module2_residual_senet_block(
    x,
    block_idx: int,
    num_filters: int = 64,
    kernel_size: int = 3,
    se_reduction_ratio: int = 8,
    reg_param: float = 1e-5,
):
    """Residual Conv1D block followed by channel-wise SENet attention."""
    shortcut = x

    y = Conv1D(
        num_filters,
        kernel_size=kernel_size,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(reg_param),
        name=f"module2_{block_idx}_conv1",
    )(x)
    y = BatchNormalization(name=f"module2_{block_idx}_bn1")(y)
    y = Activation("relu", name=f"module2_{block_idx}_relu1")(y)

    y = Conv1D(
        num_filters,
        kernel_size=kernel_size,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(reg_param),
        name=f"module2_{block_idx}_conv2",
    )(y)
    y = BatchNormalization(name=f"module2_{block_idx}_bn2")(y)

    y = senet_module(
        y,
        reduction_ratio=se_reduction_ratio,
        reg_param=reg_param,
        name_prefix=f"module2_{block_idx}_senet",
    )

    shortcut_channels = shortcut.shape[-1]
    if shortcut_channels is None or int(shortcut_channels) != int(num_filters):
        shortcut = Conv1D(
            num_filters,
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_regularizer=l2(reg_param),
            name=f"module2_{block_idx}_shortcut_projection",
        )(shortcut)
        shortcut = BatchNormalization(name=f"module2_{block_idx}_shortcut_bn")(shortcut)

    y = Add(name=f"module2_{block_idx}_residual_add")([shortcut, y])
    return Activation("relu", name=f"module2_{block_idx}_out_relu")(y)


def module3_prediction_head(
    x,
    dense_width: int = 128,
    dropout_rate: float = 0.3,
    reg_param: float = 1e-5,
    num_outputs: int = 1,
    final_activation: str = "sigmoid",
):
    """Flatten-based prediction head retained for baseline fidelity."""
    x = Flatten(name="module3_flatten")(x)
    x = Dropout(dropout_rate, name="module3_dropout")(x)

    for idx in (1, 2):
        x = Dense(
            dense_width,
            use_bias=False,
            kernel_regularizer=l2(reg_param),
            name=f"module3_dense_{idx}",
        )(x)
        x = BatchNormalization(name=f"module3_dense_{idx}_bn")(x)
        x = Activation("relu", name=f"module3_dense_{idx}_relu")(x)

    return Dense(
        num_outputs,
        activation=final_activation,
        kernel_regularizer=l2(reg_param),
        name="prediction",
    )(x)


def make_model_yuan_wang(
    input_dim: Optional[int] = None,
    pairs: int = 8,
    plain_bits: int = 64,
    feature_channels: int = 3,
    num_filters: int = 64,
    module1_dense_width: int = 128,
    head_dense_width: int = 128,
    residual_blocks: int = 2,
    kernel_size: int = 3,
    se_reduction_ratio: int = 8,
    dropout_rate: float = 0.3,
    reg_param: float = 1e-5,
    num_outputs: int = 1,
    final_activation: str = "sigmoid",
) -> Model:
    """Build a Yuan-Wang style RK-SENet for ``NDCMultiPairGenerator``.

    The current codebase uses three fields per pair: ``DeltaC``, ``C`` and
    ``C*``.  Consequently, ``feature_channels`` should normally remain 3.
    ``input_dim`` may be supplied explicitly, but it must agree with
    ``pairs * feature_channels * plain_bits``.
    """
    pairs = _positive_int("pairs", pairs)
    plain_bits = _positive_int("plain_bits", plain_bits)
    feature_channels = _positive_int("feature_channels", feature_channels)
    num_filters = _positive_int("num_filters", num_filters)
    module1_dense_width = _positive_int("module1_dense_width", module1_dense_width)
    head_dense_width = _positive_int("head_dense_width", head_dense_width)
    residual_blocks = _positive_int("residual_blocks", residual_blocks)
    kernel_size = _positive_int("kernel_size", kernel_size)
    num_outputs = _positive_int("num_outputs", num_outputs)

    expected_dim = pairs * feature_channels * plain_bits
    if input_dim is None:
        input_dim = expected_dim
    else:
        input_dim = _positive_int("input_dim", input_dim)
        if input_dim != expected_dim:
            raise ValueError(
                "input_dim does not match the current RK multi-pair representation: "
                f"got input_dim={input_dim}, but pairs * feature_channels * plain_bits "
                f"= {pairs} * {feature_channels} * {plain_bits} = {expected_dim}."
            )

    inp = Input(shape=(input_dim,), name="input_bits")

    # Generator flattening convention: (pairs, 3, plain_bits).
    x = Reshape(
        (pairs, feature_channels, plain_bits),
        name="reshape_pairs_channels_bits",
    )(inp)
    x = Permute((1, 3, 2), name="permute_pairs_bits_channels")(x)
    x = Reshape(
        (pairs * plain_bits, feature_channels),
        name="reshape_sequence_channels",
    )(x)

    x = module1_initial_embedding(
        x,
        num_filters=num_filters,
        dense_width=module1_dense_width,
        reg_param=reg_param,
    )

    for block_idx in range(1, residual_blocks + 1):
        x = module2_residual_senet_block(
            x,
            block_idx=block_idx,
            num_filters=num_filters,
            kernel_size=kernel_size,
            se_reduction_ratio=se_reduction_ratio,
            reg_param=reg_param,
        )

    out = module3_prediction_head(
        x,
        dense_width=head_dense_width,
        dropout_rate=dropout_rate,
        reg_param=reg_param,
        num_outputs=num_outputs,
        final_activation=final_activation,
    )

    return Model(inputs=inp, outputs=out, name="YuanWang_RK_SENet_ND")


def make_multipair_senet(
    plain_bits: int,
    pairs: int,
    **kwargs,
) -> Model:
    """Convenience factory matching the other model modules in the codebase."""
    return make_model_yuan_wang(
        plain_bits=plain_bits,
        pairs=pairs,
        feature_channels=3,
        **kwargs,
    )


# Backward-friendly aliases used by different training scripts.
make_yuan_wang_model = make_model_yuan_wang
make_model_yuanwang = make_model_yuan_wang
make_model = make_model_yuan_wang
build_model = make_model_yuan_wang