from re import I
from typing import Any, Dict, Tuple

import tensorflow as tf
from data_utils import prepare_datasets


def create_shakespeare_model0(
    vocab_size: int,
    stateful: bool = False,
    embedding_dim: int = 16,
    gru_units: int = 128,
    batch_size: int = 32,
) -> tf.keras.Model:
    """
    Creates a Shakespeare-style text generation model.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the embedding layer. Default is 16.
        gru_units (int): Number of units in the GRU layer. Default is 128.

    Returns:
        tf.keras.Model: The created Shakespeare model.
    """

    return tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                # input_shape=(batch_size, None),
            ),
            tf.keras.layers.GRU(gru_units, return_sequences=True, stateful=stateful),
            tf.keras.layers.Dense(vocab_size, activation="softmax"),
        ]
    )


def create_shakespeare_model(
    vocab_size: int,
    stateful: bool = False,
    embedding_dim: int = 16,
    gru_units: int = 128,
    batch_size: int = 1,
) -> tf.keras.Model:
    """
    Creates a Shakespeare-style text generation model.

    Args:
        vocab_size (int): Size of the vocabulary.
        stateful (bool): Whether to use stateful RNN. Default is False.
        embedding_dim (int): Dimension of the embedding layer. Default is 16.
        gru_units (int): Number of units in the GRU layer. Default is 128.
        batch_size (int): Batch size for stateful RNNs. Default is 32.

    Returns:
        tf.keras.Model: The created Shakespeare model.
    """
    inputs = tf.keras.Input(
        shape=(None,),  dtype=tf.int32, batch_size=batch_size if stateful else None,
    )
    x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(
        inputs
    )
    # tf.print(stateful, batch_size)
    # tf.print("Shape after Embedding:", x.shape)  # Debug print
    x = tf.keras.layers.GRU(gru_units, return_sequences=True, stateful=stateful)(x)
    # tf.print("Shape after GRU:", x.shape)  # Debug print
    outputs = tf.keras.layers.Dense(vocab_size, activation="softmax")(x)
    # tf.print("Shape after Dense:", outputs.shape)  # Debug print
    return tf.keras.Model(inputs, outputs)


def compile_model(model: tf.keras.Model, learning_rate: float = 0.001) -> None:
    """
    Compiles the given model.

    Args:
        model (tf.keras.Model): The model to compile.
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.
    """
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )


def train_shakespeare_model(
    model: tf.keras.Model,
    train_set: tf.data.Dataset,
    valid_set: tf.data.Dataset,
    stateful: bool = False,
    epochs: int = 10,
    model_name: str = "shakespeare_model",
) -> tf.keras.callbacks.History:
    """
    Trains the Shakespeare-style text generation model.

    Args:
        model (tf.keras.Model): The model to train.
        train_set (tf.data.Dataset): Training dataset.
        valid_set (tf.data.Dataset): Validation dataset.
        epochs (int): Number of training epochs. Default is 10.
        model_name (str): Name for saving the best model. Default is "shakespeare_model".

    Returns:
        tf.keras.callbacks.History: Training history.
    """

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            f"{model_name}.keras",
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
    ]

    class FlexibleResetStatesCallback(tf.keras.callbacks.Callback):
        '''flexible callback that allows for conditional state resetting'''
        def __init__(self, reset_frequency=1):
            super().__init__()
            self.reset_frequency = reset_frequency

        def on_epoch_begin(self, epoch, logs=None):
            if epoch % self.reset_frequency == 0:
                if hasattr(self.model, "reset_states"):
                    self.model.reset_states()
                else:
                    for layer in self.model.layers:
                        if hasattr(layer, "reset_states"):
                            layer.reset_states()
                print(f"States reset at epoch {epoch}")

    class ResetStatesCallback(tf.keras.callbacks.Callback):
        '''Prevent unintended state carryover between unrelated sequences or at logical boundaries (like epoch ends).'''
        def on_epoch_begin(self, epoch, logs=None):
            if hasattr(self.model, "reset_states"):
                self.model.reset_states()
            else:
                # If the model doesn't have reset_states, try to reset states of RNN layers
                for layer in self.model.layers:
                    if hasattr(layer, "reset_states"):
                        layer.reset_states()

    if stateful:
        callbacks.append(ResetStatesCallback())

    return model.fit(
        train_set,
        validation_data=valid_set,
        epochs=epochs,
        callbacks=callbacks,
    )


def create_and_train_shakespeare_model(
    raw_text: str,
    model_params: Dict[str, Any] = None,
    training_params: Dict[str, Any] = None,
) -> Tuple[
    tf.keras.Model,
    tf.keras.callbacks.History,
    tf.data.Dataset,
    tf.keras.layers.TextVectorization,
]:
    """
    Creates and trains a Shakespeare-style text generation model.

    Args:
        raw_text (str): The raw text data to train on.
        model_params (Dict[str, Any]): Parameters for model creation. Defaults to None.
        training_params (Dict[str, Any]): Parameters for model training. Defaults to None.

    Returns:
        Tuple[tf.keras.Model, tf.keras.callbacks.History, tf.data.Dataset, tf.keras.layers.TextVectorization]:
        A tuple containing the trained model, training history, test dataset, and text vectorization layer.
    """
    model_params = model_params or {}
    training_params = training_params or {}

    stateful = training_params.get("stateful", False)
    tf.random.set_seed(training_params.get("seed", 42))

    train_set, valid_set, test_set, text_vec_layer = prepare_datasets(
        raw_text,
        length=training_params.get("sequence_length", 100),
        batch_size=training_params.get("batch_size", 32),
        stateful=training_params.get("stateful", False),
    )

    vocab_size = text_vec_layer.vocabulary_size()

    model = create_shakespeare_model(
        vocab_size,
        stateful=stateful,
        embedding_dim=model_params.get("embedding_dim", 16),
        gru_units=model_params.get("gru_units", 128),
        batch_size=training_params.get("batch_size", 32),
    )

    compile_model(model, learning_rate=training_params.get("learning_rate", 0.001))
    history = train_shakespeare_model(
        model,
        train_set,
        valid_set,
        stateful=stateful,
        epochs=training_params.get("epochs", 10),
        model_name=training_params.get("model_name", "shakespeare_model"),
    )

    return model, history, test_set, text_vec_layer


# save the model
