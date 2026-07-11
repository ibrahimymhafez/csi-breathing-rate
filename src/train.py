from pandas.io.parsers import readers
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_processing import X_train, y_train
from model import model

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=25,          # increased from 10
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,           # halve the LR when stuck
    patience=5,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,        # larger batch for stability
    validation_split=0.15,
    callbacks=[early_stop, reduce_lr]
)
