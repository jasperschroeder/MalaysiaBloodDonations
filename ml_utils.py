import mlflow.tensorflow
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout, Input, SimpleRNN, Concatenate, GRU
from keras.models import Model
from keras.optimizers import Adam, RMSprop
import mlflow
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import warnings
warnings.filterwarnings("ignore")


def create_early_stopping(monitor:str, patience:int, restore_best_weights:bool=True, **kwargs):
    return EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=restore_best_weights, **kwargs)


def get_or_create_mlflow_experiment(experiment_name: str):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        return experiment_id
    return experiment.experiment_id


def train_val_split_lstm_feature_data(
    X_seq: np.ndarray,
    X_features: np.ndarray,
    y: np.ndarray,
    train_frac: float,
    val_frac: float
):
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be less than 1")
    if not (0 < train_frac < 1) or not (0 < val_frac < 1):
        raise ValueError("train_frac and val_frac must be between 0 and 1")

    
    train_size = int(len(X_seq) * train_frac)
    val_size = int(len(X_seq) * val_frac)
    
    X_seq_train = X_seq[:train_size]
    X_seq_val = X_seq[train_size:(train_size + val_size)]
    X_seq_test = X_seq[(train_size + val_size):]
    
    X_features_train = X_features[:train_size]
    X_features_val = X_features[train_size:(train_size + val_size)]
    X_features_test = X_features[(train_size + val_size):]
    
    y_train = y[:train_size]
    y_val = y[train_size:(train_size + val_size)]
    y_test = y[(train_size + val_size):]
    
    return X_seq_train, X_seq_val, X_seq_test, X_features_train, X_features_val, X_features_test, y_train, y_val, y_test


def build_seq_model(seq_shape, features_shape, seq_type: str, seq_units: int, dense_units: int, activation: str, 
                dropout: float, optimizer: str, learning_rate: float = 0.001, loss:str='mse', metrics:list=['mae']
):
    
    # inputs 
    seq_input = Input(shape=(seq_shape, 1))
    features_input = Input(shape=(features_shape,))
    
    # seq branch
    if seq_type == 'LSTM':
        x_seq = LSTM(seq_units, activation=activation, return_sequences=False)(seq_input)
    elif seq_type == 'GRU':
        x_seq = GRU(seq_units, activation=activation, return_sequences=False)(seq_input)
    else:
        x_seq = SimpleRNN(seq_units, activation=activation, return_sequences=False)(seq_input)    
    x_seq = Dropout(dropout)(x_seq)
    
    # Dense branch 
    x = Concatenate()([x_seq, features_input])
    x = Dense(dense_units, activation='relu')(x) # Setting relu as default activation for dense layers
    x = Dropout(dropout)(x)
    output = Dense(1)(x)
    
    # Selecting optimizer with learning rate
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        opt = optimizer
    
    model = Model(inputs=[seq_input, features_input], outputs=output)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    
    return model
    

def run_experiment(
    X_seq_train, X_features_train, y_train, X_seq_val, X_features_val, y_val, 
    seq_type:str, seq_units:int, dense_units:int, activation: str, dropout: float, 
    optimizer: str, learning_rate: float, batch_size: int, experiment_id: str
):

    mlflow.tensorflow.autolog(log_models=True, log_datasets=False, silent=True)

    with mlflow.start_run(experiment_id=experiment_id):
        early_stopping = create_early_stopping(monitor='val_loss', patience=25)
        
        model = build_seq_model(
            seq_shape=X_seq_train.shape[1], features_shape=X_features_train.shape[1], seq_type=seq_type,
            seq_units=seq_units, dense_units=dense_units, activation=activation, dropout=dropout,
            optimizer=optimizer, learning_rate=learning_rate
        )
                    
        _ = model.fit(
            [X_seq_train, X_features_train],
            y_train,
            validation_data=([X_seq_val, X_features_val], y_val),
            epochs=1000,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=None,
            shuffle=False
        )
        
        val_loss = model.evaluate([X_seq_val, X_features_val], y_val)
        print(f"Validation Loss: {val_loss[0]}, Validation MAE: {val_loss[1]}")
        
        mlflow.log_param("seq_type", seq_type)
        mlflow.log_param("seq_units", seq_units)
        mlflow.log_param("dense_units", dense_units)
        mlflow.log_param("activation", activation)
        mlflow.log_param("optimizer", optimizer)
        mlflow.log_param("dropout", dropout)
        

def get_best_model(experiment_id, metric:str="metrics.val_loss"):

    model_dfs = mlflow.search_runs(experiment_id)
    best_run_id = model_dfs.sort_values(metric, ascending=True)['run_id'].iloc[0]
    return mlflow.tensorflow.load_model(f"runs:/{best_run_id}/model")
