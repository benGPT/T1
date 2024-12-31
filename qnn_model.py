import pennylane as qml
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def create_qnn_model(n_qubits, n_layers):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        for i in range(n_layers):
            qml.StronglyEntanglingLayers(weights[i], wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}

    qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)

    model = tf.keras.models.Sequential([
        qlayer,
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_data(data, window_size=5):
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def train_qnn_model(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']].values)

    X, y = prepare_data(scaled_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    n_qubits = 5
    n_layers = 2
    model = create_qnn_model(n_qubits, n_layers)

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    return model, history

def predict_qnn(model, data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']].values)

    X, _ = prepare_data(scaled_data)
    predictions = model.predict(X)

    return scaler.inverse_transform(predictions)[:, 0]

