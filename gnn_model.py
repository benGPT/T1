import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import networkx as nx
from spektral.layers import GCNConv
from spektral.data import Graph

def create_gnn_model(n_features):
    model = tf.keras.Sequential([
        GCNConv(32, activation='relu'),
        GCNConv(16, activation='relu'),
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

def create_graph(X):
    graphs = []
    for window in X:
        G = nx.Graph()
        for i in range(len(window)):
            G.add_node(i, features=window[i])
        for i in range(len(window)):
            for j in range(i+1, len(window)):
                G.add_edge(i, j)
        A = nx.adjacency_matrix(G).todense()
        N = window.shape[0]
        F = window.shape[1]
        graphs.append(Graph(x=window, a=A))
    return graphs

def train_gnn_model(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']].values)

    X, y = prepare_data(scaled_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    graphs_train = create_graph(X_train)
    graphs_test = create_graph(X_test)

    model = create_gnn_model(X_train.shape[2])

    history = model.fit([g.x for g in graphs_train], y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    return model, history

def predict_gnn(model, data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']].values)

    X, _ = prepare_data(scaled_data)
    graphs = create_graph(X)

    predictions = model.predict([g.x for g in graphs])

    return scaler.inverse_transform(predictions)[:, 0]

