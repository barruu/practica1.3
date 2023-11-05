rom keras.models import Sequential
from keras.layers import Dense

# Crear el model
model = Sequential()

# Afegir la capa d'entrada amb 2 nodes
model.add(Dense(2, input_dim=2, activation='relu'))

model.add(Dense(3, activation='relu'))

# Afegir la capa de sortida amb 2 nodes
model.add(Dense(2, activation='sigmoid'))

# Compilar el model amb l'optimitzador 'adam' i la pèrdua 'binary_crossentropy' ja que és un problema de classificació binària
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Dades d'entrada per a la XOR i AND
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y_xor = [[0, 1], [1, 0], [1, 0], [0, 1]]  # Sortida desitjada per XOR
Y_and = [[0, 0], [0, 1], [0, 1], [1, 0]]  # Sortida desitjada per AND

# Entrenar la xarxa per a la XOR
model.fit(X, Y_xor, epochs=1000, verbose=0)

# Avaluar el rendiment per a la XOR
print("Avaluació per XOR:")
print(model.predict(X))

# Reinicialitzar els pesos del model per a l'entrenament de la AND
model.set_weights([[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], [0, 0, 0], [[1, -1], [-1, 1], [1, 1]], [0, 0]])

# Entrenar la xarxa per a la AND
model.fit(X, Y_and, epochs=1000, verbose=0)

# Avaluar el rendiment per a la AND
print("\nAvaluació per AND:")
print(model.predict(X))