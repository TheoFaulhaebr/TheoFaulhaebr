import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Definieren Sie das Modell
class ThioAI(keras.Model):
    def __init__(self):
        super(ThioAI, self).__init__()
        self.layer1 = layers.Dense(64, activation='relu')
        self.layer2 = layers.Dense(32, activation='relu')
        self.output_layer = layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return self.output_layer(x)

# Instanziieren Sie das Modell
thio = ThioAI()

# Kompilieren Sie das Modell
thio.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# Trainieren Sie das Modell (Beispiel mit Dummy-Daten)
x_train = tf.random.normal([100, 30])
y_train = tf.random.uniform([100], maxval=10, dtype=tf.int32)

thio.fit(x_train, y_train, epochs=10)

# Modellbewertung (wieder mit Dummy-Daten)
x_test = tf.random.normal([20, 30])
y_test = tf.random.uniform([20], maxval=10, dtype=tf.int32)

test_loss, test_accuracy = thio.evaluate(x_test, y_test)
print(f'Testgenauigkeit: {test_accuracy:.4f}')

