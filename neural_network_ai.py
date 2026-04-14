import tensorflow as tf
from tensorflow.keras import layers, models

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train, X_test = X_train / 255.0, X_test / 255.0

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),   
    layers.Dense(128, activation='relu'),    
    layers.Dropout(0.2),                    
    layers.Dense(10, activation='softmax')   
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("AI Training shuru hochche...")
model.fit(X_train, y_train, epochs=5)

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
print(f'\nTest Accuracy: {test_acc*100:.2f}%')