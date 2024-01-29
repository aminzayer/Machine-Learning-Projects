import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# بارگیری داده‌های MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# مقیاس دهی داده‌ها
X_train = X_train / 255.0
X_test = X_test / 255.0

# تبدیل برچسب‌ها به فرمت one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# ساخت مدل
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# کامپایل مدل
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# آموزش مدل
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# ارزیابی دقت مدل
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)
