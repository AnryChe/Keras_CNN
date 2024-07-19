import pandas as pd
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os

batch_size = 512
num_classes = 10
epochs = 5
data_augmentation = False
num_predictions = 20

filters_quantity = [12, 24, 48]
dense_quantity = [4, 6]

summ_info = pd.DataFrame(columns=['CNN name', 'Filters', 'Layers', 'Test loss', 'Test accuracy'])

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'тренировочные примеры')
print(X_test.shape[0], 'тестовые примеры')

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

X_train = X_train / 255.0
X_test = X_test / 255.0

for f_i in filters_quantity:
    for d_i in dense_quantity:
        cnn_name = f'Model_{f_i}x{d_i}'
        model = keras.Sequential(name=cnn_name)

        for i in range(d_i - 1):
            max_dense = len(range(d_i))
            model.add(Conv2D(f_i, (max_dense - i, max_dense - i), padding='same', input_shape=X_train.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        # print(model.summary())
        model.compile(loss='categorical_crossentropy',
                      optimizer='SGD',
                      metrics=['accuracy'])
        print(f'{cnn_name} is compiled')
        model.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=0,
                  validation_data=(X_test, y_test),
                  shuffle=True)
        scores = model.evaluate(X_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
        summ_info.loc[len(summ_info.index)] = [cnn_name, f_i, d_i, scores[0], scores[1]]

print(summ_info.sort_values('Filters'))
print(summ_info.sort_values('Layers'))
