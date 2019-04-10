import numpy as np
import matplotlib.pyplot as plt
import keras
import keras.datasets as kd
import keras.models as km
import keras.layers as kl

(x_train, y_train), (x_test, y_test) = kd.cifar10.load_data()
labels = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def take_subset_based_on_labels(X,y,indices_to_keep):
    boolean_mask = np.logical_or.reduce([y==i for i in indices_to_keep]).squeeze()
    for i,j in enumerate(indices_to_keep):
        y[y==j]=i
    return X[boolean_mask]/255.,y[boolean_mask]

new_labels = [labels[i] for i in [3,4,8]]
x_train,y_train = take_subset_based_on_labels(x_train,y_train,[3,4,8])
x_test,y_test = take_subset_based_on_labels(x_test,y_test,[3,4,8])

# Convert class vectors to binary class matrices.
N = len(new_labels)

y_train = keras.utils.to_categorical(y_train, N)
y_test = keras.utils.to_categorical(y_test, N)

model = km.Sequential()

conv_1 = kl.Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:])
model.add(conv_1)
act_1 = kl.Activation('relu')
model.add(act_1)
bn_1 = kl.BatchNormalization()
model.add(bn_1)
mp_1 = kl.MaxPooling2D(pool_size=(2,2))
model.add(mp_1)
# Convolution with 64 kernels
conv_2 = kl.Conv2D(64, (3, 3), padding='same')
model.add(conv_2)

# Activation with ReLU
act_2 = kl.Activation('relu')
model.add(act_2)

# Normalization of output
bn_2 = kl.BatchNormalization()
model.add(bn_2)

# Downsampling with max pooling
mp_2 = kl.MaxPooling2D(pool_size=(2,2))
model.add(mp_2)

# Convolution with 32 kernels
conv_3 = kl.Conv2D(32, (3, 3), padding='same')
model.add(conv_3)

# Activation with ReLU
act_3 = kl.Activation('relu')
model.add(act_3)

# Normalization of output
bn_3 = kl.BatchNormalization()
model.add(bn_3)


gap = kl.GlobalAveragePooling2D()
model.add(gap)

bn_4 = kl.BatchNormalization()
v = model.add(bn_4)

final_dense = kl.Dense(N)
model.add(final_dense)

softmax = kl.Activation('softmax')
model.add(softmax)

training_data_predictions = model.predict(x_train)

# initiate adam optimizer
opt = keras.optimizers.adam(lr=0.001)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=64,
          epochs=10,
          validation_data=(x_test, y_test),
          shuffle=True)


