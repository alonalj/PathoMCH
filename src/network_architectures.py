from tensorflow.python import keras as backend


def inception_keras(c):
    from inception import InceptionV3
    c.network_name = 'inception'
    model_input = backend.layers.Input(shape=(c.IMG_SIZE, c.IMG_SIZE, c.N_CHANNELS))
    base_model = InceptionV3(shape=(c.IMG_SIZE, c.IMG_SIZE, c.N_CHANNELS),
                             input_tensor=model_input,
                             include_top=False)
    x = base_model.output
    x = backend.layers.GlobalMaxPool2D()(x)
    out_activation = backend.layers.Dense(1, activation='sigmoid', name='sigmoid_activation_2class')(x)

    model = backend.models.Model(inputs=base_model.input, outputs=out_activation)
    return model
