import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


MODEL_CONFIGS = {
    "atto": {"depths": [2, 2, 6, 2], "dim": [40, 80, 160, 320]},
    "femto": {"depths": [2, 2, 6, 2], "dim": [48, 96, 192, 384]},
    "pico": {"depths": [2, 2, 6, 2], "dim": [64, 128, 256, 512]},
    "nano": {"depths": [2, 2, 8, 2], "dim": [80, 160, 320, 640]},
    "tiny": {"depths": [3, 3, 9, 3], "dim": [96, 192, 384, 768]},
    "small": {"depths": [3, 3, 27, 3], "dim": [96, 192, 384, 768]},
    "base": {"depths": [3, 3, 27, 3], "dim": [128, 256, 512, 1024]},
    "large": {"depths": [3, 3, 27, 3], "dim": [192, 384, 768, 1536]},
    "huge": {"depths": [3, 3, 27, 3], "dim": [352, 704, 1408, 2816]},
}


class GRN(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.epsilon = 1e-6

    def build(self, input_shape):

        self.gamma = self.add_weight(
            name="gamma",
            shape=(1, 1, 1, input_shape[-1]),
            initializer=tf.zeros_initializer(),
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(1, 1, 1, input_shape[-1]),
            initializer=tf.zeros_initializer(),
        )

    def call(self, inputs):
        gamma = tf.cast(self.gamma, tf.float32)
        beta = tf.cast(self.beta, tf.float32)
        x = tf.cast(inputs, tf.float32)

        Gx = tf.pow(
            (tf.reduce_sum(tf.pow(x, 2), axis=(1, 2), keepdims=True) + self.epsilon),
            0.5,
        )
        Nx = Gx / tf.reduce_mean(Gx, axis=-1, keepdims=True) + self.epsilon

        result = gamma * (x * Nx) + beta + x
        return tf.cast(result, inputs.dtype)


class StochasticDepth(layers.Layer):
    """Stochastic Depth module.
    It is also referred to as Drop Path in `timm`.
    References:
        (1) github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_path = drop_path

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


class Block(tf.keras.layers.Layer):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = layers.Conv2D(dim, kernel_size=7, padding="same", groups=dim)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.pwconv1 = layers.Dense(dim * 4)
        self.act = layers.Activation("gelu")
        self.pwconv2 = GRN()
        if layer_scale_init_value > 0:
            self.gamma = tf.Variable(layer_scale_init_value * tf.ones((dim,)))
        else:
            self.gamma = None

        self.drop_path = (
            StochasticDepth(drop_path)
            if drop_path > 0.0
            else layers.Activation("linear")
        )

    def call(self, x):
        inputs = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x

        x = inputs + self.drop_path(x)
        return x


def ConvNeXt(
    drop_path_rate,
    layer_scale_init_value,
    num_classes=1000,
    input_shape=(None, None, 3),
    depths=None,
    dims=None,
):
    inputs = layers.Input(input_shape)
    x = inputs

    stem = keras.Sequential(
        [
            layers.Conv2D(dims[0], kernel_size=4, strides=4),
            layers.LayerNormalization(epsilon=1e-6),
        ]
    )
    downsample_layers = []
    downsample_layers.append(stem)

    for i in range(3):
        downsample_layer = keras.Sequential(
            [
                layers.LayerNormalization(epsilon=1e-6),
                layers.Conv2D(dims[i + 1], kernel_size=2, strides=2),
            ]
        )
        downsample_layers.append(downsample_layer)

    dp_rates = [x for x in tf.linspace(0.0, drop_path_rate, sum(depths))]
    cur = 0
    for i in range(4):
        x = downsample_layers[i](x)
        for j in range(depths[i]):
            x = Block(
                dim=dims[i],
                drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value,
            )(x)
        cur += depths[i]

    x = layers.GlobalAvgPool2D()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    outputs = layers.Dense(num_classes)(x)
    return keras.Model(inputs, outputs)


def ConvNeXt_tiny(input_shape, num_classes=1000, drop_path_rate=0.0, layer_scale_init_value=1e-6):
    model = ConvNeXt(
        drop_path_rate=drop_path_rate,
        layer_scale_init_value=layer_scale_init_value,
        num_classes=num_classes,
        input_shape=input_shape,
        depths=MODEL_CONFIGS["tiny"]["depths"],
        dims=MODEL_CONFIGS["tiny"]["dim"],
    )

    return model


def ConvNeXt_small(input_shape, num_classes=1000, drop_path_rate=0.0, layer_scale_init_value=1e-6):
    model = ConvNeXt(
        drop_path_rate=drop_path_rate,
        layer_scale_init_value=layer_scale_init_value,
        num_classes=num_classes,
        input_shape=input_shape,
        depths=MODEL_CONFIGS["small"]["depths"],
        dims=MODEL_CONFIGS["small"]["dim"],
    )

    return model


def ConvNeXt_base(input_shape, num_classes=1000, drop_path_rate=0.0, layer_scale_init_value=1e-6):
    model = ConvNeXt(
        drop_path_rate=drop_path_rate,
        layer_scale_init_value=layer_scale_init_value,
        num_classes=num_classes,
        input_shape=input_shape,
        depths=MODEL_CONFIGS["base"]["depths"],
        dims=MODEL_CONFIGS["base"]["dim"],
    )

    return model


def ConvNeXt_large(input_shape, num_classes=1000, drop_path_rate=0.0, layer_scale_init_value=1e-6):
    model = ConvNeXt(
        drop_path_rate=drop_path_rate,
        layer_scale_init_value=layer_scale_init_value,
        num_classes=num_classes,
        input_shape=input_shape,
        depths=MODEL_CONFIGS["large"]["depths"],
        dims=MODEL_CONFIGS["large"]["dim"],
    )

    return model


def ConvNeXt_xlarge(input_shape, num_classes=1000, drop_path_rate=0.0, layer_scale_init_value=1e-6):
    model = ConvNeXt(
        drop_path_rate=drop_path_rate,
        layer_scale_init_value=layer_scale_init_value,
        num_classes=num_classes,
        input_shape=input_shape,
        depths=MODEL_CONFIGS["xlarge"]["depths"],
        dims=MODEL_CONFIGS["xlarge"]["dim"],
    )

    return model