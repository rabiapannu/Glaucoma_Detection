from tensorflow.keras import layers, models


class ResUNet:
    def __init__(self, input_shape, filters=None):
        if filters is None:
            filters = [16, 32, 64, 128, 256]
        self.input_shape = input_shape
        self.filters = filters
        self.model = self.create_model()  # Call the function when instantiated

    def bn_act(self, x, act=True):
        """ Batch Normalization followed by optional activation """
        x = layers.BatchNormalization()(x)
        if act:
            x = layers.Activation("relu")(x)
        return x

    def conv_block(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):
        """ Convolutional Block with BatchNorm and Activation """
        conv = self.bn_act(x)
        conv = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
        return conv

    def stem(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):
        """ Initial stem block with skip connection """
        conv = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        conv = self.conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)

        shortcut = layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = self.bn_act(shortcut, act=False)

        output = layers.Add()([conv, shortcut])
        return output

    def residual_block(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):
        """ Residual Block with two convolutional layers and skip connection """
        res = self.conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
        res = self.conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

        shortcut = layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = self.bn_act(shortcut, act=False)

        output = layers.Add()([shortcut, res])
        return output

    def upsample_concat_block(self, x, xskip):
        """ Upsampling block with concatenation """
        u = layers.UpSampling2D((2, 2))(x)
        c = layers.Concatenate()([u, xskip])
        return c

    def create_model(self):
        """ Constructs the ResUNet model """
        inputs = layers.Input(self.input_shape)
        f = self.filters

        # Encoder
        e0 = inputs
        e1 = self.stem(e0, f[0])
        e2 = self.residual_block(e1, f[1], strides=2)
        e3 = self.residual_block(e2, f[2], strides=2)
        e4 = self.residual_block(e3, f[3], strides=2)
        e5 = self.residual_block(e4, f[4], strides=2)

        # Bridge
        b0 = self.conv_block(e5, f[4], strides=1)
        b1 = self.conv_block(b0, f[4], strides=1)

        # Decoder
        u1 = self.upsample_concat_block(b1, e4)
        d1 = self.residual_block(u1, f[4])

        u2 = self.upsample_concat_block(d1, e3)
        d2 = self.residual_block(u2, f[3])

        u3 = self.upsample_concat_block(d2, e2)
        d3 = self.residual_block(u3, f[2])

        u4 = self.upsample_concat_block(d3, e1)
        d4 = self.residual_block(u4, f[1])

        # Output layer
        outputs = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)

        # Model
        model = models.Model(inputs, outputs)
        return model
