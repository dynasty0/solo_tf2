import tensorflow as tf


class ResNet101(tf.keras.Model):
    def __init__(self, cfg):
        super(ResNet101,self).__init__()
        out = ["conv2_block2_out", "conv3_block3_out", "conv4_block22_out", "conv5_block3_out"]

        shape = cfg['input_shape']
        base_model = tf.keras.applications.ResNet101V2(input_shape=shape,
                                                    include_top=False,
                                                    weights='imagenet')
        # extract certain feature maps for FPN
        self.model = tf.keras.Model(inputs=base_model.input,
                                              outputs=[base_model.get_layer(x).output for x in out])

    def call(self, input, training=True):
        return self.model(input)


class ResNet50(tf.keras.Model):
    def __init__(self, cfg):
        super(ResNet50,self).__init__()
        out = ["conv2_block2_out", "conv3_block3_out", "conv4_block5_out", "conv5_block3_out"]

        shape = cfg['input_shape']
        base_model = tf.keras.applications.ResNet50V2(input_shape=shape,
                                                    include_top=False,
                                                    weights='imagenet')
        # extract certain feature maps for FPN
        self.model = tf.keras.Model(inputs=base_model.input,
                                              outputs=[base_model.get_layer(x).output for x in out])

    def call(self, input, training=True):
        return self.model(input)