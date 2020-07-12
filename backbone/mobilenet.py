import tensorflow as tf


class MobileNetV2(tf.keras.Model):
    def __init__(self, cfg):
        super(MobileNetV2, self).__init__()
        out = ["block_3_expand_relu", "block_6_expand_relu", "block_13_expand_relu", "block_16_expand_relu"]

        shape = cfg['input_shape']
        alpha = 1.0
        if "alpha" in cfg:
            alpha = cfg['alpha']
        base_model = tf.keras.applications.MobileNetV2(input_shape = shape,
                                                    alpha = alpha,
                                                    include_top = False,
                                                    weights ='imagenet')
        # extract certain feature maps for FPN
        self.model = tf.keras.Model(inputs=base_model.input,
                                              outputs=[base_model.get_layer(x).output for x in out])

    def call(self, input, training=True):
        return self.model(input)