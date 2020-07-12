import tensorflow as tf 

class FPN(tf.keras.Model):
    def __init__(self, cfg):
        '''Feature Pyramid Networks
        
        Attributes
        ---
            out_channels: int. the channels of pyramid feature maps.
        '''
        super(FPN, self).__init__()
        
        self.out_channels = cfg["out_channels"]
        self.num_features = len(cfg["in_channels"])
        self.lateral_conv_list = []
        for i in range(self.num_features):
            self.lateral_conv_list.append(
                tf.keras.layers.Conv2D(
                    self.out_channels, (1,1),
                    kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer = tf.constant_initializer(value = -4.5 ), #-np.log(99.)
                    name="fpn/lateral_" + str(i+2)
                )
            )

        self.upsample_list = []
        for i in range(1,self.num_features):
            self.upsample_list.append(
                tf.keras.layers.UpSampling2D(
                    size=(2,2),name = "fpn/upsample_" + str(i+2)
                )
            )
        self.fpn_conv_list = []

        for i in range(self.num_features):
            self.fpn_conv_list.append(
                tf.keras.layers.Conv2D(
                    self.out_channels, (3,3), padding = "SAME", \
                    kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None),
                    bias_initializer = tf.constant_initializer(value = -4.5 ), #-np.log(99.)
                    name="fpn/p" + str(i+2)
                )
            )
        
        self.fpn_conv_list.append(
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='fpn/p6')
        )
        
            
    def call(self, inputs, training=True):
        # C2, C3, C4, C5 = inputs
        
        for i in range(self.num_features):
            self.lateral_conv_list[i](inputs[i])

        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_conv_list)
        ]

        for i in range(self.num_features - 2, -1, -1):
            shape = laterals[i].shape[1:3]
            # laterals[i] += self.upsample_list[i](laterals[i + 1])
            laterals[i] += tf.image.resize(laterals[i + 1], shape, method='bilinear')
            # _, h, w, _ = laterals[i + 1].shape
            # laterals[i] += tf.image.resize(laterals[i + 1], \
            #     [h * 2, w * 2], method='nearest')

        outputs = [self.fpn_conv_list[i](laterals[i]) for i in range(self.num_features)]

        outputs.append(self.fpn_conv_list[self.num_features](outputs[-1]))
        return outputs