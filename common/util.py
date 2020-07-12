import tensorflow as tf
nn = tf.keras.layers
from inspect import isfunction
import tensorflow_addons as tfa

def is_channels_first(data_format):
    """
    Is tested data format channels first.
    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    Returns
    -------
    bool
        A flag.
    """
    return data_format == "channels_first"


def get_channel_axis(data_format):
    """
    Get channel axis.
    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    Returns
    -------
    int
        Channel axis.
    """
    return 1 if is_channels_first(data_format) else -1

class ReLU6(nn.Layer):
    """
    ReLU6 activation layer.
    """
    def __init__(self, **kwargs):
        super(ReLU6, self).__init__(**kwargs)

    def call(self, x):
        return tf.nn.relu6(x)


class Swish(nn.Layer):
    """
    Swish activation function from 'Searching for Activation Functions,' https://arxiv.org/abs/1710.05941.
    """
    def call(self, x):
        return x * tf.nn.sigmoid(x)


class HSigmoid(nn.Layer):
    """
    Approximated sigmoid function, so-called hard-version of sigmoid from 'Searching for MobileNetV3,'
    https://arxiv.org/abs/1905.02244.
    """
    def __init__(self, **kwargs):
        super(HSigmoid, self).__init__(**kwargs)

    def call(self, x):
        return tf.nn.relu6(x + 3.0) / 6.0


class HSwish(nn.Layer):
    """
    H-Swish activation function from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    """
    def __init__(self, **kwargs):
        super(HSwish, self).__init__(**kwargs)

    def call(self, x):
        return x * tf.nn.relu6(x + 3.0) / 6.0


class PReLU2(nn.Layer):
    """
    Parametric leaky version of a Rectified Linear Unit (with wide alpha).
    Parameters:
    ----------
    alpha : int, default 0.25
        Negative slope coefficient.
    """
    def __init__(self,
                 alpha=0.25,
                 **kwargs):
        super(PReLU2, self).__init__(**kwargs)
        self.active = nn.LeakyReLU(alpha=alpha)

    def call(self, x):
        return self.active(x)


def get_activation_layer(activation):
    """
    Create activation layer from string/function.
    Parameters:
    ----------
    activation : function, or str, or nn.Layer
        Activation function or name of activation function.
    Returns
    -------
    nn.Layer
        Activation layer.
    """
    assert (activation is not None)
    if isfunction(activation):
        return activation()
    elif isinstance(activation, str):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "relu6":
            return ReLU6()
        elif activation == "swish":
            return Swish()
        elif activation == "hswish":
            return HSwish()
        elif activation == "sigmoid":
            return tf.nn.sigmoid
        elif activation == "hsigmoid":
            return HSigmoid()
        else:
            raise NotImplementedError()
    else:
        assert (isinstance(activation, nn.Layer))
        return activation


class Conv2d(tf.keras.layers.Layer):
    """
    Standard convolution layer.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default True
        Whether the layer uses a bias vector.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 use_bias=True,
                 data_format="channels_last",
                 **kwargs):
        super(Conv2d, self).__init__(**kwargs)
        assert (in_channels is not None)
        self.data_format = data_format
        self.use_conv = (groups == 1)
        self.use_dw_conv = (groups > 1) and (groups == out_channels) and (out_channels == in_channels)

        # assert (strides == 1) or (dilation == 1)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(strides, int):
            strides = (strides, strides)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.use_pad = (padding[0] > 0) or (padding[1] > 0)
        if self.use_pad:
            self.pad = nn.ZeroPadding2D(
                padding=padding,
                data_format=data_format)
            # if is_channels_first(data_format):
            #     self.paddings_tf = [[0, 0], [0, 0], list(padding), list(padding)]
            # else:
            #     self.paddings_tf = [[0, 0], list(padding), list(padding), [0, 0]]

        if self.use_conv:
            self.conv = nn.Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding="valid",
                data_format=data_format,
                dilation_rate=dilation,
                use_bias=use_bias,
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
                name="conv")
        elif self.use_dw_conv:
            # assert (dilation[0] == 1) and (dilation[1] == 1)
            self.dw_conv = nn.DepthwiseConv2D(
                kernel_size=kernel_size,
                strides=strides,
                padding="valid",
                data_format=data_format,
                dilation_rate=dilation,
                use_bias=use_bias,
                name="dw_conv")
        else:
            assert (groups > 1)
            assert (in_channels % groups == 0)
            assert (out_channels % groups == 0)
            self.groups = groups
            self.convs = []
            for i in range(groups):
                self.convs.append(nn.Conv2D(
                    filters=(out_channels // groups),
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="valid",
                    data_format=data_format,
                    dilation_rate=dilation,
                    use_bias=use_bias,
                    name="convgroup{}".format(i + 1)))

    def call(self, x):
        if self.use_pad:
            x = self.pad(x)
            # x = tf.pad(x, paddings=self.paddings_tf)
        if self.use_conv:
            try:
                x = self.conv(x)
            except tf.errors.InvalidArgumentError as ex:
                if self.conv.dilation_rate != (1, 1):
                    conv_ = nn.Conv2D(
                        filters=self.conv.filters,
                        kernel_size=self.conv.kernel_size,
                        strides=self.conv.strides,
                        padding="valid",
                        data_format=self.data_format,
                        dilation_rate=self.conv.dilation_rate,
                        use_bias=self.conv.use_bias,
                        name="conv_")
                    _ = conv_(x)
                    conv_.weights[0].assign(self.conv.weights[0])
                    if len(self.conv.weights) > 1:
                        conv_.weights[1].assign(self.conv.weights[1])
                    x = conv_(x)
                else:
                    raise ex
            # x = self.conv(x)
        elif self.use_dw_conv:
            x = self.dw_conv(x)
        else:
            yy = []
            xx = tf.split(x, num_or_size_splits=self.groups, axis=get_channel_axis(self.data_format))
            for xi, convi in zip(xx, self.convs):
                yy.append(convi(xi))
            x = tf.concat(yy, axis=get_channel_axis(self.data_format))
        return x

class BatchNorm(nn.BatchNormalization):
    """
    MXNet/Gluon-like batch normalization.
    Parameters:
    ----------
    momentum : float, default 0.9
        Momentum for the moving average.
    epsilon : float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 momentum=0.9,
                 epsilon=1e-5,
                 data_format="channels_last",
                 **kwargs):
        super(BatchNorm, self).__init__(
            axis=get_channel_axis(data_format),
            momentum=momentum,
            epsilon=epsilon,
            **kwargs)

class ConvBlock(tf.keras.layers.Layer):
    """
    Standard convolution block with Batch normalization and activation.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation="relu",
                 data_format="channels_last",
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        assert (in_channels is not None)
        self.activate = (activation is not None)
        self.use_bn = use_bn

        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            data_format=data_format,
            name="conv")
        if self.use_bn:
            self.bn = BatchNorm(
                epsilon=bn_eps,
                data_format=data_format,
                name="bn")
        if self.activate:
            self.activ = get_activation_layer(activation)

    def call(self, x, training=None):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x, training=training)
        if self.activate:
            x = self.activ(x)
        return x


class ConvBlock_gn(tf.keras.layers.Layer):
    """
    Standard convolution block with Batch normalization and activation.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation="relu",
                 data_format="channels_last",
                 **kwargs):
        super(ConvBlock_gn, self).__init__(**kwargs)
        assert (in_channels is not None)
        self.activate = (activation is not None)
        self.use_bn = use_bn

        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            data_format=data_format,
            name="conv")
        if self.use_bn:
            self.bn = tfa.layers.GroupNormalization(
                groups = 32, name = "gn")
        if self.activate:
            self.activ = get_activation_layer(activation)

    def call(self, x, training=None):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x, training=training)
        if self.activate:
            x = self.activ(x)
        return x