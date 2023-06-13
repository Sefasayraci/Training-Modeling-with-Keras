


from keras.models import Model
from keras.layers import *#Resizing,Layer,LeakyReLU,AveragePooling2D,SeparableConv2D , TimeDistributed,ConvLSTM2D,Input,Concatenate,Add, Conv2D, MaxPooling2D, multiply, concatenate, BatchNormalization, Dropout, Lambda,UpSampling2D,Conv3D,MaxPooling3D,Conv3DTranspose,Conv2DTranspose,Activation
from tensorflow.keras.optimizers import Adam
from keras.metrics import MeanIoU
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.regularizers import l2
import tensorflow as tf




alpha = 0.1

def attention_network(IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS, n_filters=16, batchnorm=True):
    ### https://github.com/robinvvinod/unet/blob/master/network.py
    # contracting path
    input_img = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    c0 = inception_block(input_img,
                         n_filters=n_filters,
                         batchnorm=batchnorm,
                         strides=1,
                         recurrent=2,
                         layers=[[(3, 1), (3, 1)]])  # 512x512x512

    c1 = inception_block(c0,
                         n_filters=n_filters * 2,
                         batchnorm=batchnorm,
                         strides=2,
                         recurrent=2,
                         layers=[[(3, 1), (3, 1)]])  # 256x256x256

    c2 = inception_block(c1,
                         n_filters=n_filters * 4,
                         batchnorm=batchnorm,
                         strides=2,
                         recurrent=2,
                         layers=[[(3, 1), (3, 1)]])  # 128x128x128

    c3 = inception_block(c2,
                         n_filters=n_filters * 8,
                         batchnorm=batchnorm,
                         strides=2,
                         recurrent=2,
                         layers=[[(3, 1), (3, 1)]])  # 64x64x64

    # bridge

    b0 = inception_block(c3,
                         n_filters=n_filters * 16,
                         batchnorm=batchnorm,
                         strides=2,
                         recurrent=2,
                         layers=[[(3, 1), (3, 1)]])  # 32x32x32

    # expansive path

    attn0 = AttnGatingBlock(c3, b0, n_filters * 16)
    u0 = transpose_block(b0,
                         attn0,
                         n_filters=n_filters * 8,
                         batchnorm=batchnorm,
                         recurrent=2)  # 64x64x64

    attn1 = AttnGatingBlock(c2, u0, n_filters * 8)
    u1 = transpose_block(u0,
                         attn1,
                         n_filters=n_filters * 4,
                         batchnorm=batchnorm,
                         recurrent=2)  # 128x128x128

    attn2 = AttnGatingBlock(c1, u1, n_filters * 4)
    u2 = transpose_block(u1,
                         attn2,
                         n_filters=n_filters * 2,
                         batchnorm=batchnorm,
                         recurrent=2)  # 256x256x256

    u3 = transpose_block(u2,
                         c0,
                         n_filters=n_filters,
                         batchnorm=batchnorm,
                         recurrent=2)  # 512x512x512

    outputs = Conv2D(filters=1, kernel_size=1, strides=1,
                     activation='sigmoid')(u3)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


def transpose_block(input_tensor,
                    skip_tensor,
                    n_filters,
                    kernel_size=3,
                    strides=1,
                    batchnorm=True,
                    recurrent=1):

    # A wrapper of the Keras Conv3DTranspose block to serve as a building block for upsampling layers

    shape_x = K.int_shape(input_tensor)
    shape_xskip = K.int_shape(skip_tensor)

    conv = Conv2DTranspose(filters=n_filters,
                           kernel_size=kernel_size,
                           padding='same',
                           strides=(shape_xskip[1] // shape_x[1],
                                    shape_xskip[2] // shape_x[2]),
                           kernel_initializer="he_normal")(input_tensor)
    conv = LeakyReLU(alpha=alpha)(conv)

    act = conv2d_block(conv,
                       n_filters=n_filters,
                       kernel_size=kernel_size,
                       strides=1,
                       batchnorm=batchnorm,
                       dilation_rate=1,
                       recurrent=recurrent)
    output = Concatenate(axis=3)([act, skip_tensor])
    return output

   
def AttnGatingBlock(x, g, inter_shape):

    shape_x = K.int_shape(x)
    shape_g = K.int_shape(g)

    # Getting the gating signal to the same number of filters as the inter_shape
    phi_g = Conv2D(filters=inter_shape,
                   kernel_size=1,
                   strides=1,
                   padding='same')(g)

    # Getting the x signal to the same shape as the gating signal
    theta_x = Conv2D(filters=inter_shape,
                     kernel_size=3,
                     strides=(shape_x[1] // shape_g[1],
                              shape_x[2] // shape_g[2]),
                     padding='same')(x)

    # Element-wise addition of the gating and x signals
    add_xg = add([phi_g, theta_x])
    add_xg = Activation('relu')(add_xg)

    # 1x1x1 convolution
    psi = Conv2D(filters=1, kernel_size=1, padding='same')(add_xg)
    psi = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(psi)

    # Upsampling psi back to the original dimensions of x signal
    upsample_sigmoid_xg = UpSampling2D(
        size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(psi)

    # # Expanding the filter axis to the number of filters in the original x signal
    upsample_sigmoid_xg = expend_as(upsample_sigmoid_xg, 1)

    # Element-wise multiplication of attention coefficients back onto original x signal
    attn_coefficients = multiply([upsample_sigmoid_xg, x])

    # Final 1x1x1 convolution to consolidate attention signal to original x dimensions
    output = Conv2D(filters=1,
                    kernel_size=1,
                    strides=1,
                    padding='same')(attn_coefficients)
    output = BatchNormalization()(output)
    return output
 


def inception_block(input_tensor,
                    n_filters,
                    kernel_size=3,
                    strides=1,
                    batchnorm=True,
                    recurrent=1,
                    layers=[]):

    # Inception-style convolutional block similar to InceptionNet
    # The first convolution follows the function arguments, while subsequent inception convolutions follow the parameters in
    # argument, layers

    # layers is a nested list containing the different secondary inceptions in the format of (kernel_size, dil_rate)

    # E.g => layers=[ [(3,1),(3,1)], [(5,1)], [(3,1),(3,2)] ]
    # This will implement 3 sets of secondary convolutions
    # Set 1 => 3x3 dil = 1 followed by another 3x3 dil = 1
    # Set 2 => 5x5 dil = 1
    # Set 3 => 3x3 dil = 1 followed by 3x3 dil = 2

    res = conv2d_block(input_tensor,
                       n_filters=n_filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       batchnorm=batchnorm,
                       dilation_rate=1,
                       recurrent=recurrent)

    temp = []
    for layer in layers:
        local_res = res
        for conv in layer:
            incep_kernel_size = conv[0]
            incep_dilation_rate = conv[1]
            local_res = conv2d_block(local_res,
                                     n_filters=n_filters,
                                     kernel_size=incep_kernel_size,
                                     strides=1,
                                     batchnorm=batchnorm,
                                     dilation_rate=incep_dilation_rate,
                                     recurrent=recurrent)
        temp.append(local_res)

    temp = concatenate(temp)
    res = conv2d_block(temp,
                       n_filters=n_filters,
                       kernel_size=1,
                       strides=1,
                       batchnorm=batchnorm,
                       dilation_rate=1)

    shortcut = conv2d_block(input_tensor,
                            n_filters=n_filters,
                            kernel_size=1,
                            strides=strides,
                            batchnorm=batchnorm,
                            dilation_rate=1)
    if batchnorm:
        shortcut = BatchNormalization()(shortcut)

    output = Add()([shortcut, res])
    return output


def conv2d_block(input_tensor,
                 n_filters,
                 kernel_size=3,
                 batchnorm=True,
                 strides=1,
                 dilation_rate=1,
                 recurrent=1):

    # A wrapper of the Keras Conv3D block to serve as a building block for downsampling layers
    # Includes options to use batch normalization, dilation and recurrence

    conv = Conv2D(filters=n_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  kernel_initializer="he_normal",
                  padding="same",
                  dilation_rate=dilation_rate)(input_tensor)
    if batchnorm:
        conv = BatchNormalization()(conv)
    output = LeakyReLU(alpha=alpha)(conv)

    for _ in range(recurrent - 1):
        conv = Conv2D(filters=n_filters,
                      kernel_size=kernel_size,
                      strides=1,
                      kernel_initializer="he_normal",
                      padding="same",
                      dilation_rate=dilation_rate)(output)
        if batchnorm:
            conv = BatchNormalization()(conv)
        res = LeakyReLU(alpha=alpha)(conv)
        output = Add()([output, res])

    return output


def expend_as(tensor, rep):

    # Anonymous lambda function to expand the specified axis by a factor of argument, rep.
    # If tensor has shape (512,512,N), lambda will return a tensor of shape (512,512,N*rep), if specified axis=2

    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                       arguments={'repnum': rep})(tensor)
    return my_repeat



#####################################################################



class FilterLayer(Layer):
    def __init__(self):
        super(FilterLayer, self).__init__()
     

    def call(self, x):

        list_channels = list()

        for i in range(16):
            temp_x = x[:,:,:,i]
            temp_x = tf.expand_dims(temp_x, axis=3)
            list_channels.append(temp_x)

        output = tf.concat(list_channels,axis=3)
        return output  
def unet_blackbone(input,kernel_initializer =  'he_uniform'):
    
    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu',  kernel_initializer=kernel_initializer, padding='same')(input)
    # c1 = Dropout(0.1)(c1)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu',  kernel_initializer=kernel_initializer, padding='same')(p1)
    # c2 = Dropout(0.1)(c2)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    # c3 = Dropout(0.2)(c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(64, (3, 3), activation='relu',  kernel_initializer=kernel_initializer, padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    # c4 = Dropout(0.2)(c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    # c5 = Dropout(0.3)(c5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    # u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    # c6 = Dropout(0.2)(c6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    # u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    # c7 = Dropout(0.2)(c7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    # u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    # c8 = Dropout(0.1)(c8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    # u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    # c9 = Dropout(0.1)(c9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
     
    output = Conv2D(1, (1, 1), activation='sigmoid')(c9) # 'sigmoid'
    return output

def partial_unet(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS,kernel_initializer =  'he_uniform'):
    #Build the model for 128x128 and 2d images
    list_outputs = list()
    list_inputs  = list()
 
    for i in range(IMG_CHANNELS):
        input = Input((IMG_HEIGHT, IMG_WIDTH, 1))
        output = unet_blackbone(input,kernel_initializer =  'he_uniform')
        list_inputs.append(input)
        list_outputs.append(output)
     
    model = Model(inputs=list_inputs, outputs=list_outputs)
    return model
# def partial_unet(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS,kernel_initializer =  'he_uniform'):
#     #Build the model for 128x128 and 2d images
#     list_outputs = list()

#     inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
#     inputs = FilterLayer()(inputs)
#     for i in range(16):
#         output = unet_blackbone(tf.expand_dims(inputs[:,:,:,i],axis=-1),kernel_initializer =  'he_uniform')
#         list_outputs.append(output)
     
#     model = Model(inputs=[inputs], outputs=list_outputs)
#     return model



 #Try others if you want. For example, kernel_initializer='he_normal'

def conv_block(x, num_filters,kernel_initializer):
    x = Conv2D(num_filters, (3, 3), padding="same",kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3, 3), padding="same",kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def build_model(IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1,kernel_initializer='he_normal'):
    num_filters = [16, 32, 48, 64]
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    skip_x = []
    x = inputs
    ## Encoder
    for f in num_filters:
        x = conv_block(x, f,kernel_initializer)
        skip_x.append(x)
        x = MaxPooling2D((2, 2))(x)

    ## Bridge
    x = conv_block(x, num_filters[-1],kernel_initializer)

    num_filters.reverse()
    skip_x.reverse()
    ## Decoder
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2))(x)
        xs = skip_x[i]
        x = concatenate([x, xs])
        x = conv_block(x, f,kernel_initializer)

    ## Output
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)

    return Model(inputs, x)


def unet_2d_model_512_512(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS,kernel_initializer =  'he_uniform'):
    #Build the model for 128x128 and 2d images
    
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # s = Lambda(lambda x: x / 255)(inputs)
    
    #Contraction path
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    p1 = MaxPooling2D((2, 2),(2,2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = MaxPooling2D((2, 2),(2,2))(c2)
     
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
    p3 = MaxPooling2D((2, 2),(2,2))(c3)
     
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    p4 = MaxPooling2D((2, 2),(2,2))(c4)
     
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(512, (3, 3),(2,2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
     
    u7 = Conv2DTranspose(256, (3, 3),(2,2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)
     
    u8 = Conv2DTranspose(128, (3, 3),(2,2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
     
    u9 = Conv2DTranspose(64, (3, 3),(2,2), padding='same')(c8)
    u9 = concatenate([u9, c1]) # axis=3
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
     
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=tf.keras.initializers.HeNormal(),
    )(block_input)
    x = BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus_resnet50(IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS):
    """
    # Estimation_of_Road_Boundary_for_Intelligent_Vehicles_Based_on_DeepLabV3_Architecture.pdf or
    # https://www.researchgate.net/publication/354108791_Estimation_of_Road_Boundary_for_Intelligent_Vehicles_Based_on_DeepLabV3_Architecture
    """
    model_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    resnet50 = tf.keras.applications.ResNet50(
        weights=None, include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = UpSampling2D(
        size=(IMG_HEIGHT // 4 // x.shape[1], IMG_WIDTH // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = UpSampling2D(
        size=(IMG_HEIGHT // x.shape[1], IMG_WIDTH // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = Conv2D(1, kernel_size=(1, 1),activation="sigmoid", padding="same")(x)
    return Model(inputs=model_input, outputs=model_output)

def DeeplabV3Plus_vgg16(IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS):
    """
     # Estimation_of_Road_Boundary_for_Intelligent_Vehicles_Based_on_DeepLabV3_Architecture.pdf or
     # https://www.researchgate.net/publication/354108791_Estimation_of_Road_Boundary_for_Intelligent_Vehicles_Based_on_DeepLabV3_Architecture
    """
    model_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    vgg16 = tf.keras.applications.VGG16(
        weights=None, include_top=False, input_tensor=model_input
    )
    x = vgg16.get_layer("block5_conv3").output
    x = DilatedSpatialPyramidPooling(x)

    input_e = UpSampling2D(
        size=(IMG_HEIGHT // 4 // x.shape[1], IMG_WIDTH // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_d = vgg16.get_layer("block3_conv3").output
    input_d = convolution_block(input_d, num_filters=256, kernel_size=1)

    x = Concatenate(axis=-1)([input_e, input_d])
    x = convolution_block(x)
    x = convolution_block(x)
    x = UpSampling2D(
        size=(IMG_HEIGHT // x.shape[1], IMG_WIDTH // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = Conv2D(1, kernel_size=(1, 1),activation="sigmoid", padding="same")(x)
    return Model(inputs=model_input, outputs=model_output)

def DeeplabV3Plus_vgg19(n_class, IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS):
    """
     # Estimation_of_Road_Boundary_for_Intelligent_Vehicles_Based_on_DeepLabV3_Architecture.pdf or
     # https://www.researchgate.net/publication/354108791_Estimation_of_Road_Boundary_for_Intelligent_Vehicles_Based_on_DeepLabV3_Architecture
    """
    model_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    vgg19 = tf.keras.applications.VGG19(
        weights=None, include_top=False, input_tensor=model_input
    )
    x = vgg19.get_layer("block5_conv4").output
    x = DilatedSpatialPyramidPooling(x)

    input_e = UpSampling2D(
        size=(IMG_HEIGHT // 4 // x.shape[1], IMG_WIDTH // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_d = vgg19.get_layer("block3_conv4").output
    input_d = convolution_block(input_d, num_filters=256, kernel_size=1)

    x = Concatenate(axis=-1)([input_e, input_d])
    x = convolution_block(x)
    x = convolution_block(x)
    x = UpSampling2D(
        size=(IMG_HEIGHT // x.shape[1], IMG_WIDTH // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = Conv2D(n_class, kernel_size=(1, 1),activation="sigmoid", padding="same")(x)
    return Model(inputs=model_input, outputs=model_output)


def dense_unet_2d(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS,kernel_initializer =  'he_uniform'):
    #Build the model for 128x128 and 2d images
    
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # s = Lambda(lambda x: x / 255)(inputs)
    
    c1 = dense_block(inputs, 128, kernel_initializer)
    
    c2 = transition_block(c1, 64, kernel_initializer)
    c2 = dense_block(c2, 64, kernel_initializer)
    
    c3 = transition_block(c2, 128, kernel_initializer)
    c3 = dense_block(c3, 128, kernel_initializer)
    
    c4 = transition_block(c3, 256, kernel_initializer)
    c4 = dense_block(c4, 256, kernel_initializer)
    
    c5 = transition_block(c4, 512, kernel_initializer)
    c5 = dense_block(c5, 512, kernel_initializer)
    
    
    u_1 = up_sampling(c5, c4, 512)
    
    u_2 = dense_block(u_1, 512, kernel_initializer)
    u_2 = up_sampling(u_2, c3, 256)
    
    u_3 = dense_block(u_2, 256, kernel_initializer)
    u_3 = up_sampling(u_3, c2, 128)
    
    u_4 = dense_block(u_3, 128, kernel_initializer)
    u_4 = up_sampling(u_4, c1, 64)
    
    
    outputs = dense_block(u_4, 64, kernel_initializer)
    outputs = dense_block(outputs, 32, kernel_initializer)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(outputs) 

    model = Model(inputs=[inputs], outputs=[outputs])

    return model 


def up_sampling(input_, c, filter_):
    x = Conv2DTranspose(filter_, (2, 2), strides=(2, 2), padding='same')(input_)
    x = concatenate([x, c], axis=-1)
    return x

def transition_block(input_,filter_,kernel_initializer):
    x = BatchNormalization()(input_)
    x = Conv2D(filter_, (1, 1), activation='relu', kernel_initializer=kernel_initializer, padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    return x

def dense_block(input_,filter_,kernel_initializer):
    
    den_conv_1 = dense_conv_layer(input_, filter_, kernel_initializer)
    den_conv_1_in =Concatenate()([den_conv_1, input_])

    den_conv_2 = dense_conv_layer(den_conv_1_in, filter_, kernel_initializer)
    den_conv_2_in = Concatenate()([den_conv_2, den_conv_1_in])
    
    den_conv_3 = dense_conv_layer(den_conv_2_in, filter_, kernel_initializer)
    den_conv_3_in = Concatenate()([den_conv_3, den_conv_2_in])
        
    den_conv_4 = dense_conv_layer(den_conv_3_in, filter_, kernel_initializer)    
    return den_conv_4

def dense_conv_layer(input_,filter_,kernel_initializer):
    
    x = BatchNormalization()(input_)
    x = Conv2D(filter_, (1, 1), activation='relu', kernel_initializer=kernel_initializer, padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filter_, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(x)
    x = Dropout(0.1)(x)
    return x


def unet_2d_model_128_128_seperable(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS,kernel_initializer =  'he_uniform'):
    #Build the model for 128x128 and 2d images
    from keras.backend import tf as ktf

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # s = Lambda(lambda x: x / 255)(inputs)
 
    #Contraction path
    c1 = SeparableConv2D (16, (3, 3), activation='relu',  kernel_initializer=kernel_initializer, padding='same')(inputs)
    # c1 = Dropout(0.1)(c1)
    c1 = BatchNormalization()(c1)
    c1 = SeparableConv2D (16, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = SeparableConv2D (32, (3, 3), activation='relu',  kernel_initializer=kernel_initializer, padding='same')(p1)
    # c2 = Dropout(0.1)(c2)
    c2 = BatchNormalization()(c2)
    c2 = SeparableConv2D (32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = SeparableConv2D (64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    # c3 = Dropout(0.2)(c3)
    c3 = BatchNormalization()(c3)
    c3 = SeparableConv2D (64, (3, 3), activation='relu',  kernel_initializer=kernel_initializer, padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = SeparableConv2D (128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    # c4 = Dropout(0.2)(c4)
    c4 = BatchNormalization()(c4)
    c4 = SeparableConv2D (128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = SeparableConv2D (256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    # c5 = Dropout(0.3)(c5)
    c5 = BatchNormalization()(c5)
    c5 = SeparableConv2D (256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = SeparableConv2D (128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    # c6 = Dropout(0.2)(c6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = SeparableConv2D (64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    # c7 = Dropout(0.2)(c7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = SeparableConv2D (32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    # c8 = Dropout(0.1)(c8)
    c8 = BatchNormalization()(c8)
    c8 = SeparableConv2D (32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = SeparableConv2D (16, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    # c9 = Dropout(0.1)(c9)
    c9 = BatchNormalization()(c9)
    c9 = SeparableConv2D (16, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
     
    outputs = SeparableConv2D (1, (1, 1), activation='sigmoid')(c9) # 'sigmoid'
     
    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def unet_2d_model_128_128(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS,kernel_initializer =  'he_uniform'):
    #Build the model for 128x128 and 2d images
    from keras.backend import tf as ktf

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # s = Lambda(lambda x: x / 255)(inputs)
 
    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu',  kernel_initializer=kernel_initializer, padding='same')(inputs)
    # c1 = Dropout(0.1)(c1)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu',  kernel_initializer=kernel_initializer, padding='same')(p1)
    # c2 = Dropout(0.1)(c2)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    # c3 = Dropout(0.2)(c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(64, (3, 3), activation='relu',  kernel_initializer=kernel_initializer, padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    # c4 = Dropout(0.2)(c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    # c5 = Dropout(0.3)(c5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    # c6 = Dropout(0.2)(c6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    # c7 = Dropout(0.2)(c7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    # c8 = Dropout(0.1)(c8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    # c9 = Dropout(0.1)(c9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
     
    outputs = Conv2D(IMG_CHANNELS, (1, 1), activation='sigmoid')(c9) # 'sigmoid'
     
    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def unet_2d_model_128_128_LSTM(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS,kernel_initializer =  'he_uniform'):
    #Build the model for 128x128 and 2d images
    
    inputs = Input((16,IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # s = Lambda(lambda x: x / 255)(inputs)
    
    #Contraction path
    c1 = ConvLSTM2D(16, (3, 3), return_sequences=True, activation='relu',  kernel_initializer=kernel_initializer, padding='same')(inputs)
    # c1 = Dropout(0.1)(c1)
    c1 = BatchNormalization()(c1)
    c1 = TimeDistributed(Conv2D(16, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))(c1)
    p1 = TimeDistributed(MaxPooling2D((2,2),(2,2)))(c1)
    
    c2 = ConvLSTM2D(32, (3, 3), return_sequences=True, activation='relu',  kernel_initializer=kernel_initializer, padding='same')(p1)
    # c2 = Dropout(0.1)(c2)
    c2 = BatchNormalization()(c2)
    c2 = TimeDistributed(Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))(c2)
    p2 = TimeDistributed(MaxPooling2D((2,2),(2,2)))(c2)
     
    c3 = ConvLSTM2D(64, (3, 3), return_sequences=True, activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    # c3 = Dropout(0.2)(c3)
    c3 = BatchNormalization()(c3)
    c3 = TimeDistributed(Conv2D(64, (3, 3), activation='relu',  kernel_initializer=kernel_initializer, padding='same'))(c3)
    p3 = TimeDistributed(MaxPooling2D((2,2),(2, 2)))(c3)
     
    c4 = ConvLSTM2D(128, (3, 3), return_sequences=True, activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    # c4 = Dropout(0.2)(c4)
    c4 = BatchNormalization()(c4)
    c4 = TimeDistributed(Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))(c4)
    p4 = TimeDistributed(MaxPooling2D((2,2),(2, 2)))(c4)
     
    c5 = ConvLSTM2D(256, (3, 3), return_sequences=True, activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    # c5 = Dropout(0.3)(c5)
    c5 = BatchNormalization()(c5)
    c5 = TimeDistributed(Conv2D(256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))(c5)
    
    #Expansive path 
    u6 = TimeDistributed(Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same'))(c5)
    u6 = concatenate([u6, c4])
    c6 = TimeDistributed(Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))(u6)
    # c6 = Dropout(0.2)(c6)
    c6 = BatchNormalization()(c6)
    c6 = TimeDistributed(Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))(c6)
     
    u7 = TimeDistributed(Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same'))(c6)
    u7 = concatenate([u7, c3])
    c7 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))(u7)
    # c7 = Dropout(0.2)(c7)
    c7 = BatchNormalization()(c7)
    c7 = TimeDistributed(Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))(c7)
     
    u8 = TimeDistributed(Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same'))(c7)
    u8 = concatenate([u8, c2])
    c8 = TimeDistributed(Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))(u8)
    # c8 = Dropout(0.1)(c8)
    c8 = BatchNormalization()(c8)
    c8 = TimeDistributed(Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))(c8)
     
    u9 = TimeDistributed(Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same'))(c8)
    u9 = concatenate([u9, c1])
    c9 = TimeDistributed(Conv2D(16, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))(u9)
    # c9 = Dropout(0.1)(c9)
    c9 = BatchNormalization()(c9)
    c9 = TimeDistributed(Conv2D(16, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))(c9)
     
    outputs = TimeDistributed(Conv2D(1, (1, 1), activation='sigmoid'))(c9) # 'sigmoid'
     
    model = Model(inputs=[inputs], outputs=[outputs])
    return model




# def weighted_binary_loss(X):

#     print(X)
#     y_pred, weights, y_true = X
#     print(y_pred)
#     print(weights)
#     print(y_true)
    
#     loss = K.binary_crossentropy(y_pred, y_true)
#     # loss = multiply([loss, weights])
#     # loss2 = multiply([y_true, weights])
#     # loss = tf.image.adjust_saturation(loss,1)
#     return loss

# def identity_loss(y_true, y_pred):
#     return y_pred

# def unet_2d_model_128_128_v2(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS,kernel_initializer =  'he_uniform', w_decay = 0.01):
#     #Build the model for 128x128 and 2d images
    
#     inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
#     mask_weights = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
#     true_masks = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
#     # s = Lambda(lambda x: x / 255)(inputs)
    
#     #Contraction path
#     c1 = Conv2D(16, (3, 3), activation='relu',  kernel_regularizer=l2(w_decay), kernel_initializer=kernel_initializer, padding='same')(inputs)
#     # c1 = Dropout(0.1)(c1)
#     c1 = BatchNormalization()(c1)
#     c1 = Conv2D(16, (3, 3), activation='relu', kernel_regularizer=l2(w_decay), kernel_initializer=kernel_initializer, padding='same')(c1)
#     p1 = MaxPooling2D((2, 2))(c1)
    
#     c2 = Conv2D(32, (3, 3), activation='relu',kernel_regularizer=l2(w_decay),  kernel_initializer=kernel_initializer, padding='same')(p1)
#     # c2 = Dropout(0.1)(c2)
#     c2 = BatchNormalization()(c2)
#     c2 = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(w_decay), kernel_initializer=kernel_initializer, padding='same')(c2)
#     p2 = MaxPooling2D((2, 2))(c2)
     
#     c3 = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(w_decay), kernel_initializer=kernel_initializer, padding='same')(p2)
#     # c3 = Dropout(0.2)(c3)
#     c3 = BatchNormalization()(c3)
#     c3 = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(w_decay),  kernel_initializer=kernel_initializer, padding='same')(c3)
#     p3 = MaxPooling2D((2, 2))(c3)
     
#     c4 = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(w_decay), kernel_initializer=kernel_initializer, padding='same')(p3)
#     # c4 = Dropout(0.2)(c4)
#     c4 = BatchNormalization()(c4)
#     c4 = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(w_decay), kernel_initializer=kernel_initializer, padding='same')(c4)
#     p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
#     c5 = Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(w_decay), kernel_initializer=kernel_initializer, padding='same')(p4)
#     # c5 = Dropout(0.3)(c5)
#     c5 = BatchNormalization()(c5)
#     c5 = Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(w_decay), kernel_initializer=kernel_initializer, padding='same')(c5)
    
#     #Expansive path 
#     u6 = Conv2DTranspose(128, (2, 2),kernel_regularizer=l2(w_decay), strides=(2, 2), padding='same')(c5)
#     u6 = concatenate([u6, c4])
#     c6 = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(w_decay), kernel_initializer=kernel_initializer, padding='same')(u6)
#     # c6 = Dropout(0.2)(c6)
#     c6 = BatchNormalization()(c6)
#     c6 = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(w_decay), kernel_initializer=kernel_initializer, padding='same')(c6)
     
#     u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_regularizer=l2(w_decay), padding='same')(c6)
#     u7 = concatenate([u7, c3])
#     c7 = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(w_decay), kernel_initializer=kernel_initializer, padding='same')(u7)
#     # c7 = Dropout(0.2)(c7)
#     c7 = BatchNormalization()(c7)
#     c7 = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(w_decay), kernel_initializer=kernel_initializer, padding='same')(c7)
     
#     u8 = Conv2DTranspose(32, (2, 2), kernel_regularizer=l2(w_decay), strides=(2, 2), padding='same')(c7)
#     u8 = concatenate([u8, c2])
#     c8 = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(w_decay), kernel_initializer=kernel_initializer, padding='same')(u8)
#     # c8 = Dropout(0.1)(c8)
#     c8 = BatchNormalization()(c8)
#     c8 = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(w_decay), kernel_initializer=kernel_initializer, padding='same')(c8)
     
#     u9 = Conv2DTranspose(16, (2, 2), kernel_regularizer=l2(w_decay), strides=(2, 2), padding='same')(c8)
#     u9 = concatenate([u9, c1], axis=3)
#     c9 = Conv2D(16, (3, 3), activation='relu', kernel_regularizer=l2(w_decay), kernel_initializer=kernel_initializer, padding='same')(u9)
#     # c9 = Dropout(0.1)(c9)
#     c9 = BatchNormalization()(c9)
#     c9 = Conv2D(16, (3, 3), activation='relu', kernel_regularizer=l2(w_decay), kernel_initializer=kernel_initializer, padding='same')(c9)
     
#     outputs = Conv2D(1, (1, 1), kernel_regularizer=l2(w_decay), activation='sigmoid')(c9) # 'sigmoid'
    
#     # loss = Lambda(weighted_binary_loss,output_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))([outputs,mask_weights,true_masks])
#     # model = Model(inputs=[inputs,mask_weights,true_masks], outputs=[outputs])
#     model = Model(inputs=[inputs], outputs=[outputs])
#     return model

def unet_2d_model_256_256(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS,kernel_initializer =  'he_uniform'):
    #Build the model for 256x256 and 2d images 
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # s = Lambda(lambda x: x / 255)(inputs)
    
    #Contraction path
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
     
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)
     
    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
     
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
    
    c9 = Conv2D(2, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9) # yeni ekledim

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    return model



################################################################
def unet_3d_model_128_128(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes,kernel_initializer =  'he_uniform'):
    #Build the model for 128x128 and 3d images
    
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)
    
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)
     
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)
     
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)
     
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    
    #Expansive path 
    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
     
    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)
     
    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
     
    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
     
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    # return model    
    return model


def vgg16(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,kernel_initializer= 'he_uniform'):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    c1 = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer=kernel_initializer)(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer=kernel_initializer)(c1)
    p1 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(c1)
    
    c2 = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c2)
    p2 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(c2)
    
    c3 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c3)
    p3 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(c3)
    
    c4 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c4)
    p4 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(c4)
    
    c5 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(p4)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c5)
    p5 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(c5)
    
    
    c6 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(p5)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", kernel_initializer=kernel_initializer)(c6)
    
    u7 = Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(c6)
    u7 = concatenate([u7, c5])
    c7 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(u7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(c7)

    u8 = Conv2DTranspose(512, (2,2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c4])
    c8 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(u8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(c8)
  
    u9 = Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(c8)
    u9 = concatenate([u9, c3])
    c9 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(u9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(c9)
 
    u10 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c9)
    u10 = concatenate([u10, c2])
    c10 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(u10)
    c10 = BatchNormalization()(c10)
    c10 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(c10)
                                   
    u11 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c10)
    u11 = concatenate([u11, c1])
    c11 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(u11)
    c11 = BatchNormalization()(c11)
    c11 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(c11)
                                   
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c11)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

class ResUnet():
    def __init__(self,IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_CHANNELS = IMG_CHANNELS
        
    def bn_act(self,x,act=True):
        x = BatchNormalization()(x)
        # x = Dropout(0.2)(x)
        if act == True:
            x = Activation('relu')(x)
        return x

    def conv_block(self,x,filters,kernel_size=(3,3),padding='same',strides=1):
        conv = self.bn_act(x)
        conv = Conv2D(filters,kernel_size,padding=padding,strides=strides)(conv)
        return conv
    def stem(self,x,filters,kernel_size=(3,3),padding='same',strides=1):
        conv = Conv2D(filters,kernel_size=kernel_size,padding=padding,strides=strides)(x)
        conv = self.conv_block(x,filters,kernel_size=kernel_size,padding=padding,strides=strides)

        shortcut = Conv2D(filters,kernel_size=(1,1),padding=padding,strides=strides)(x)
        shortcut = self.bn_act(shortcut,act=False)

        output = Add()([conv,shortcut])
        return output
    def residual_block(self,x,filters,kernel_size=(3,3),padding='same',strides=1):
        res = self.conv_block(x,filters,kernel_size=kernel_size,padding=padding,strides=strides)
        res = self.conv_block(res,filters,kernel_size=kernel_size,padding=padding,strides=1)

        shortcut = Conv2D(filters,kernel_size=(1,1),padding=padding,strides=strides)(x)
        shortcut = self.bn_act(shortcut,act=False)

        output = Add()([shortcut,res])
        return output
    def upsample_concat_block(self,x,xskip):
        u = UpSampling2D((2,2))(x)
        c = Concatenate()([u,xskip])
        return c

    def get_model(self,n_class=6):
        f = [16,32,64,128,256]
        inputs = Input((self.IMG_HEIGHT,self.IMG_WIDTH,self.IMG_CHANNELS))

        # Encoder
        e0 = inputs
        e1 = self.stem(e0,f[0])
        e2 = self.residual_block(e1,f[1],strides=2)
        e3 = self.residual_block(e2,f[2],strides=2)
        e4 = self.residual_block(e3,f[3],strides=2)
        e5 = self.residual_block(e4,f[4],strides=2)

        # Bridge
        b0 = self.conv_block(e5,f[4],strides=1)
        b1 = self.conv_block(b0,f[4],strides=1)

        # Decoder
        u1 = self.upsample_concat_block(b1,e4)
        d1 = self.residual_block(u1,f[4])

        u2 = self.upsample_concat_block(d1,e3)
        d2 = self.residual_block(u2,f[3])

        u3 = self.upsample_concat_block(d2,e2)
        d3 = self.residual_block(u3,f[2])

        u4 = self.upsample_concat_block(d3,e1)
        d4 = self.residual_block(u4,f[1])
        
        outputs = Conv2D(n_class,(1,1),padding='same',activation='sigmoid')(d4)
        model = Model(inputs,outputs)
        return model
          


class ResUnet_with_regulazer_or_dilation():
    def __init__(self,IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_CHANNELS = IMG_CHANNELS
        
    def bn_act(self,x,act=True):
        x = BatchNormalization()(x)
        # x = Dropout(0.2)(x)
        if act == True:
            x = Activation('relu')(x)
        return x

    def conv_block(self,x,filters,kernel_regularizer,activity_regularizer,bias_regularizer,kernel_size=(3,3),padding='same',strides=1):
        conv = self.bn_act(x)
        conv = Conv2D(filters,kernel_size,padding=padding,strides=strides,kernel_regularizer=kernel_regularizer,activity_regularizer=activity_regularizer,bias_regularizer=bias_regularizer)(conv)
        return conv
    def stem(self,x,filters,kernel_regularizer,activity_regularizer,bias_regularizer,kernel_size=(3,3),padding='same',strides=1):
        conv = Conv2D(filters,kernel_size=kernel_size,padding=padding,strides=strides,kernel_regularizer=kernel_regularizer,activity_regularizer=activity_regularizer,bias_regularizer=bias_regularizer)(x)
        conv = self.conv_block(x,filters,kernel_regularizer=kernel_regularizer,activity_regularizer=activity_regularizer,bias_regularizer=bias_regularizer,kernel_size=kernel_size,padding=padding,strides=strides)

        shortcut = Conv2D(filters,kernel_size=(1,1),padding=padding,strides=strides,kernel_regularizer=kernel_regularizer,activity_regularizer=activity_regularizer,bias_regularizer=bias_regularizer)(x)
        shortcut = self.bn_act(shortcut,act=False)

        output = Add()([conv,shortcut])
        return output
    def residual_block(self,x,filters,kernel_regularizer,activity_regularizer,bias_regularizer,kernel_size=(3,3),padding='same',strides=1):
        res = self.conv_block(x,filters,kernel_regularizer=kernel_regularizer,activity_regularizer=activity_regularizer,bias_regularizer=bias_regularizer,kernel_size=kernel_size,padding=padding,strides=strides)
        res = self.conv_block(res,filters,kernel_regularizer=kernel_regularizer,activity_regularizer=activity_regularizer,bias_regularizer=bias_regularizer,kernel_size=kernel_size,padding=padding,strides=1)

        shortcut = Conv2D(filters,kernel_regularizer=kernel_regularizer,activity_regularizer=activity_regularizer,bias_regularizer=bias_regularizer,kernel_size=(1,1),padding=padding,strides=strides)(x)
        shortcut = self.bn_act(shortcut,act=False)

        output = Add()([shortcut,res])
        return output
    def upsample_concat_block(self,x,xskip):
        u = UpSampling2D((2,2))(x)
        c = Concatenate()([u,xskip])
        return c

    def get_model(self,kernel_regularizer=None,activity_regularizer=None,bias_regularizer=None):
        f = [16,32,64,128,256]
        inputs = Input((self.IMG_HEIGHT,self.IMG_WIDTH,self.IMG_CHANNELS))

        # Encoder
        e0 = inputs
        e1 = self.stem(e0,f[0],kernel_regularizer=kernel_regularizer,activity_regularizer=activity_regularizer,bias_regularizer=bias_regularizer)
        e2 = self.residual_block(e1,f[1],kernel_regularizer=kernel_regularizer,activity_regularizer=activity_regularizer,bias_regularizer=bias_regularizer,strides=2)
        e3 = self.residual_block(e2,f[2],kernel_regularizer=kernel_regularizer,activity_regularizer=activity_regularizer,bias_regularizer=bias_regularizer,strides=2)
        e4 = self.residual_block(e3,f[3],kernel_regularizer=kernel_regularizer,activity_regularizer=activity_regularizer,bias_regularizer=bias_regularizer,strides=2)
        e5 = self.residual_block(e4,f[4],kernel_regularizer=kernel_regularizer,activity_regularizer=activity_regularizer,bias_regularizer=bias_regularizer,strides=2)

        # Bridge
        b0 = self.conv_block(e5,f[4],kernel_regularizer=kernel_regularizer,activity_regularizer=activity_regularizer,bias_regularizer=bias_regularizer,strides=1)
        b1 = self.conv_block(b0,f[4],kernel_regularizer=kernel_regularizer,activity_regularizer=activity_regularizer,bias_regularizer=bias_regularizer,strides=1)

        # Decoder
        u1 = self.upsample_concat_block(b1,e4)
        d1 = self.residual_block(u1,f[4],kernel_regularizer=kernel_regularizer,activity_regularizer=activity_regularizer,bias_regularizer=bias_regularizer)

        u2 = self.upsample_concat_block(d1,e3)
        d2 = self.residual_block(u2,f[3],kernel_regularizer=kernel_regularizer,activity_regularizer=activity_regularizer,bias_regularizer=bias_regularizer)

        u3 = self.upsample_concat_block(d2,e2)
        d3 = self.residual_block(u3,f[2],kernel_regularizer=kernel_regularizer,activity_regularizer=activity_regularizer,bias_regularizer=bias_regularizer)

        u4 = self.upsample_concat_block(d3,e1)
        d4 = self.residual_block(u4,f[1],kernel_regularizer=kernel_regularizer,activity_regularizer=activity_regularizer,bias_regularizer=bias_regularizer)
        
        outputs = Conv2D(1,(1,1),kernel_regularizer=kernel_regularizer,activity_regularizer=activity_regularizer,bias_regularizer=bias_regularizer,padding='same')(d4)
        outputs = Conv2D(1,(1,1),dilation_rate=10,padding='same',activation='sigmoid')(outputs)

        model = Model(inputs,outputs)
        return model
             

        
        

if __name__ == "__main__":
    #Test 3D if everything is working ok. 
    # model = simple_unet_model(128, 128, 128, 3, 4)
    # print(model.summary())
    # print(model.input_shape)
    # print(model.output_shape)

    
    # #Test 2D if everything is working ok. 
    # model = unet_2d_model_256_256(256, 256, 1)
    # model.summary()
    # print(model.input_shape)
    # print(model.output_shape)

    # #Test 2D if everything is working ok. 
    # model = unet_2d_model_128_128(256, 256, 1)
    # model.summary()
    # print(model.input_shape)
    # print(model.output_shape)
    
    # resunet = ResUnet(256, 256, 1)
    # model = resunet.get_model()
    # model.summary()
    # print(model.input_shape)
    # print(model.output_shape)
     #Test 2D if everything is working ok.
        
    # import keras.utils.vis_utils
    # from importlib import reload
    # reload(keras.utils.vis_utils)

    from keras.utils.vis_utils import plot_model
    # model = vgg16(256, 256, 1)
    # model = partial_unet(64,64,16)
    # model.summary()
    # print(model.input_shape)
    # print(model.output_shape)
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    model = ResUnet_with_regulazer().get_model()
    model.summary()
    print(model.input_shape)
    print(model.output_shape)
