import math
import keras
from keras.models import *
from keras.layers import *
from keras import layers
import keras.backend as K
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

IMAGE_ORDERING = 'channels_last'
def one_side_pad( x ):
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    if IMAGE_ORDERING == 'channels_first':
        x = Lambda(lambda x : x[: , : , :-1 , :-1 ] )(x)
    elif IMAGE_ORDERING == 'channels_last':
        x = Lambda(lambda x : x[: , :-1 , :-1 , :  ] )(x)
    return x

def identity_block(input_tensor, kernel_size, filter_num, block):


    conv_name_base = 'res' + block + '_branch'
    in_name_base = 'in' + block + '_branch'

    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(input_tensor)
    x = Conv2D(filter_num, (3, 3) , data_format=IMAGE_ORDERING , name=conv_name_base + '2a')(x)
    x = InstanceNormalization(axis=3,name=in_name_base + '2a')(x)
    x = Activation('relu')(x)
    #dense
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    x1 = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    x1 = Conv2D(filter_num , (3, 3), data_format=IMAGE_ORDERING , name=conv_name_base + '2c')(x1)
    x1 = InstanceNormalization(axis=3,name=in_name_base + '2c')(x1)
    #dense
    x1 = layers.add([x1, x])
    x1 = Activation('relu')(x1)
    # 残差网络
    x1 = layers.add([x1, input_tensor])
    x1 = Activation('relu')(x1)
    return x

def eca_block(input_feature, b=1, gamma=2, name=""):
	channel = input_feature._keras_shape[-1]
	kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
	kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
	
	avg_pool = GlobalAveragePooling2D()(input_feature)
	
	x = Reshape((-1,1))(avg_pool)
	x = Conv1D(1, kernel_size=kernel_size, padding="same", name = "eca_layer_"+str(name), use_bias=False,)(x)
	x = Activation('sigmoid')(x)
	x = Reshape((1, 1, -1))(x)

	output = multiply([input_feature,x])
	return output

def spatial_attention(input_feature, name=""):
	kernel_size = 7

	cbam_feature = input_feature
	
	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	concat = Concatenate(axis=3)([avg_pool, max_pool])

	cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					kernel_initializer='he_normal',
					use_bias=False,
					name = "spatial_attention_"+str(name))(concat)	
	cbam_feature = Activation('sigmoid')(cbam_feature)
		
	return multiply([input_feature, cbam_feature])

def ecbam_block(input_feature,name=""):
	eca_feature = eca_block(input_feature)
	ecbam_feature = spatial_attention(eca_feature, name=name)   
	return ecbam_feature

def get_resnet(input_height, input_width, channel):
    img_input = Input(shape=(input_height,input_width , 3 ))
    # 128,128,3 -> 128,128,64
    x = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(64, (7, 7), data_format=IMAGE_ORDERING)(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # 128,128,64 -> 64,64,128
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), data_format=IMAGE_ORDERING, strides=2)(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # 64,64,128 -> 32,32,256
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), data_format=IMAGE_ORDERING, strides=2)(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = ecbam_block(x,name = 'ecbam1')
    for i in range(9):
        x = identity_block(x, 3, 256, block=str(i))
    x = ecbam_block(x,name = 'ecbam2')
    # 32,32,256 -> 64,64,128
    x = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(x)
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), data_format=IMAGE_ORDERING)(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    # 64,64,128 -> 128,128,64
    x = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(x)
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    x = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING)(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)    

    # 128,128,64 -> 128,128,3
    x = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(x)
    x = Conv2D(channel, (7, 7), data_format=IMAGE_ORDERING)(x)
    x = Activation('tanh')(x)  
    model = Model(img_input,x)
    return model


base_model = get_resnet(128,128,3)
print(base_model.summary())