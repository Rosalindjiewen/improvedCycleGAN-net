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



def channel_attention(input_feature, ratio=8, name=""):
	
	channel = input_feature._keras_shape[-1]
	
	shared_layer_one = Dense(channel//ratio,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=False,
							 bias_initializer='zeros',
							 name = "channel_attention_shared_one_"+str(name))
	shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=False,
							 bias_initializer='zeros',
							 name = "channel_attention_shared_two_"+str(name))
	
	avg_pool = GlobalAveragePooling2D()(input_feature)    
	max_pool = GlobalMaxPooling2D()(input_feature)

	avg_pool = Reshape((1,1,channel))(avg_pool)
	max_pool = Reshape((1,1,channel))(max_pool)

	avg_pool = shared_layer_one(avg_pool)
	max_pool = shared_layer_one(max_pool)

	avg_pool = shared_layer_two(avg_pool)
	max_pool = shared_layer_two(max_pool)
	
	cbam_feature = Add()([avg_pool,max_pool])
	cbam_feature = Activation('sigmoid')(cbam_feature)
	
	
	return multiply([input_feature, cbam_feature])

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

def cbam_block(cbam_feature, ratio=8, name=""):
	cbam_feature = channel_attention(cbam_feature, ratio, name=name)
	cbam_feature = spatial_attention(cbam_feature, name=name)
	return cbam_feature


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
    x = cbam_block(x,name = 'cbam1')
    for i in range(9):
        x = identity_block(x, 3, 256, block=str(i))
    x = cbam_block(x,name = 'cbam2')
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