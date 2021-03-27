from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, GaussianNoise, BatchNormalization, Dense

import json

with open('model_config.json', 'rb') as f:
    m_config = json.load(f)

class Gen(Model):

    def __init__(self):
        super(Gen, self).__init__()
        
        self.gen_layers = []
        for i, layer in enumerate(m_config['generator']):
            self.gen_layers.append(Conv2DTranspose( filters = layer['filters'],
                                                    kernel_size = layer['kernel_size'],   
                                                    padding = layer['padding'], 
                                                    strides = layer['strides'], 
                                                    name = 'g%i'%(i + 1)
                                        )
                                    )

    def call(self, inputs):
        x = self.gen_layers[0](inputs)
        for layer in self.gen_layers[1:]
            x = layer(x)        
        return x

class Dis(Model):

    def __init__(self):
        super(Dis, self).__init__()
        
        self.dis_layers = []
        for i, layer in enumerate(m_config['discriminator']):
            if layer['type'] = 'conv':
                self.dis_layers.append(Conv2D(  filters = layer['filters'],
                                                kernel_size = layer['kernel_size'],   
                                                padding = layer['padding'], 
                                                strides = layer['strides'], 
                                                name = 'd%i_c'%(i + 1)
                                            )
                                        )
            elif layer['type'] = 'ff':
                self.dis_layers.append(Dense(   units = layer['units'],
                                                activation = layer['activation'],
                                                name = 'd%i_f'%(i + 1)
                                            )
                                        )

    def call(self, inputs):
        x = self.dis_layers[0](inputs)
        for layer in self.dis_layers[1:]
            x = layer(x)        
        return x


def gen_loss(d_x_fake):
    return tf.reduce_mean(d_x_fake)

def dis_loss(d_x_real, d_x_fake):
    return tf.reduce_mean(d_x_real - d_x_fake)