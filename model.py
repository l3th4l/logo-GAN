from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, GaussianNoise, BatchNormalization

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
                                                    name = 'e%i'%(i + 1)
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
            self.dis_layers.append(Conv2D(  filters = layer['filters'],
                                            kernel_size = layer['kernel_size'],   
                                            padding = layer['padding'], 
                                            strides = layer['strides'], 
                                            name = 'e%i'%(i + 1)
                                        )
                                    )

    def call(self, inputs):
        x = self.dis_layers[0](inputs)
        for layer in self.dis_layers[1:]
            x = layer(x)
        
        return x


def gen_loss():
    return 0 

def dis_loss():
    return 0