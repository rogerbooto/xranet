import tensorflow as tf
import tensorflow_addons as tfa # For Instance normalization; from https://www.tensorflow.org/addons/tutorials/layers_normalizations
from tensorflow.keras import Model
import tensorflow.keras.layers as tfkl
import tensorflow.keras.initializers as tfki
import tensorflow.keras.regularizers as tfkr
class XRANet(Model):
    def __init__(self, 
                 filters, 
                 depth,
                 outputShape=(256,256, 1), 
                 dropout=0.5, 
                 kernel_size=3, 
                 poolSize=(2, 2), 
                 padding="same", 
                 initializer=tfki.GlorotUniform(), 
                 regularizer=tfkr.l1_l2(), 
                 activation=tf.nn.relu, 
                 normalize_input=None, 
                 groups=2, 
                 name="XRANet", 
                 light=True, 
                 **kwargs):
        """Custom Keras model. 
            Architecture: Encoder-decoder
            Type: Unet
            Implemented distance loss: No (as of Nov. Mars 20th 2021)

        Args:
            filters ([type]): [description]
            depth ([type]): [description]
            dropout (float, optional): [description]. Defaults to 0.5.
            kernel_size (int, optional): [description]. Defaults to 3.
            poolSize (tuple, optional): [description]. Defaults to (2, 2).
            padding (str, optional): [description]. Defaults to "same".
            activation ([type], optional): [description]. Defaults to tf.nn.relu.
            name (str, optional): [description]. Defaults to "XRANet".

        Raises:
            ValueError: [description]
        """
        super(XRANet, self).__init__(name=name, **kwargs)

        self.modelName = name

        self.activation = activation

        self.normalize_input = normalize_input

        self.groups = groups

        self.filters = filters

        self.dropout = dropout

        self.kernel_size = kernel_size

        self.depth = depth

        self.padding = padding

        self.poolSize = poolSize

        self.__outputShape = outputShape

        self.light = light
        
        self.initializer = initializer

        self.regularizer = regularizer

        if self.__checkKernelSize():
            raise ValueError("Kernel size must be an odd number")
    
    def __validateArchitecture(self, inputShape):
        """[summary]

        Args:
            inputShape ([type]): [description]

        Raises:
            ValueError: [description]
        """
        tmp_inputShape = inputShape.as_list()[1:] if not isinstance(inputShape, list) else inputShape[0].as_list()[1:]

        for _  in range(self.depth):
            tmp_inputShape[0] = tmp_inputShape[0] // self.poolSize[0]
            tmp_inputShape[1] = tmp_inputShape[1] // self.poolSize[1] 

            if tmp_inputShape[0] < 2 or tmp_inputShape[1] < 2:
                raise ValueError("Cannot construct architecture with input shape: {} and pool size: {}".format(inputShape, self.poolSize))

    def __checkKernelSize(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.kernel_size % 2 == 0

    def __buildMultipleInput(self, inputShape):
        inputs = []

        inputs.append(tfkl.Conv2D(self.filters, kernel_size= self.kernel_size, padding=self.padding, name="input1_conv"))
        inputs.append(tfkl.Conv2D(self.filters, kernel_size= self.kernel_size, padding=self.padding, name="input2_conv"))
        inputs.append(tfkl.SeparableConv2D(self.filters, kernel_size= self.kernel_size, padding=self.padding, name="inputs_separable_conv"))

        return inputs
    
    def __buildAttentions(self, encoder=False, prefix="de"):

        attentions = []

        for i in range(self.depth):

            block = []

            block.append(tfkl.Conv2D(self.filters * (2**(self.depth - (i+1))), strides=(1, 1), kernel_size= self.kernel_size, padding=self.padding, name="{}_attention_conv_{}_1".format(prefix, i+1)))

            block.append(tfkl.Conv2D(self.filters * (2**(self.depth - (i+1))), strides=(1, 1), kernel_size= self.kernel_size, padding=self.padding, name="{}_attention_conv_{}_2".format(prefix, i+1)))

            block.append(tfkl.Activation(tf.nn.swish, name="{}_attention_activation_{}_1".format(prefix, i+1)))

            block.append(tfkl.Conv2D(1, strides=(1, 1), kernel_size= self.kernel_size, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, activation= tf.nn.relu, padding=self.padding, name="{}_attention_conv_{}_3".format(prefix, i+1)))

            block.append(tfkl.Conv2D(self.filters * (2**(i+1)), kernel_size= self.kernel_size, padding= self.padding, name="{}_attention_conv_{}_4".format(prefix, i+1)))

            attentions.append(block)

        return attentions
    
    def __buildEncoder(self, inputShape):
        """[summary]

        Args:
            inputShape ([type]): [description]

        Returns:
            [type]: [description]
        """
        self.__validateArchitecture(inputShape)

        encoder = []

        for i in range(self.depth):

            block = []
            block.append(tfkl.BatchNormalization(name="en_bnor_{}_1".format(i+1)))
            block.append(tfkl.Conv2D(self.filters * (2**(i+1)), kernel_size= 1, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, activation= self.activation, padding=self.padding, name="en_scale_{}_1".format(i+1)))

            redux = 4

            #Inception stack 1 - Begin
            # 1x1 conv
            block.append(tfkl.Conv2D((self.filters * (2**(i+1))), kernel_size= 1, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, activation= self.activation, padding=self.padding, name="en_incp_1x1_{}_1".format(i+1)))
            # 3x3 conv
            block.append(tfkl.Conv2D(1 if self.filters // redux < 1 else (self.filters * (2**(i+1))) // redux, kernel_size= 1, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, activation= self.activation, padding=self.padding, name="en_incp_cr3x3_{}_1".format(i+1))) #Computation reduction 
            block.append(tfkl.Conv2D(self.filters * (2**(i+1)), kernel_size= 3, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, activation= self.activation, padding=self.padding, name="en_incp_3x3_{}_1".format(i+1)))
            # 5x5 conv
            block.append(tfkl.Conv2D(1 if self.filters // redux < 1 else (self.filters * (2**(i+1))) // redux, kernel_size= 1, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, activation= self.activation, padding=self.padding, name="en_incp_cr5x5_{}_1".format(i+1))) #Computation reduction 
            block.append(tfkl.Conv2D(self.filters * (2**(i+1)), kernel_size= 5, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, activation= self.activation, padding=self.padding, name="en_incp_5x5_{}_1".format(i+1)))
            # 7x7 conv
            block.append(tfkl.Conv2D(1 if self.filters // redux < 1 else (self.filters * (2**(i+1))) // redux, kernel_size= 1, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, activation= self.activation, padding=self.padding, name="en_incp_cr7x7_{}_1".format(i+1))) #Computation reduction 
            block.append(tfkl.Conv2D(self.filters * (2**(i+1)), kernel_size= 7, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, activation= self.activation, padding=self.padding, name="en_incp_7x7_{}_1".format(i+1)))
            
            block.append(tfkl.MaxPool2D(pool_size= 3, strides=(1,1), padding= self.padding, name="en_incp_pool_{}".format(i+1)))
            #Inception stack 1 - End

            block.append(tfkl.Conv2D(self.filters * (2**(i+1)), kernel_size= self.kernel_size, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, activation= self.activation, padding=self.padding, name="en_conv_{}_1".format(i+1)))
            block.append(tfkl.BatchNormalization(name="en_bnor_{}_2".format(i+1)))
            block.append(tfkl.Conv2D(self.filters * (2**(i+1)), kernel_size= self.kernel_size, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, activation= self.activation, padding=self.padding, name="en_conv_{}_2".format(i+1)))
            block.append(tfkl.Dropout(self.dropout, name="en_drop_{}_1".format(i+1)))
            block.append(tfkl.Conv2D(self.filters * (2**(i+1)), kernel_size= self.kernel_size, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, activation= self.activation, padding=self.padding, name="en_conv_{}_3".format(i+1)))

            block.append(tfkl.MaxPool2D(pool_size=self.poolSize, name="en_pool_{}".format(i+1)))
            encoder.append(block)    

        return encoder

    def __buildBottleneck(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        bottleneck = []
        
        bottleneck.append(tfkl.Conv2D(self.filters  * (2**self.depth), 
            kernel_size=self.kernel_size, 
            padding=self.padding, 
            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, activation= self.activation,                                                                    
            name="bnk_conv_1"))
                                  
        bottleneck.append(tfkl.Conv2D(self.filters  * (2**self.depth), 
            kernel_size=self.kernel_size, 
            padding=self.padding, 
            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, activation= self.activation,                                                                    
            name="bnk_conv_2"))

        bottleneck.append(tfkl.Conv2D(self.filters  * (2**self.depth), 
            kernel_size=self.kernel_size, 
            padding=self.padding, 
            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, activation= self.activation,                                                                    
            name="bnk_conv_3"))
                                  
        bottleneck.append(tfkl.Conv2D(self.filters  * (2**self.depth), 
            kernel_size=self.kernel_size, 
            padding=self.padding, 
            kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, activation= self.activation,                                                                    
            name="bnk_conv_4"))

        return bottleneck

    def __buildDecoder(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        decoder = []

        for i in range(self.depth):
            block = []
            
            block.append(tfkl.UpSampling2D(size=self.poolSize, name="de_upsamp_{}_1".format(i+1), interpolation="bilinear"))
            block.append(tfkl.Conv2D(self.filters * (2**(self.depth - (i+1))), kernel_size= self.kernel_size, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, activation= self.activation, padding=self.padding, name="de_conv_{}_1".format(i+1)))
            block.append(tfkl.BatchNormalization(name="de_bnor_{}".format(i+1)))
            block.append(tfkl.Conv2D(self.filters * (2**(self.depth - (i+1))), kernel_size= self.kernel_size, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, activation= self.activation, padding=self.padding, name="de_conv_{}_2".format(i+1)))
            block.append(tfkl.Dropout(self.dropout, name="de_drop_{}".format(i+1)))
            
            block.append(tfkl.Conv2D(self.filters * (2**(self.depth - (i+1))), kernel_size= self.kernel_size, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, activation= self.activation, padding=self.padding, name="de_conv_{}_3".format(i+1)))

            decoder.append(block)

        return decoder

    def __buildOutputs(self):
        outputs = []

        outputs.append(tfkl.Conv2D(self.__outputShape[-1], kernel_size= 1, padding= self.padding, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, activation= tf.nn.sigmoid, name="output"))

        return outputs
    
    def __instanceNormalization(self):
        return tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform",
                                   name="inst_norm")
    
    def __layerNormalization(self):
        return tf.keras.layers.LayerNormalization(axis=1 , center=True , scale=True, name="layer_norm")

    def __groupNormalization(self, groups=2):
        return tfa.layers.GroupNormalization(groups=groups, axis=3, name="group_norm")

    def build(self, inputShape):
        """[summary]

        Args:
            inputShape ([type]): [description]
        """

        if not self.normalize_input is None and self.normalize_input.lower() == "instance":
            self.__normalized_input = self.__instanceNormalization()
        if not self.normalize_input is None and self.normalize_input.lower() == "layer":
            self.__normalized_input = self.__layerNormalization()
            
        if not self.normalize_input is None and self.normalize_input.lower() == "group":
            if  self.groups > 1:
                self.__normalized_input = self.__groupNormalization(self.groups)
            else:
                raise ValueError("Groups must be at least 2")

        self.__encoder = self.__buildEncoder(inputShape)
        self.__bottleneck = self.__buildBottleneck()
        self.__decoder = self.__buildDecoder()
        self.__outputs = self.__buildOutputs()
        self.__attentionsDec = self.__buildAttentions()
        self.__attentionsEnc = self.__buildAttentions(encoder=True, prefix="en")
        
        if isinstance(inputShape, list):
            self.__multiInput = self.__buildMultipleInput(inputShape)

    def call(self, inputs, training=None):
        """Model inference

        Args:
            x ([tf.Tensor]): Graph inputs

        Returns:
            [type]: [description]
        """
        to_concatenate = []

        x = None

        if isinstance(inputs, list):
            input1 = self.__multiInput[0](inputs[0])
            input2 = self.__multiInput[1](inputs[1])

            inputs = tfkl.concatenate([input1, input2])

            inputs = self.__multiInput[2](inputs)

        if not self.normalize_input is None:
            inputs = self.__normalized_input(inputs)

        # TODO: Manage multiple input models
        concat_step = 0
        for i, (encoderBlock, attentionsBlock) in enumerate(zip(self.__encoder, self.__attentionsEnc)):

            if (encoderBlock[0].name.find("drop_") != -1) and training == False:
                if i == 0:
                    x = inputs

            if i == 0:
                x = encoderBlock[0](inputs)
            else:
                x = encoderBlock[0](x)
            

            # Skip connection
            skip = x

            blockSize = len(encoderBlock)

            for j in range(1, blockSize):

                if (encoderBlock[j].name.find("drop_") != -1) and training == False:
                    continue

                if j == 1:
                    #scale the number of filters
                    skip = encoderBlock[j](x)

                if encoderBlock[j].name.find("1x1") != -1:
                    conv1 = encoderBlock[j](x)
                elif encoderBlock[j].name.find("cr3x3") != -1:
                    conv3 = encoderBlock[j](x)
                    conv3 = encoderBlock[j+1](conv3)
                    j += 2
                elif encoderBlock[j].name.find("cr5x5") != -1:
                    conv5 = encoderBlock[j](x)
                    conv5 = encoderBlock[j+1](conv5)
                    j += 2
                elif encoderBlock[j].name.find("cr7x7") != -1:
                    conv7 = encoderBlock[j](x)
                    conv7 = encoderBlock[j+1](conv7)
                    x = tfkl.concatenate([conv1, conv3, conv5, conv7], axis=-1, name="incp_cnct_{}".format(i+1)) # Multi scale concatenate 1 without max pool
                    j += 2

                elif encoderBlock[j].name.find("incp_pool") != -1:
                    pool = encoderBlock[j](x)

                    x = tfkl.concatenate([conv1, conv3, conv5, conv7, pool], axis=-1, name="incp_pool_cnct_{}".format(i+1)) # Multi scale concatenate 2 with max pool

                    #Attention mechanism
                    concat = x
                    theta = attentionsBlock[0](pool)
                    phi = attentionsBlock[1](concat)
                    addition = tfkl.add(inputs=[theta, phi], name="en_attention_add_{}_1".format(i+1))
                    activation = attentionsBlock[2](addition)
                    rate = attentionsBlock[3](activation)
                    x = tfkl.multiply(inputs=[pool, rate], name="en_attention_mul_{}_1".format(i+1))

                    #Feature space scaling
                    x = attentionsBlock[4](x)
                    #Will be added after concatenation to preserve residual integrity
                    x = tfkl.Add()([x, skip])

                    to_concatenate.append(x)
                    concat_step += 1
                elif encoderBlock[j].name.find("incp") == -1:
                    x = encoderBlock[j](x)

        to_cncttmp = []
        concat_step //= self.depth
        for idx in range(0, len(to_concatenate), concat_step):
            obj = {}

            for step in range(concat_step):
                obj["{}".format(step)] = to_concatenate[idx + step]
            
            to_cncttmp.append(obj)

        to_concatenate = to_cncttmp

        del to_cncttmp

        to_concatenate.reverse()

        for layer in self.__bottleneck:
            if (layer.name.find("drop_") != -1) and training == False:
                continue

            x = layer(x)

        for i, (decoderBlock, attentionsBlock) in enumerate(zip(self.__decoder, self.__attentionsDec)):
            # Upsampling
            if i == 0:
                x = decoderBlock[0](x) 
            else:
                x = decoderBlock[0](prev)
            
            blockSize = len(decoderBlock) - 1

            for j in range(1, blockSize):

                if j == 1:
                    x = decoderBlock[j](x) # Convolution
                    upConv = x

                    x = tfkl.concatenate([x, to_concatenate[i]["0"]]) # Concatenate wth encoder features 

                    for cstep in range(1, concat_step):
                        x = tfkl.concatenate([x, to_concatenate[i]["{}".format(cstep)]], name="de_cnct_{}".format(i+1))

                    concat = x
                    theta = attentionsBlock[0](upConv) # Attention conv1
                    phi = attentionsBlock[1](concat) # Attention conv 2
                    
                    addition = tfkl.add(inputs=[theta, phi], name="attention_add_{}_1".format(i+1)) # Check feture alignments
                    
                    activation = attentionsBlock[2](addition) # Assign weights to features
                    
                    rate = attentionsBlock[3](activation)
                    
                    x = tfkl.multiply(inputs=[upConv, rate], name="attention_mul_{}_1".format(i+1))
                    x = attentionsBlock[4](x)
                elif (decoderBlock[j].name.find("drop_") != -1) and training == False:
                    continue
                else:
                    x = decoderBlock[j](x)
            
            if i == len(self.__decoder) - 1:
                x = decoderBlock[-1](x)
            else:
                prev = decoderBlock[-1](x)

        # TODO: Manage multiple output models
        for block in self.__outputs:
            x = block(x)

        return x
