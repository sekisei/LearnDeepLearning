import tensorflow as tf
import numpy as np
import ImageProcessingForAI
from PIL import Image
from tqdm import tqdm

class dcgan_model():
    def __init__(self, height = 256, width = 256, dropout_rate = 0.5):
        self.inputShape = (height, width, 3)
        self.d_rate = dropout_rate
        
        self.discriminator = tf.keras.Sequential(
            [
                tf.keras.Input(shape = (height, width, 3)),
                tf.keras.layers.Conv2D(16, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2DLayer1'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(32, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2DLayer2'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(self.d_rate),
                tf.keras.layers.AveragePooling2D(pool_size = (2, 2), padding = 'valid'),
                tf.keras.layers.Conv2D(32, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2DLayer3'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2DLayer4'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(self.d_rate),
                tf.keras.layers.AveragePooling2D(pool_size = (2, 2), padding = 'valid'),
                tf.keras.layers.Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2DLayer5'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(128, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2DLayer6'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(self.d_rate),
                tf.keras.layers.Conv2D(1, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2DLayer7'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.GlobalMaxPooling2D(),
            ],
            name = "discriminator",
        )

        # Basic U-Net pix2pix example.
        self.generator_top = tf.keras.Sequential(
            [
                tf.keras.Input(shape = (height, width, 3)),
                tf.keras.layers.Conv2D(16, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2DLayer1_'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(32, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2DLayer2_'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(self.d_rate),
                tf.keras.layers.AveragePooling2D(pool_size = (2, 2), padding='valid'),
                tf.keras.layers.Conv2D(32, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2DLayer3_'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2DLayer4_'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(self.d_rate),
                tf.keras.layers.AveragePooling2D(pool_size = (2, 2), padding='valid'),
                tf.keras.layers.Conv2D(128, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2DLayer5_'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(1, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2DLayer6_'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(self.d_rate),
            ],
            name = 'generator_top',
        )

        #Decoding: deConv2D5 -->
        self.de_conv2D5 = tf.keras.layers.Conv2DTranspose(
            128,
            (3, 3),
            activation = tf.nn.leaky_relu,
            padding = 'same',
            name = 'de_conv2DLayer5'
        )(self.generator_top.output)

        #self.bn_1 = tf.keras.layers.BatchNormalization(epsilon = 1.0e-9)(self.de_conv2D5)

        self.de_conv2D4 = tf.keras.layers.Conv2DTranspose(
            64,
            (3, 3),
            activation = tf.nn.leaky_relu,
            padding = 'same',
            strides = (2, 2),
            name = 'de_conv2DLayer4'
        )(self.de_conv2D5)

        self.l_dropout_1_ = tf.keras.layers.Dropout(self.d_rate, name = 'l_dropout_1_')(self.de_conv2D4)

        # Residual connection
        self.residual_1 = tf.keras.layers.Add()(
            [self.generator_top.get_layer('conv2DLayer4_').output, self.l_dropout_1_]
        )

        #self.bn_2 = tf.keras.layers.BatchNormalization(epsilon = 1.0e-9)(self.residual_1)

        self.de_conv2D3 = tf.keras.layers.Conv2DTranspose(
            32,
            (3, 3),
            activation = tf.nn.leaky_relu,
            padding = 'same',
            name = 'de_conv2DLayer3'
        )(self.residual_1)

        self.de_conv2D2 = tf.keras.layers.Conv2DTranspose(
            32,
            (3, 3),
            activation = tf.nn.leaky_relu,
            padding = 'same',
            strides = (2, 2),
            name = 'de_conv2DLayer2'
        )(self.de_conv2D3)

        self.l_dropout_2_ = tf.keras.layers.Dropout(self.d_rate, name = 'l_dropout_2_')(self.de_conv2D2)

        # Residual connection
        self.residual_2 = tf.keras.layers.Add()(
            [self.generator_top.get_layer('conv2DLayer2_').output, self.l_dropout_2_]
        )

        #self.bn_3 = tf.keras.layers.BatchNormalization(epsilon = 1.0e-9)(self.residual_2)

        self.de_conv2D1 = tf.keras.layers.Conv2DTranspose(
            16,
            (3, 3),
            activation = tf.nn.leaky_relu,
            padding = 'same',
            name = 'de_conv2DLayer1'
        )(self.residual_2)

        self.output_conv2D = tf.keras.layers.Conv2DTranspose(
            3,
            (3, 3),
            activation = 'linear',
            padding = 'same',
            name = 'output_conv2D'
        )(self.de_conv2D1)

        self.generator = tf.keras.Model(inputs = self.generator_top.input, outputs = self.output_conv2D, name = 'Unet_for_GAN')
    
    def get_models(self): return self.discriminator, self.generator

class dcgan(tf.keras.Model):
   
    def __init__(self, D = None, G = None):
        super().__init__()
        self.D_BC_loss_tracker = tf.keras.metrics.Mean(name = 'D: loss_CL')
        self.G_BC_loss_tracker = tf.keras.metrics.Mean(name = 'G: loss_CL')
        self.D_acc_tracker = tf.keras.metrics.BinaryAccuracy(name = 'D: Acc')
        self.G_acc_tracker = tf.keras.metrics.BinaryAccuracy(name = 'G: Acc')
        self.D = D
        self.G = G
        self.input_G = None 
        self.origin = None

    def setData(self, input_G, origin):
        self.input_G = input_G
        self.origin = origin

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    # The data argument will not be used
    def train_step(self, data):
        Input_for_G, Original_image = self.input_G.__next__(), self.origin.__next__()
        Input_for_G = Input_for_G + tf.random.uniform(tf.shape(Input_for_G), minval = 0, maxval = 0.1, seed = 1)
        Original_image = Original_image + tf.random.uniform(tf.shape(Original_image), minval = 0, maxval = 0.1, seed = 1)

        # Create dataset
        G_out = self.G(Input_for_G)

        dsize_input_G = Input_for_G.shape[0]
        dsize_input_D = Original_image.shape[0]
        Label_for_G = tf.ones((dsize_input_G, 1), dtype = tf.int32)
        Input_for_D = tf.concat([G_out, Original_image], axis = 0)
        Label_for_D = tf.concat(
            [
                tf.zeros(
                    (dsize_input_G, 1), dtype = tf.int32
                ), 
                tf.ones(
                    (dsize_input_D, 1), dtype = tf.int32
                )
            ], axis = 0
        )

        #----Train discriminator----
        # Get gradient tape
        with tf.GradientTape() as D_tape:
            D_out_on_train_D = self.D(Input_for_D)
            D_LossCL = self.loss_fn(Label_for_D, D_out_on_train_D)

        # Get gradients and apply
        D_grads = D_tape.gradient(D_LossCL, self.D.trainable_variables)
        self.d_optimizer.apply_gradients(zip(D_grads, self.D.trainable_variables))

        #----Train Generator----
        with tf.GradientTape() as G_tape:
            D_out_on_train_G = self.D(self.G(Input_for_G))
            G_LossCL = self.loss_fn(Label_for_G, D_out_on_train_G)
        
        # Get gradients and apply
        G_grads = G_tape.gradient(G_LossCL, self.G.trainable_variables)
        self.g_optimizer.apply_gradients(zip(G_grads, self.G.trainable_variables))

        # Update states
        self.D_BC_loss_tracker.update_state(D_LossCL)
        self.D_acc_tracker.update_state(Label_for_D, tf.math.sigmoid(D_out_on_train_D))
        self.G_BC_loss_tracker.update_state(G_LossCL)
        self.G_acc_tracker.update_state(Label_for_G, tf.math.sigmoid(D_out_on_train_G))

        res = {
            'D: Loss':self.D_BC_loss_tracker.result(),
            'G: Loss':self.G_BC_loss_tracker.result(),
            'D: Acc':self.D_acc_tracker.result(),
            'G: Acc':self.G_acc_tracker.result()
        }

        return res 

    def setData(self, input_G, origin):
        self.input_G = input_G
        self.origin = origin 
    
    def call(self, inputs):
        pass

    @property
    def metrics(self): return [self.D_BC_loss_tracker, self.G_BC_loss_tracker, self.D_acc_tracker, self.G_acc_tracker]    

if __name__ == '__main__':
    
    epochs = 20
    batch_size = 64

    generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1 / 255.0)
    origin = generator.flow_from_directory(
        './dataset_for_DCGAN/origin/',
        target_size = (128,128),
        batch_size = batch_size,
        class_mode = None,
        seed = 1 #同じシードを指定
    )
    input_G = generator.flow_from_directory(
        './dataset_for_DCGAN/fake/',
        target_size = (128,128),
        batch_size = batch_size,
        class_mode = None,
        seed = 1 #同じシードを指定
    )

    D, G = dcgan_model(height = 128, width = 128, dropout_rate = 0.5).get_models()

    gan = dcgan(D = D, G = G)
    gan.compile(
        d_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0006),
        g_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0006),
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits = True)
    )

    gan.setData(input_G, origin)
    dummy = zip(input_G, origin)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = './log', histogram_freq = 1)

    gan.fit(
        x = dummy, #何故かcallを呼ばれるので対策
        steps_per_epoch = input_G.samples // batch_size,
        epochs = epochs,
        verbose = 1,
        callbacks = [tensorboard_callback]
    )
    
    G.save('./save_model/GAN')
    #'''

    #'''
    generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1 / 255.0)
    input_G_test = generator.flow_from_directory(
        #'./dataset_for_flow_GAN/input_G/',
        './test_imgs/test',
        target_size = (128,128),
        batch_size = 1,
        class_mode = None,
        seed = 1 #同じシードを指定
    )
    G = tf.keras.models.load_model('./save_model/GAN')
    for idx in tqdm(range(0, input_G_test.samples), total = input_G_test.samples):
        image = input_G_test.__next__()
        In = np.reshape(image, (1, 128, 128, 3))
        Out = np.reshape(G(In), (128, 128, 3)) * 255
        tf.keras.preprocessing.image.save_img('./res/' + str(idx) + 'out.png', Out)
        if (idx == 100): break
    #'''

    #'''
    # 画像の受け取り
    pil_image = Image.open('./test_imgs/in1.jpg')
    width, height = pil_image.size

    # 自作クラスの初期化
    SC = ImageProcessingForAI.split_and_reconstruct(width_stride = 128, height_stride = 128, crop_size = 128)

    # 一枚の画像からバッチ作成                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    image_resize = SC.downsize_image_for_crop(pil_image)     #バッチ作成時に都合が良くなるようにリサイズ
    pos_list = SC.save_pos_list(image_resize)                #再構築に必要な切り取り座標を保存
    image_batch = SC.get_image_batch(image_resize, pos_list) #画像バッチ化
    #print(image_batch.shape)

    # 作成したバッチ画像をAIにパス
    # バッチが大きすぎるので分割して渡す
    tf.keras.backend.set_learning_phase(0)
    GModel = tf.keras.models.load_model('./save_model/GAN')

    stack = []
    def sigmoid(x): return 1 / (1 + np.exp(-x))
    for image in tqdm(image_batch):
        val = GModel(
                np.reshape(image / 255.0, (1, 128, 128, 3))
            ).numpy()
        stack.append(
            sigmoid(val) * 255.0
        )

    res = np.reshape(np.array(stack), (len(stack), 128, 128, 3))
    # AIの出力のバッチを一枚の画像化
    canvas = SC.image_construct(res, pos_list, out_width = width, out_height = height, mode = 'RGB')
    canvas = canvas.resize(canvas.size)
    canvas.save('test2.jpg')
    #'''

    # ToDo
    # 1. データセットをoriginももう少し人工物よりにする(Generatorに優しくする)
    # 2. 画像再構築時にノイズになってしまう原因を調べる


