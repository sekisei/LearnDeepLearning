import tensorflow as tf
from tensorflow import keras
import numpy as np
from model import baseModel
import os
from tqdm import tqdm
import ImageProcessingForAI
from PIL import Image

# GPU無効化用
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class P2P_UNet():
    def __init__(self, load_path = './'):
        #super().__init__(height = height, width = width)

        # Basic U-Net pix2pix example.
        # input  --> Conv2D1   --> Conv2D2 --> Pool2D1 --> Conv2D3 -->   Conv2D4 --> Pool2D2 -->   Conv2D5 --┐
        #                             |                                   |                                  |
        #                             | Residual(Conv2D2+deConv2D2)       | Residual(Conv2D4+deConv2D4)      |
        #                             |                                   |                                  |
        # output <-- deConv2D1 <-- deConv2D2(Up scale) <-- deConv2D3 <-- deConv2D4(Up scale) <-- deConv2D5 --┘

        self.bM_object = baseModel(height = 128, width = 128, dropout_rate = 0.5)
        self.bM = self.bM_object.genBaseModel()

        # Encoding
        self.conv2D5 = self.bM.get_layer('conv2DLayer5').output

        #Decoding: deConv2D5 -->
        self.de_conv2D5 = tf.keras.layers.Conv2DTranspose(
            128,
            (3, 3),
            activation = tf.nn.leaky_relu,
            padding = 'same',
            name = 'de_conv2DLayer5'
        )(self.conv2D5)

        self.de_conv2D4 = tf.keras.layers.Conv2DTranspose(
            64,
            (3, 3),
            activation = tf.nn.leaky_relu,
            padding = 'same',
            strides = (2, 2),
            name = 'de_conv2DLayer4'
        )(self.de_conv2D5)

        self.l_dropout_1_ = tf.keras.layers.Dropout(self.bM_object.d_rate, name = 'l_dropout_1_')(self.de_conv2D4)

        # Residual connection
        self.residual_1 = tf.keras.layers.Add()(
            [self.bM.get_layer('conv2DLayer4').output, self.l_dropout_1_]
        )

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

        self.l_dropout_2_ = tf.keras.layers.Dropout(self.bM_object.d_rate, name = 'l_dropout_2_')(self.de_conv2D2)

        # Residual connection
        self.residual_2 = tf.keras.layers.Add()(
            [self.bM.get_layer('conv2DLayer2').output, self.l_dropout_2_]
        )

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
            activation = 'sigmoid',
            padding = 'same',
            name = 'output_conv2D'
        )(self.de_conv2D1)

        self.unetModel = tf.keras.Model(inputs = self.bM.input, outputs = self.output_conv2D, name = 'P2P_Unet')

    def getModel(self): return self.unetModel

if __name__ == '__main__':

    # 学習用
    epochs = 10
    batch_size = 32

    '''
    p2p = P2P_UNet(load_path = './dataset_for_flow/model')
    model = p2p.getModel()
    generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1 / 255.0)

    input = generator.flow_from_directory(
        './dataset_for_flow/ext_image/',
        target_size = (128,128),
        batch_size = batch_size,
        class_mode = None,
        seed = 1 #同じシードを指定
    )
    ground_truth = generator.flow_from_directory(
        './dataset_for_flow/ext_ground_truth/',
        target_size = (128,128),
        batch_size = batch_size,
        class_mode = None,
        seed = 1 #同じシードを指定
    )

    z = zip(input, ground_truth)

    model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1.0e-4), loss = 'BinaryCrossentropy', metrics = tf.keras.metrics.BinaryAccuracy())

    #学習
    model.fit(
        z,
        steps_per_epoch = input.samples // batch_size,
        #validation_data = valid_gen,
        #validation_steps = valid_gen.samples // batch_size,
        epochs = epochs,
        verbose = 1
    )

    model.save('./save/model')
    '''

    #テスト用
    '''
    generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1 / 255.0)
    input = generator.flow_from_directory(
        './dataset_for_flow/ext_image/',
        target_size = (128,128),
        batch_size = 1,
        class_mode = None,
        seed = 1 #同じシードを指定
    )

    model = tf.keras.models.load_model('./save/model')

    for idx in tqdm(range(0, input.samples), total = input.samples):
        image = input.__next__()
        In = np.reshape(image, (1, 128, 128, 3))
        Out = np.reshape(model(In), (128, 128, 3)) * 255
        tf.keras.preprocessing.image.save_img('./res/' + str(idx) + 'out.png', Out)
    '''

    #'''
    # 画像の受け取り
    pil_image = Image.open('./test_imgs/test5.jpg')
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
    GModel = tf.keras.models.load_model('./save/model')

    stack = []
    for image in tqdm(image_batch):
        val = GModel(
                np.reshape(image / 255.0, (1, 128, 128, 3))
            ).numpy()
        stack.append(val * 255.0)

    res = np.reshape(np.array(stack), (len(stack), 128, 128, 3))
    # AIの出力のバッチを一枚の画像化
    canvas = SC.image_construct(res, pos_list, out_width = width, out_height = height, mode = 'RGB')
    canvas = canvas.resize(canvas.size)
    canvas.save('test2.png')
    #'''

