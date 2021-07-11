
#from tool import DLtool
from tensorflow import keras
import tensorflow as tf
import model
from PIL import Image, ImageChops
import numpy as np
import DLtool

#テスト用
import ImageProcessingForAI
import os
from tqdm import tqdm

# GPU無効化用
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class GAINModel(tf.keras.Model):
    #layerName, height = 128, width = 128, dropout_rate = 0.5
    def __init__(self, inputs, outputs):
        super().__init__(inputs = inputs, outputs = outputs)
        self.BC_loss_tracker = tf.keras.metrics.Mean(name = 'loss_CL')
        self.LE_loss_tracker = tf.keras.metrics.Mean(name = 'loss_Ext')
        self.acc_tracker = tf.keras.metrics.BinaryAccuracy()
        self.iou_tracker = tf.keras.metrics.MeanIoU(num_classes = 2)
        self.ext_data = None

    def train_step(self, data):
        cl_input, cl_label = data
        seg_input = self.ext_data['train']['seg_input'].__next__()
        seg_ground_truth = self.ext_data['train']['seg_ground_truth'].__next__()

        with tf.GradientTape() as tapeForAll:
            outputsForClassify, layerOutForClassify = self(cl_input) #y_pred = model(x)
            AttentionMap = DLtool.getAttentionMap(model = self, img = seg_input)
            AMResized = DLtool.resize_Attention_Map(attention_map = AttentionMap, img_size = seg_input.shape[1])
            AMResized_normalized = AMResized / tf.reduce_max(AMResized)
            AMResized_binary = tf.cast(
                tf.compat.v1.to_float(tf.greater(AMResized_normalized, 0.5)),
                tf.int32
            )
            LossCL = tf.keras.losses.BinaryCrossentropy(name = 'binary_crossentropy', from_logits = True)(cl_label, outputsForClassify)
            LossE = tf.keras.losses.MeanSquaredError(name = 'MSE')(AMResized_binary, seg_ground_truth)
            LossExt = LossCL * 1.0 + LossE * 50.0

        allGrads = tapeForAll.gradient(LossExt, self.trainable_variables)
        self.optimizer.apply_gradients(zip(allGrads, self.trainable_variables))

        self.BC_loss_tracker.update_state(LossCL)
        self.LE_loss_tracker.update_state(LossE)
        self.acc_tracker.update_state(cl_label, tf.math.sigmoid(outputsForClassify))
        self.iou_tracker.update_state(seg_ground_truth, AMResized_binary)
        res = {
            'LossCL': self.BC_loss_tracker.result(),
            'LossExt': self.LE_loss_tracker.result(),
            'BinaryAccuracy': self.acc_tracker.result(),
            'IoU': self.iou_tracker.result()
        }

        return res

    def test_step(self, data):
        cl_input, cl_label = data
        seg_input = self.ext_data['valid']['seg_input'].__next__()
        seg_ground_truth = self.ext_data['valid']['seg_ground_truth'].__next__()

        with tf.GradientTape() as tapeForAll:
            outputsForClassify, layerOutForClassify = self(cl_input, training = True) #y_pred = model(x)
            AttentionMap = DLtool.getAttentionMap(model = self, img = seg_input)
            AMResized = DLtool.resize_Attention_Map(attention_map = AttentionMap, img_size = seg_input.shape[1])
            AMResized_normalized = AMResized / tf.reduce_max(AMResized)
            AMResized_binary = tf.cast(
                tf.compat.v1.to_float(tf.greater(AMResized_normalized, 0.5)),
                tf.int32
            )
            LossCL = tf.keras.losses.BinaryCrossentropy(name = 'binary_crossentropy', from_logits = True)(cl_label, outputsForClassify)
            LossE = tf.keras.losses.MeanSquaredError(name = 'MSE')(AMResized_binary, seg_ground_truth)

        self.BC_loss_tracker.update_state(LossCL)
        self.LE_loss_tracker.update_state(LossE)
        self.acc_tracker.update_state(cl_label, tf.math.sigmoid(outputsForClassify))
        self.iou_tracker.update_state(seg_ground_truth, AMResized_binary)
        res = {
            'LossCL': self.BC_loss_tracker.result(),
            'LossExt': self.LE_loss_tracker.result(),
            'BinaryAccuracy': self.acc_tracker.result(),
            'IoU': self.iou_tracker.result()
        }
        return res

    @property
    def metrics(self): return [self.BC_loss_tracker, self.LE_loss_tracker, self.acc_tracker, self.iou_tracker]

    def set_ext_data(self, gen): self.ext_data = gen

if __name__ == '__main__':
    batch_size = 64
    epochs = 10

    #tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    #tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    #**************************二度割注意
    generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1 / 255.0)

    # 学習用
    # Batch Normalization入れてから様子がおかしい
    # 過学習を起こしているようなので、フィルター数を減らす
    # 特徴抽出を助けるためにlayer noramalizationを採用する
    #'''
    base = model.baseModel(height = 128, width = 128, dropout_rate = 0.3)
    bModel = base.genBaseModel()
    '''
    Conv2D6 = tf.keras.layers.Conv2D(
            1,
            (3, 3),
            activation = tf.nn.leaky_relu,
            padding = 'same',
            name = 'conv2DLayer6'
    )(bModel.get_layer('conv2DLayer5').output)
    output = tf.keras.layers.GlobalAveragePooling2D()(Conv2D6)
    '''
    GModel = GAINModel(inputs = bModel.input, outputs = [bModel.output, bModel.get_layer('conv2DLayer5').output])

    train_gen = generator.flow_from_directory(
        '../datasets/dataset_for_flow/train',
        target_size = (128,128),
        batch_size = batch_size,
        class_mode = "binary",
        shuffle = True
    )

    valid_gen = generator.flow_from_directory(
        '../datasets/dataset_for_flow/valid',
        target_size = (128,128),
        batch_size = batch_size,
        class_mode = "binary",
        shuffle = True
    )

    ext_gen = {'train':None, 'valid':None}
    ext_train_gen = {
        'seg_input': generator.flow_from_directory(
            '../datasets/dataset_for_flow/ext_image',
            target_size = (128,128),
            batch_size = batch_size,
            class_mode = None,
            seed = 1 #同じシードを指定
        ),
        'seg_ground_truth': generator.flow_from_directory(
            '../datasets/dataset_for_flow/ext_ground_truth',
            target_size = (128,128),
            batch_size = batch_size,
            class_mode = None,
            color_mode = 'grayscale',
            seed = 1 #同じシードを指定
        )
    }
    ext_gen['train'] = ext_train_gen
    ext_gen['valid'] = ext_train_gen #***************データセットが揃うまで***************

    GModel.set_ext_data(ext_gen)
    GModel.compile(optimizer = tf.keras.optimizers.Adam(lr = 1.0e-5))

    #学習
    GModel.fit(
        train_gen,
        steps_per_epoch = train_gen.samples // batch_size,
        #validation_data = valid_gen,
        #validation_steps = valid_gen.samples // batch_size,
        epochs = epochs,
        verbose = 1
    )

    GModel.save('../save/dataset_for_flow/gain')
    '''
    seg_input = ext_gen['valid']['seg_input'].__next__()
    seg_ground_truth = ext_gen['valid']['seg_ground_truth'].__next__()
    with tf.GradientTape() as tapeForAll:
        AttentionMap = DLtool.getAttentionMap(model = GModel, img = seg_input)
        #AMResized = DLtool.resize_Attention_Map(attention_map = AttentionMap, img_size = seg_input.shape[1])

    HM = DLtool.getHeatMap(AttentionMap.numpy())
    for i,d in enumerate(zip(seg_ground_truth, seg_input, HM)):
        dg, di, hm = d[0], d[1], d[2]
        ig = tf.keras.preprocessing.image.array_to_img(dg)
        ii = tf.keras.preprocessing.image.array_to_img(di)
        ihm = tf.keras.preprocessing.image.array_to_img(hm)
        ig.save('./test_imgs/resultImg/label' + str(i) + '.png')
        ihm.save('./test_imgs/resultImg/HM' + str(i) + '.png')
        ii.save('./test_imgs/resultImg/input' + str(i) + '.png')
    '''
    #'''

    # 簡易テスト用
    '''
    GModel = tf.keras.models.load_model('/home/kai/GAIN/complete/dataset_for_flow2/model')
    GModel.summary()

    gen = generator.flow_from_directory(
        '/home/kai/GAIN/complete/dataset_for_flow2/test/',
        target_size = (128,128),
        batch_size = 1,
        class_mode = None,
    )

    for i in range(0, gen.samples):
        seg_input = gen.__next__()
        with tf.GradientTape() as tapeForAll: AttentionMap = DLtool.getAttentionMap(model = GModel, img = seg_input)
        HM = DLtool.getHeatMap(AttentionMap.numpy())
        seg_input = tf.keras.preprocessing.image.array_to_img(seg_input[0])
        HM = tf.keras.preprocessing.image.array_to_img(HM[0])
        image = Image.blend(seg_input, HM, 0.15)
        image.save('/home/kai/GAIN/complete/dataset_for_flow2/test/input' + str(i) + '.png')
    '''

    '''
    # 画像再構築付きのテスト
    # 画像の受け取り
    pil_image = Image.open('./test4.jpg')
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
    GModel = tf.keras.models.load_model('/home/kai/GAIN/complete/dataset_for_flow2/model')

    stack = []
    for image in image_batch:
        AttentionMap = DLtool.getAttentionMap(model = GModel, img = np.reshape(image, (1, 128, 128, 3)))
        HM = DLtool.getHeatMap(AttentionMap.numpy())
        stack.append(HM)
        #AMResized = DLtool.resize_Attention_Map(attention_map = AttentionMap, img_size = 128)
        #AMResized_normalized = AMResized / tf.reduce_max(AMResized) * 255
        #AMResized_normalized = np.reshape(AMResized_normalized.numpy(), (128, 128))
        #AM = Image.fromarray(AMResized_normalized.astype(np.uint8), 'L').convert('RGB')
        #AM = np.asarray(AM)
        #stack.append(AM)
    res = np.reshape(np.array(stack), (len(stack), 128, 128, 3))

    # AIの出力のバッチを一枚の画像化
    canvas = SC.image_construct(res, pos_list, out_width = width, out_height = height)
    canvas = canvas.resize(canvas.size)
    canvas.save('test2.png')
    '''

    # データセット作成用
    '''
    data_gen = generator.flow_from_directory(
        '/home/kai/GAIN/complete/dataset_for_flow_moss/train/',
        target_size = (128,128),
        batch_size = 1,
        class_mode = None,
    )
    GModel = tf.keras.models.load_model('/home/kai/GAIN/complete/dataset_for_flow2/model')
    for idx in tqdm(range(0, data_gen.samples)):
        image = data_gen.__next__()
        AttentionMap = DLtool.getAttentionMap(model = GModel, img = image)
        AMResized = DLtool.resize_Attention_Map(attention_map = AttentionMap, img_size = 128)
        AMResized_normalized = AMResized / tf.reduce_max(AMResized) * 255
        AMResized_normalized = np.reshape(AMResized_normalized.numpy(), (128, 128))
        AMResized_normalized = np.stack([AMResized_normalized, AMResized_normalized, AMResized_normalized], 2)
        AM = Image.fromarray(AMResized_normalized.astype(np.uint8), 'RGB')
        image = Image.fromarray(np.reshape(image * 255, (128, 128, 3)).astype(np.uint8), 'RGB')
        out = ImageChops.subtract(image, AM)
        #out = Image.composite(Image.effect_noise((128, 128), 20), image, AM)
        out.save('/home/kai/GAIN/complete/dataset_for_flow_moss/dataset/input/images/' + str(idx) + 'test.png')
        AM.save('/home/kai/GAIN/complete/dataset_for_flow_moss/dataset/segment/images/' + str(idx) + 'test.png')
        image.save('/home/kai/GAIN/complete/dataset_for_flow_moss/dataset/ground_truth/images/' + str(idx) + 'test.png')

    '''
