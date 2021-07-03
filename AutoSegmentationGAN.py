import tensorflow as tf
from tensorflow.python.keras.engine.training_utils import cast_to_model_input_dtypes
import model
import DLtool

#まずはGeneratorを作り、そこからに種類のAMを作成する

class ASGAN(tf.keras.Model):
    def __init__(self, Generator, Discriminator):
        super().__init__()
        self.BC_loss_tracker = tf.keras.metrics.Mean(name = 'loss_CL')
        self.D_BC_loss_tracker = tf.keras.metrics.Mean(name = 'D: loss_CL')
        self.G_BC_loss_tracker = tf.keras.metrics.Mean(name = 'G: loss_CL')
        self.D_acc_tracker = tf.keras.metrics.BinaryAccuracy(name = 'D: Acc')
        self.G_acc_tracker = tf.keras.metrics.BinaryAccuracy(name = 'G: Acc')
        self.acc_tracker = tf.keras.metrics.BinaryAccuracy(name = 'Acc')
        self.loss_fn = None
        
        self.Generator = Generator
        self.Discriminator = Discriminator
        self.real_img_generator = None
        self.data = None

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def setData(self, real_img_generator, data): 
        self.real_img_generator = real_img_generator
        self.data = data

    def train_step(self, data):
        real_img = next(self.real_img_generator)
        cl_input, cl_label = next(self.data)
         
        with tf.GradientTape() as tape:
            outputsForClassify, layerOutForClassify = self.Generator(cl_input)
            AttentionMap = DLtool.getAttentionMap(model = self.Generator, img = cl_input)
            AttentionMapInverse = DLtool.getAttentionMap(model = self.Generator, img = cl_input, inverse = -1)
            AMResized = DLtool.resize_Attention_Map(attention_map = AttentionMap, img_size = cl_input.shape[1])
            AMResizedInverse = DLtool.resize_Attention_Map(attention_map = AttentionMapInverse, img_size = cl_input.shape[1])
            AMResized_normalized = AMResized / tf.reduce_max(AMResized)
            AMResized_normalized_inv = AMResized / tf.reduce_max(AMResizedInverse)
            LossCL = self.loss_fn(tf.reshape(cl_label, (len(cl_label), 1)), outputsForClassify)

        # Create datasets
        target_attention = tf.math.multiply(AMResized_normalized, cl_input)
        non_target_attention = tf.math.multiply(AMResized_normalized_inv, cl_input)
        constructed_img = tf.add(target_attention, non_target_attention)
        
        #----Train generator (standard)----
        Grads = tape.gradient(LossCL, self.Generator.trainable_variables)
        self.optimizer.apply_gradients(zip(Grads, self.Generator.trainable_variables))

        dsize_input_D = real_img.shape[0]
        dsize_input_G = cl_input.shape[0]
        Label_for_G = tf.ones((dsize_input_G, 1), dtype = tf.int32)
        Input_for_D = tf.concat([constructed_img, real_img], axis = 0)
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
            D_out_on_train_D = self.Discriminator(Input_for_D)
            D_LossCL = self.loss_fn(Label_for_D, D_out_on_train_D)

        # Get gradients and apply
        D_grads = D_tape.gradient(D_LossCL, self.Discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(D_grads, self.Discriminator.trainable_variables))

        #----Train Generator----
        with tf.GradientTape() as G_tape:
            outputsForClassify, layerOutForClassify = self.Generator(cl_input)
            AttentionMap = DLtool.getAttentionMap(model = self.Generator, img = cl_input)
            AttentionMapInverse = DLtool.getAttentionMap(model = self.Generator, img = cl_input, inverse = -1)
            AMResized = DLtool.resize_Attention_Map(attention_map = AttentionMap, img_size = cl_input.shape[1])
            AMResizedInverse = DLtool.resize_Attention_Map(attention_map = AttentionMapInverse, img_size = cl_input.shape[1])
            AMResized_normalized = AMResized / tf.reduce_max(AMResized)
            AMResized_normalized_inv = AMResized / tf.reduce_max(AMResizedInverse)

            # Create datasets
            target_attention = tf.math.multiply(AMResized_normalized, cl_input)
            non_target_attention = tf.math.multiply(AMResized_normalized_inv, cl_input)
            constructed_img = tf.add(target_attention, non_target_attention)

            D_out_on_train_G = self.Discriminator(constructed_img)
            G_LossCL = self.loss_fn(Label_for_G, D_out_on_train_G)
            
        # Get gradients and apply
        G_grads = G_tape.gradient(G_LossCL, self.Generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(G_grads, self.Generator.trainable_variables))

        # Update states
        self.D_BC_loss_tracker.update_state(D_LossCL)
        self.BC_loss_tracker.update_state(LossCL)
        self.D_acc_tracker.update_state(Label_for_D, tf.math.sigmoid(D_out_on_train_D))
        self.G_BC_loss_tracker.update_state(G_LossCL)
        self.G_acc_tracker.update_state(Label_for_G, tf.math.sigmoid(D_out_on_train_G))
        self.acc_tracker.update_state(cl_label, tf.math.sigmoid(outputsForClassify))

        res = {
            'Loss':self.BC_loss_tracker.result(),
            'D: Loss':self.D_BC_loss_tracker.result(),
            'G: Loss':self.G_BC_loss_tracker.result(),
            'D: Acc':self.D_acc_tracker.result(),
            'G: Acc':self.G_acc_tracker.result(),
            'Acc':self.acc_tracker.result(),
        }

        return res
    
    def call(self, inputs):
        pass

    @property
    def metrics(self): 
        return [
            self.BC_loss_tracker,
            self.D_BC_loss_tracker,
            self.G_BC_loss_tracker,
            self.D_acc_tracker,
            self.G_acc_tracker,
            self.acc_tracker
        ]


if __name__ == '__main__':

    # Create discriminator
    bModel = model.baseModel(height = 128, width = 128, dropout_rate = 0.5).genBaseModel()
    Conv2D6 = tf.keras.layers.Conv2D(
            1,
            (3, 3),
            activation = tf.nn.leaky_relu,
            padding = 'same',
            name = 'conv2DLayer6'
    )(bModel.get_layer('conv2DLayer5').output)
    output = tf.keras.layers.GlobalAveragePooling2D()(Conv2D6)
    Generator = tf.keras.Model(inputs = bModel.input, outputs = [output, bModel.get_layer('conv2DLayer5').output])

    # Create generator
    bModel_ = model.baseModel(height = 128, width = 128, dropout_rate = 0.5).genBaseModel()
    Conv2D6_ = tf.keras.layers.Conv2D(
            1,
            (3, 3),
            activation = tf.nn.leaky_relu,
            padding = 'same',
            name = 'conv2DLayer6_'
    )(bModel_.get_layer('conv2DLayer5').output)
    output_ = tf.keras.layers.GlobalAveragePooling2D()(Conv2D6_)
    Discriminator = tf.keras.Model(inputs = bModel_.input, outputs = output_)

    asgan = ASGAN(Generator = Generator, Discriminator = Discriminator)

    epochs = 1
    batch_size = 32

    generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1 / 255.0)

    train_gen = generator.flow_from_directory(
        '../datasets/dataset_for_flow/train',
        target_size = (128,128),
        batch_size = batch_size,
        class_mode = "binary",
        shuffle = True
    )

    real_img_gen = generator.flow_from_directory(
        '../datasets/dataset_for_flow/train',
        target_size = (128,128),
        batch_size = batch_size,
        class_mode = None,
        shuffle = True
    )

    valid_gen = generator.flow_from_directory(
        '../datasets/dataset_for_flow/valid',
        target_size = (128,128),
        batch_size = batch_size,
        class_mode = "binary",
        shuffle = True
    )

    asgan.setData(real_img_gen, train_gen)

    asgan.compile(
        d_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0006),
        g_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0006),
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits = True)
    )

    #学習
    #'''
    asgan.fit(
        train_gen,
        steps_per_epoch = train_gen.samples // batch_size,
        #validation_data = valid_gen,
        #validation_steps = valid_gen.samples // batch_size,
        epochs = epochs,
        verbose = 1
    )
    bModel.save('../save/asgan/')
    #'''

    '''
    bModel = tf.keras.models.load_model('../save/asgan/')
    input_img = next(train_gen)[0]
    with tf.GradientTape() as tapeForAll:
        AttentionMap = DLtool.getAttentionMap(model = bModel, img = input_img, inverse = 1)
        #AMResized = DLtool.resize_Attention_Map(attention_map = AttentionMap, img_size = seg_input.shape[1])

    HM = DLtool.getHeatMap(AttentionMap.numpy())
    for i,d in enumerate(zip(input_img, HM)):
        di, hm = d[0], d[1]
        ii = tf.keras.preprocessing.image.array_to_img(di)
        ihm = tf.keras.preprocessing.image.array_to_img(hm)
        ihm.save('../res/HM' + str(i) + '.png')
        ii.save('../res/input' + str(i) + '.png')
    '''