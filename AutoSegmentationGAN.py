import tensorflow as tf
import GAINModel
import DLtool

class ASGAN(tf.keras.Model):
    def __init__(self, inputs, outputs):
        super().__init__(inputs = inputs, outputs = outputs)
        self.BC_loss_tracker = tf.keras.metrics.Mean(name = 'loss_CL')
        self.LE_loss_tracker = tf.keras.metrics.Mean(name = 'loss_Ext')
        self.acc_tracker = tf.keras.metrics.BinaryAccuracy()
        self.ext_data = None

    def train_step(self, data):
        cl_input, cl_label = data
        
        with tf.GradientTape() as tapeForAll:
            outputsForClassify, layerOutForClassify = self(cl_input)
            AttentionMap = DLtool.getAttentionMap(model = self, img = cl_input)
            AMResized = DLtool.resize_Attention_Map(attention_map = AttentionMap, img_size = cl_input.shape[1])
            AMResized_normalized = AMResized / tf.reduce_max(AMResized)
            AMResized_binary = tf.cast(
                tf.compat.v1.to_float(tf.greater(AMResized_normalized, 0.5)),
                tf.int32
            )




            #'''
            LossCL = tf.keras.losses.BinaryCrossentropy(name = 'binary_crossentropy', from_logits = True)(cl_label, outputsForClassify)
            #LossE = tf.keras.losses.MeanSquaredError(name = 'MSE')(AMResized_binary, seg_ground_truth)
            LossExt = LossCL * 1.0# + LossE * 50.0
            #'''

        #'''
        allGrads = tapeForAll.gradient(LossExt, self.trainable_variables)
        self.optimizer.apply_gradients(zip(allGrads, self.trainable_variables))

        self.BC_loss_tracker.update_state(LossCL)
        #self.LE_loss_tracker.update_state(LossE)
        self.acc_tracker.update_state(cl_label, tf.math.sigmoid(outputsForClassify))
        #self.iou_tracker.update_state(seg_ground_truth, AMResized_binary)
        res = {
            'LossCL': self.BC_loss_tracker.result(),
            'LossExt': self.LE_loss_tracker.result(),
            'BinaryAccuracy': self.acc_tracker.result()
        }
        #'''

        return res
