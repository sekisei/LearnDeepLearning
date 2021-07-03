import tensorflow as tf
import numpy as np
import cv2
#GPUメモリが足りないので使用できない
'''
@cuda.jit#('void(float32[:,:,:,:], float32[:,:,:,:], float32[:], float32[:])', device = True)
def counter_for_IoU(y_pred, y_label, result, confusion_matrix):
    bIdx = cuda.blockIdx.x #batch
    bIdy = cuda.blockIdx.y #width
    bIdz = cuda.blockIdx.z #height

    #confusion_matrix
    #(1,0) <===> confusion_matrix[1], confusion_matrix[0]
    TP, FN, FP = result[1], result[2], result[3]
    if y_pred[bIdx][bIdy][bIdz][0] == confusion_matrix[1]:
       if y_label[bIdx][bIdy][bIdz][0] == confusion_matrix[1]:
          TP += 1
       FP += 1

    if y_pred[bIdx][bIdy][bIdz][0] == confusion_matrix[0]:
       if y_label[bIdx][bIdy][bIdz][0] == confusion_matrix[1]:
          FN += 1
'''

def counter_for_IoU_on_CPU(y_pred, y_label):
    batch_size = y_pred.shape[0]
    width = y_pred.shape[1]
    height = y_pred.shape[2]
    TP, FN, FP = 0, 0, 0

    for bIdx in range(0, batch_size):
        for bIdy in range(0, width):
            for bIdz in range(0, height):
                if y_pred[bIdx][bIdy][bIdz][0] >= 0.5:
                   if y_label[bIdx][bIdy][bIdz][0] == 1.0:
                      TP += 1
                   FP += 1

                if y_pred[bIdx][bIdy][bIdz][0] < 0.5:
                   if y_label[bIdx][bIdy][bIdz][0] == 1.0:
                      FN += 1
    return TP, FN, FP

def IoU(attention_map, label_image):
    #shape = label_image.shape
    #result = np.zeros((3)).astype(np.float32) # TP, FN, FP
    #confusion_matrix = np.array([0, 1], dtype = np.float32)
    #counter_for_IoU[(shape[0], shape[1], shape[2]), 1](attention_map, label_image, result, confusion_matrix) #GPUメモリが足りないので使用不可
    res = counter_for_IoU_on_CPU(attention_map, label_image)
    return float(res[0]) / float(np.sum(res))

def get_accuracy_on_batch(y_pred, y_label):
    Is_equal = tf.equal(
        tf.cast(
            tf.compat.v1.to_float(tf.greater(y_pred, 0.5)),
            tf.int32
        ),
        tf.cast(
            tf.compat.v1.to_float(y_label),
            tf.int32
        )
    )
    accuracy = tf.reduce_mean(tf.cast(Is_equal, tf.float32))
    return accuracy

def getAttentionMap(model = None, img = None):
    #with tf.GradientTape() as tapeForClassify:
    #    outputsForClassify, layerOutForClassify = model(img)
    with tf.GradientTape() as tapeForSegmentaion:
        outputsForSegmentation, layerOutForSegmentation = model(img, training = True)
    gradsForSegmentaion = tapeForSegmentaion.gradient(outputsForSegmentation, layerOutForSegmentation)
    alpha = tf.math.reduce_sum(gradsForSegmentaion, axis = [1, 2])
    AttentionMap = tf.math.multiply(tf.expand_dims(tf.expand_dims(alpha, 1), 1), layerOutForSegmentation)
    AttentionMap = tf.math.reduce_sum(AttentionMap, -1)
    zero = tf.zeros(tf.shape(AttentionMap))
    AttentionMapRelu = tf.math.maximum(AttentionMap, zero)
    return AttentionMapRelu

def resize_Attention_Map(attention_map = None, img_size = None):
    #anti_NaN = 1.0e-10
    size = [img_size, img_size]
    att_map_with_channel = tf.expand_dims(attention_map, -1)
    att_map_resized = tf.image.resize(att_map_with_channel, size, method = tf.image.ResizeMethod.BILINEAR)
    return att_map_resized

def getHeatMap(AttentionMap):
    #result_on_batch = np.expand_dims(AttentionMap, axis = -1)
    #1.0e-10 -> Anti NaN
    result_on_batch = [np.uint8(AttentionMap[idx] / (AttentionMap[idx].max() + 1.0e-10) * 255.0) for idx in range(0, len(AttentionMap))]
    result_on_batch = np.array(result_on_batch)
    result_on_batch = [cv2.resize(np.expand_dims(result_on_batch[idx], axis = -1), (128, 128), interpolation = cv2.INTER_LANCZOS4) for idx in range(0, len(AttentionMap))]
    result_on_batch = np.array(result_on_batch)
    result_on_batch = [cv2.applyColorMap(result_on_batch[idx], cv2.COLORMAP_JET) for idx in range(0, len(AttentionMap))]
    result_on_batch = np.array(result_on_batch)
    result_on_batch = [cv2.cvtColor(result_on_batch[idx], cv2.COLOR_BGR2RGB) for idx in range(0, len(AttentionMap))]
    return np.array(result_on_batch)

def save_imgs(Heat_Map, target, idx = 0, dir_path = '', name_numbering = 0):
    Heat_Map = (np.float32(Heat_Map[idx])  + target[idx].reshape((128, 128, 3)) / 2.0)
    Heat_Map_img = tf.keras.preprocessing.image.array_to_img(Heat_Map)
    target_img = tf.keras.preprocessing.image.array_to_img(target[idx])
    Heat_Map_img.save(dir_path + 'Attention_Map_' + str(name_numbering)+ '.png')
    target_img.save(dir_path + 'original_' + str(name_numbering)+ '.png')

def weightGrads(gFirst, gSecond, alpha = 1.0, beta = 1.0):
    gFirstMultiplied = [tf.math.multiply(gF, alpha) for gF in gFirst]
    gSecondMultiplied = [tf.math.multiply(gS, beta) for gS in gSecond if gS != None]
    allGrads = [tf.math.add(gF, gS) for gF, gS in zip(gFirstMultiplied, gSecondMultiplied)]
    return allGrads

def get_gradients_with_lossExt(model = None, image_for_classify = None, image_for_segmentation = None, ground_truth_for_classify = None, ground_truth_for_segmentation = None):
    #if image_for_classify == None || ground_truth_for_classify == None || ground_truth_for_segmentation == None: return None
    with tf.GradientTape() as tapeForClassify:
        outputsForClassify, layerOutForClassify = model(image_for_classify)
        LossCL = tf.keras.losses.BinaryCrossentropy(name = 'binary_crossentropy', from_logits = True)(ground_truth_for_classify, outputsForClassify)

    #''' #GAIN
    with tf.GradientTape() as tapeForSegmentaion:
        AttentionMap = getAttentionMap(model = model, img = image_for_segmentation)
        LossE = tf.keras.losses.MeanSquaredError()(
            resize_Attention_Map(attention_map = AttentionMap, img_size = image_for_classify.shape[1]), ground_truth_for_segmentation
        )

    gradsForClassify = tapeForClassify.gradient(LossCL, model.trainable_weights)
    gradsForSegmentaion = tapeForSegmentaion.gradient(LossE, model.trainable_weights)
    allGrads = weightGrads(gradsForClassify, gradsForSegmentaion, alpha = 1.0, beta = 10.0)
    LossExt = tf.reduce_sum(LossCL * 1.0 + LossE * 10.0)
    return allGrads, LossExt
    #'''
    #return tapeForClassify.gradient(LossCL, model.trainable_weights), LossCL
