import tensorflow as tf
import numpy as np
from PIL import Image 

class createSegmentationDataset(tf.keras.preprocessing.image.ImageDataGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setData(self, directory):
        img_name = []
        img_path = tf.data.Dataset.list_files(directory, seed = 1, shuffle = True)
        for path in img_path: img_name.append(path.numpy().decode())
        for name in img_name: yield tf.keras.preprocessing.image.load_img(name)

    def flow_from_directory(self, path1, path2, **kwargs):
        def put_images_randomly(back_img, put_img):            
            b_img = next(back_img).convert('RGBA')
            back_img_size_width, back_img_size_height = b_img.size
            segmentation = Image.new('RGBA', (back_img_size_width, back_img_size_height), (0, 0, 0, 0))
            png_back = Image.new('RGBA', (back_img_size_width, back_img_size_height), (255, 255, 255, 255))###

            #指定した座標数だけ画像をランダムに貼り付ける(座標は重複あり、画像は種類に重複なし)
            for n in range(0, kwargs['pos_num']): 
                put_pos_x = tf.random.uniform(shape = [], minval = 0, maxval = back_img_size_width, dtype = tf.int32).numpy()
                put_pos_y = tf.random.uniform(shape = [], minval = 0, maxval = back_img_size_height, dtype = tf.int32).numpy()
                rate = tf.random.uniform(
                    shape = [], 
                    minval = kwargs['put_img_resize_rate_min'], 
                    maxval = kwargs['put_img_resize_rate_max'],
                ).numpy()
                p_img = next(put_img).convert('RGBA')
                put_img_size_width, put_img_size_height = p_img.size
                resized_put_img = p_img.resize(
                    (
                        int(put_img_size_width * rate), int(put_img_size_height * rate)
                    )
                )
                #b_img.paste(resized_put_img, (put_pos_x, put_pos_y), resized_put_img)
                segmentation.paste(resized_put_img.convert('RGBA'), (put_pos_x, put_pos_y))
                png_back.paste(resized_put_img, (put_pos_x, put_pos_y))###
                #png_back = Image.alpha_composite(png_back, resized_put_img)

            #b_img = Image.alpha_composite(b_img, png_back)
            segmentation = segmentation.convert('L')
            segmentation = segmentation.point(lambda x: 0 if x == 0 else 255)
            segmentation = np.array(segmentation)
            segmentation = np.stack((segmentation, segmentation, segmentation), axis = 2)
            segmentation = Image.fromarray(segmentation.astype('uint8'), mode = 'RGB')
            segmentation.save('TEST.png')
            png_back.save('PASTE.png')

        b_img, p_img = None, None
        back_img, put_img = None, None

        while True:
            if b_img == None or p_img == None: 
                back_img = self.setData(path1)
                put_img = self.setData(path2)

            put_images_randomly(back_img, put_img)
            yield back_img


if __name__ == '__main__':
    #ToDo
    #入力画像とラベル画像を作成し、ジェネレータからnumpy形式で渡せるようにする

    myGen = createSegmentationDataset()
    generator = myGen.flow_from_directory(
        './dataset_for_p2p/building/building/*.jpg',
        './dataset_for_p2p/add/add/*.png',
        target_size = (128, 128),
        batch_size = 32,
        class_mode = None,
        seed = 1, #同じシードを指定
        put_img_resize_rate_min = 0.1, #貼り付け画像の最小縮小率
        put_img_resize_rate_max = 0.5,   #貼り付け画像の最大縮小率
        pos_num = 10 #貼り付け座標数
    )
    generator.__next__()
    #myGen.setData('./dataset_for_DCGAN/origin/origin/*.jpg', './dataset_for_DCGAN/origin/origin/*.jpg')