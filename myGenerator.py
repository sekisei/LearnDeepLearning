import tensorflow as tf
from PIL import Image 

class createSegmentationDataset(tf.keras.preprocessing.image.ImageDataGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setData(self, directory):
        img_name = []
        img_path = tf.data.Dataset.list_files(directory, seed = 1, shuffle = True)
        for path in img_path: img_name.append(path.numpy().decode())
        for name in img_name: yield Image.open(name)

    def flow_from_directory(self, path1, path2, **kwargs):
        def my_alpha_composite(b_img, resized_put_img, pos):
            back = b_img
            put_pos_x, put_pos_y = pos
            w, h = resized_put_img.size
            cropped = back.crop((put_pos_x, put_pos_y, put_pos_x + w, put_pos_y + h))
            cropped = Image.alpha_composite(cropped, resized_put_img)
            back.paste(cropped, (put_pos_x, put_pos_y))
            return back

        def put_images_randomly(back_img, put_img):            
            b_img = next(back_img).convert('RGBA')
            segmentation = Image.new('RGBA', b_img.size)
            back_img_size_width, back_img_size_height = b_img.size

            #指定した座標数だけ画像をランダムに貼り付ける(座標は重複あり、画像は種類に重複なし)
            for n in range(0, kwargs['pos_num']): 
                p_img = next(put_img).convert('RGBA')
                put_img_size_width, put_img_size_height = p_img.size
                put_pos_x = tf.random.uniform(shape = [], minval = 0, maxval = back_img_size_width, dtype = tf.int32).numpy()
                put_pos_y = tf.random.uniform(shape = [], minval = 0, maxval = back_img_size_height, dtype = tf.int32).numpy()
                rate = tf.random.uniform(
                    shape = [], 
                    minval = kwargs['put_img_resize_rate_min'], 
                    maxval = kwargs['put_img_resize_rate_max'],
                ).numpy()
                resized_put_img = p_img.resize(
                    (
                        int(put_img_size_width * rate), int(put_img_size_height * rate)
                    )
                )
                R, G, B, A = resized_put_img.split()
                A = Image.merge('RGBA', (A, A, A, A))
                b_img = my_alpha_composite(b_img, resized_put_img, (put_pos_x, put_pos_y))
                segmentation = my_alpha_composite(segmentation, A, (put_pos_x, put_pos_y))
                
            return  segmentation.convert('RGB'), b_img.convert('RGB')

        b_img, p_img = None, None
        back_img, put_img = None, None

        while True:
            if b_img == None or p_img == None: 
                back_img = self.setData(path1)
                put_img = self.setData(path2)

            yield put_images_randomly(back_img, put_img)


if __name__ == '__main__':
    #ToDo
    #入力画像とラベル画像を作成し、ジェネレータからnumpy形式で渡せるようにする

    myGen = createSegmentationDataset()
    generator = myGen.flow_from_directory(
        '../datasets/dataset_for_flow/dataset_for_p2p/building/building/*.jpg',
        '../datasets/dataset_for_flow/dataset_for_p2p/add/add/*.png',
        target_size = (128, 128),
        batch_size = 32,
        class_mode = None,
        seed = 1, #同じシードを指定
        put_img_resize_rate_min = 0.1, #貼り付け画像の最小縮小率
        put_img_resize_rate_max = 0.5,   #貼り付け画像の最大縮小率
        pos_num = 10 #貼り付け座標数
    )
    segmentation, input_img = generator.__next__()
    segmentation.convert('RGB').save('../res/back.jpg')
    input_img.convert('RGB').save('../res/res.jpg')
