from PIL import Image
import numpy as np

class split_and_reconstruct():
    def __init__(self, width_stride = 32, height_stride = 32, crop_size = 128):
        self.width_stride = width_stride
        self.height_stride = height_stride
        self.crop_size = crop_size


    def downsize_image_for_crop(self, img):
        # この方程式のNについて解いて、画像領域に過不足がないようにリサイズする(縦横で同じ考え方)
        # stride * N + crop_size = width or height
        # Nは整数に切り捨てする(画像をダウンサイズする)
        width, height = img.size
        N_width =  (width - self.crop_size) // self.width_stride
        N_height =  (height - self.crop_size) // self.height_stride
        New_width_size = self.width_stride * N_width + self.crop_size
        New_height_size = self.height_stride * N_height + self.crop_size
        image_resize = img.resize((New_width_size, New_height_size))
        return image_resize

    def save_pos_list(self, img):
        width, height = img.size
        pos_list = []
        # (height - crop_size) + height_stride
        # 切り取りサイズを考慮した最終値 + rangeの最大値の調整
        for h in range(0, (height - self.crop_size) + self.height_stride, self.height_stride):
            for w in range(0, (width - self.crop_size) + self.width_stride, self.width_stride):
                pos_list.append((w, h, w + self.crop_size, h + self.crop_size))
        return pos_list

    def get_image_batch(self, img, pos_list):
        img_batch_list = []
        for (w_t, h_t, w_b, h_b) in pos_list:
            cropped_img = img.crop((w_t, h_t, w_b, h_b))
            img_batch_list.append(np.asarray(cropped_img))
        return np.array(img_batch_list)

    def image_construct(self, image_batch, pos_list, out_width, out_height, mode):
        canvas = Image.new(mode, (out_width, out_height))
        if mode == 'L': 
            for b_img, (w_t, h_t, w_b, h_b) in zip(image_batch, pos_list):
                canvas.paste(Image.fromarray(np.reshape(b_img.astype('uint8'), (self.crop_size, self.crop_size, 1))), (w_t, h_t))
        else:
            for b_img, (w_t, h_t, w_b, h_b) in zip(image_batch, pos_list):
                canvas.paste(
                    Image.fromarray(b_img.astype('uint8'), 'RGB'), (w_t, h_t)
                )
        return canvas

if __name__ == '__main__':
    pil_image = Image.open('./test_imgs/test1.jpg')#.convert('RGB')
    out_width, out_height = pil_image.size
    SC = split_and_reconstruct()

    image_resize = SC.downsize_image_for_crop(pil_image)
    pos_list = SC.save_pos_list(image_resize)
    image_batch = SC.get_image_batch(image_resize, pos_list)
    # Pass to AI --> batch image
    canvas = SC.image_construct(image_batch, pos_list, out_width, out_height, mode = 'RGB')
    canvas = canvas.resize(pil_image.size)
    canvas.save('test2.jpg')
