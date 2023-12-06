import os
import config
from tqdm import tqdm

import cv2
import numpy as np
from skimage.measure import regionprops

def is_alpha(mask_patch, tot_img_area):
    leish_area = regionprops(mask_patch)[0].area
    return (leish_area/tot_img_area) >= config.ALPHA

def crop(img, mask, img_id, WITH_LEISH_STRIDE, NO_LEISH_STRIDE):
    x, y = 0,0
    end_h, end_w = img.shape[0], img.shape[1]
    stride = config.NO_LEISH_STRIDE
    tot_img_area = config.WINDOW_SIZE * config.WINDOW_SIZE

    is_looping = True
    while((x + config.WINDOW_SIZE) <= end_h):
        while((y + config.WINDOW_SIZE) <= end_w):
            if (x + config.WINDOW_SIZE) > end_h or (y + config.WINDOW_SIZE) > end_w:
                is_looping = False
                break

            img_patch = img[x:x+config.WINDOW_SIZE, y:y+config.WINDOW_SIZE]
            mask_patch = mask[x:x+config.WINDOW_SIZE, y:y+config.WINDOW_SIZE]

            out_name = f'{img_id[0:-4]}-{x}-{y}.png'
            # mask_out_name = f'{img_id[0:-4]}-{x}-{y}.png'
            if np.any(mask_patch == 1):
                has_enough_leish = is_alpha(mask_patch, tot_img_area)
                stride = WITH_LEISH_STRIDE
                if has_enough_leish and img_patch.size > 0:
                    cv2.imwrite(os.path.join(config.LEISH_OUTPUT_PATH, out_name), img_patch)
                    # cv2.imwrite(LEISH_MASKS_OUTPUT_PATH+mask_out_name, mask_patch)
            else:
                stride = NO_LEISH_STRIDE
                if np.count_nonzero(img_patch == 255) < 0.5 * tot_img_area and img_patch.size > 0:
                    cv2.imwrite(os.path.join(config.NO_LEISH_OUTPUT_PATH, out_name), img_patch)

            y += stride

        x += stride
        y = 0
        if not is_looping:
            break

def dynamic_patcher(
        IMGS_FOLDER_PATH=config.ENHANCED_IMG_DIR,
        MASKS_FOLDER_PATH=config.MASKS_FOLDER_PATH,
        WITH_LEISH_STRIDE=config.WITH_LEISH_STRIDE,
        NO_LEISH_STRIDE=config.NO_LEISH_STRIDE,
    ):

    all_imgs = os.listdir(IMGS_FOLDER_PATH)
    all_masks = os.listdir(MASKS_FOLDER_PATH)

    imgs_n_masks = zip(all_imgs, all_masks)
    print('Total images = ', len(all_imgs), '\nTotal masks = ', len(all_masks))

    for img_id, mask_id in tqdm(imgs_n_masks, total=len(all_imgs)):
        img = cv2.imread(os.path.join(IMGS_FOLDER_PATH,img_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img / 255.0

        mask = cv2.imread(os.path.join(MASKS_FOLDER_PATH,mask_id), 0)
        mask = (mask > 0).astype(int)

        crop(img, mask, img_id, WITH_LEISH_STRIDE, NO_LEISH_STRIDE, SHAPE)