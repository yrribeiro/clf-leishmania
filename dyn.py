from PIL import Image
from pathlib import Path
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import cv2
from skimage.measure import regionprops

IMGS_FOLDER_PATH = './data-preproc/d1-imgRang/images/' # preprocessed images with PIL.Enhance

# Path.cwd().joinpath('./d2-patches/leish/').mkdir(parents=True, exist_ok=False)
LEISH_OUTPUT_PATH = './d1-tentativa2/leish/'

# Path.cwd().joinpath('./d2-patches/no-leish/').mkdir(parents=True, exist_ok=False)
NO_LEISH_OUTPUT_PATH = './d1-tentativa2/no-leish/'

MASKS_FOLDER_PATH = './data-preproc/d1-imgRang/masks/'
# LEISH_MASKS_OUTPUT_PATH = './d1-tentativa2/leish/masks/'
# SHAPE = (2571, 2726, 3) # D2 IMG SHAPE
SHAPE = (3264, 2448, 3) # D1 IMG SHAPE
WITH_LEISH_STRIDE = 12
NO_LEISH_STRIDE = 96
ALPHA = 0.20

def is_alpha(mask_patch, tot_img_area, ALPHA):
    leish_area = regionprops(mask_patch)[0].area
    return (leish_area/tot_img_area) >= ALPHA

def crop(img, mask, img_id, dataset, WITH_LEISH_STRIDE, NO_LEISH_STRIDE, SHAPE):
    x, y, i = 0,0,0
    end_h, end_w = SHAPE[0], SHAPE[1]
    WINDOW_SIZE = stride = 96
    tot_img_area = WINDOW_SIZE*WINDOW_SIZE

    is_looping = True
    while((x + WINDOW_SIZE) <= end_h):
        while((y + WINDOW_SIZE) <= end_w):
            if (x + WINDOW_SIZE) > end_h or (y + WINDOW_SIZE) > end_w:
                is_looping = False
                break

            img_patch = img[x:x+WINDOW_SIZE, y:y+WINDOW_SIZE]
            mask_patch = mask[x:x+WINDOW_SIZE, y:y+WINDOW_SIZE]

            label = 0
            out_name = f'{img_id[0:-4]}-{x}-{y}.png'
            # mask_out_name = f'{img_id[0:-4]}-{x}-{y}.png'
            if np.any(mask_patch == 1):
                has_enough_leish = is_alpha(mask_patch, tot_img_area, ALPHA)
                stride = WITH_LEISH_STRIDE
                if has_enough_leish:
                    label = 1
                    cv2.imwrite(LEISH_OUTPUT_PATH+out_name, img_patch)
                    # cv2.imwrite(LEISH_MASKS_OUTPUT_PATH+mask_out_name, mask_patch)
            else:
                stride = NO_LEISH_STRIDE
                if np.count_nonzero(img_patch == 255) < 0.5 * tot_img_area:
                    cv2.imwrite(NO_LEISH_OUTPUT_PATH+out_name, img_patch)

            x += stride
            y += stride
            i += 1

            # pd.concat(
            #     [
            #         dataset,
            #         pd.Series({
            #             'label': label,
            #             'patch_id': out_name,
            #             'patch_mask_id': mask_out_name,
            #             'original_image_id': img_id,
            #         })
            #     ],
            #     ignore_index=True
            # )
        if not is_looping:
            break

def dynamic_patcher(
        IMGS_FOLDER_PATH=IMGS_FOLDER_PATH,
        MASKS_FOLDER_PATH=MASKS_FOLDER_PATH,
        SHAPE=SHAPE,
        WITH_LEISH_STRIDE=WITH_LEISH_STRIDE,
        NO_LEISH_STRIDE=NO_LEISH_STRIDE,
        ALPHA=ALPHA):

    dataset = pd.DataFrame(columns=['label', 'patch_id', 'patch_mask_id', 'original_image_id'])
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

        crop(img, mask, img_id, dataset, WITH_LEISH_STRIDE, NO_LEISH_STRIDE, SHAPE)

    # dataset.to_csv('data.csv', index=False)

def generate_idv_masks(img_file, dataset, WINDOW_SIZE):
    if dataset == 1:
        images_folder = './data-preproc/d1-imgRang/images/'
        masks_folder = './data-preproc/d1-imgRang/masks/'
        out_mask_folder = './d1-tentativa2/leish/masks/'
        ext_mask = '-mask.png'
        ext = '.JPG'
    else:
        images_folder = './data-preproc/d2-pilEnhance/images/'
        masks_folder = './data-preproc/d2-pilEnhance/masks/'
        out_mask_folder = './d2-tentativa2/leish/masks/'
        ext_mask = '_label.jpg'
        ext = '.jpg'
        img_id, coord_x, coord_y = img_file.split('-')

    img_id += ext
    orig_img = cv2.imread(os.path.join(images_folder, img_id))

    img_num, *_ = img_id.split('_')
    mask_id = img_num + ext_mask
    mask = cv2.imread(os.path.join(masks_folder,mask_id))

    if orig_img is not None:
        x1, y1 = int(coord_x), int(coord_y)
        x2, y2 = x1+WINDOW_SIZE, y1+WINDOW_SIZE
        mask_idv = mask[x1:x2, y1:y2]
        out_name = os.path.join(out_mask_folder, f'{img_id}-mask-{coord_x}-{coord_y}.png')
        cv2.imwrite(out_name, mask_idv)

if __name__ == '__main__':
    dynamic_patcher()
    tot_leish, tot_no_leish = len(os.listdir(LEISH_OUTPUT_PATH))-1, len(os.listdir(NO_LEISH_OUTPUT_PATH))
    print(f'Total patches WITH LEISHMANIA generated = {tot_leish}\nTotal patches WITHOUT = {tot_no_leish}\nProportion = {tot_leish/tot_no_leish:.2f}')

    # teste unitario
    # img_id, mask_id = '1_image_enhanced.jpg', '1_label.jpg'
    # img = cv2.imread(os.path.join(IMGS_FOLDER_PATH,img_id))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # SHAPEy = img.shape
    # mask = cv2.imread(os.path.join(MASKS_FOLDER_PATH,mask_id), 0)
    # mask = (mask > 0).astype(int)
    # dataset = pd.DataFrame([])

    # crop(img, mask, img_id, dataset, WITH_LEISH_STRIDE, NO_LEISH_STRIDE, SHAPEy)

    # gerador de mascaras individuais
    # leish_patches = os.listdir('./d1-patches/leish/')
    # for filename in leish_patches:
    #     print(filename)
    #     generate_idv_masks(filename[0:-4], 1, 96)