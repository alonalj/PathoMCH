

import numpy as np
import openslide
import os
from PIL import Image
from conf import Conf
from utils import *


def check_img_is_blank(img):
    im = np.array(img)
    pct_bkg = np.mean((im > 220) * 1)
    if pct_bkg >= 0.5:
        return True, pct_bkg
    return False, pct_bkg


def deep_zoom_tile(slide_path, output_folder=Conf().IMG_PATH, xZoom=Conf().ZOOM_LEVEL, patch_size=512):
    '''
    Tiling the slide at slide_path at xZoom magnification and into patch_size patches.
    Adapted from the very elegant code by scotthoule (with several modifications):
    https://github.com/SBU-BMI/u24_lymphocyte/blob/master/patch_extraction/save_svs_to_tiles.py
    '''
    slide_name = slide_path.split('.svs')[0].split('/')[-1]
    blank_pct_list = []

    print('Tiling to {}'.format(output_folder))

    if os.path.exists('../res/blank_counts_zoom_{}_size_{}_{}.pkl'.format(xZoom, patch_size, slide_name)):
        print("Found blanks pkl for {}. Continuing.".format(slide_name))
        return

    try:
        oslide = openslide.OpenSlide(slide_path)
        if openslide.PROPERTY_NAME_MPP_X in oslide.properties:
            mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X])
        elif "XResolution" in oslide.properties:
            mag = 10.0 / float(oslide.properties["XResolution"])
        elif 'tiff.XResolution' in oslide.properties:   # for Multiplex IHC WSIs, .tiff images
            mag = 10.0 / float(oslide.properties["tiff.XResolution"])
        else:
            print('[WARNING] mpp value not found. Assuming it is 40X with mpp=0.254!', slide_path)
            mag = 10.0 / float(0.254)
        pw = int(patch_size * mag / xZoom)
        width = oslide.dimensions[0]
        height = oslide.dimensions[1]
    except:
        print('{}: exception caught'.format(slide_path))
        return

    print(slide_path, width, height)
    for x in range(1, width, pw):
        for y in range(1, height, pw):
            if x + pw > width:
                pw_x = width - x
            else:
                pw_x = pw
            if y + pw > height:
                pw_y = height - y
            else:
                pw_y = pw

            if (int(patch_size * pw_x / pw) <= 0) or \
               (int(patch_size * pw_y / pw) <= 0) or \
               (pw_x <= 0) or (pw_y <= 0):
                continue

            patch = oslide.read_region((x, y), 0, (pw_x, pw_y))
            patch = patch.resize((patch_size * pw_x // pw, patch_size * pw_y // pw), Image.ANTIALIAS)
            patch = patch.convert('RGB')
            is_blank, pct_blank = check_img_is_blank(patch)
            if is_blank:
                blank_pct_list.append(pct_blank)
            else:
                fname = '{}{}_{}_{}.jpeg'.format(output_folder, slide_name, x//pw, y//pw)
                patch.save(fname)
    save_obj(blank_pct_list, 'blank_counts_zoom_{}_size_{}_{}'.format(xZoom, patch_size, slide_name))


if __name__ == '__main__':
    slide_path = '/Users/levy.alona/Google Drive/phd/patho/unsupervised_object_detection/data/slides/diagnostic/CD4-T81 - 2020-01-17 12.06.38.ndpi'

    from conf import Conf
    c = Conf()
    out_folder = c.IMG_PATH
    deep_zoom_tile(slide_path, xZoom=5)