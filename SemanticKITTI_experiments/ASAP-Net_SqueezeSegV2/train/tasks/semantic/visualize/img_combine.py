import os
from PIL import Image


UNIT_HEGHT = 64
UNIT_WIDTH = 1024

FULL_HEIGHT = UNIT_HEGHT * 4
FULL_WIDTH = UNIT_WIDTH


LABEL_ROOT = "/mnt/sdd/hanwen/dataset/semanticKITTI/dataset/sequences"
ORIGIN_RROT = "/mnt/sdd/hanwen/dataset/semanticKITTI/results/valid/origin/sequences"
SAP_ROOT = "/mnt/sdd/hanwen/dataset/semanticKITTI/results/valid/sap/sequences"
ASAP_ROOT = "/mnt/sdd/hanwen/dataset/semanticKITTI/results/valid/asap/sequences"

SAVE_ROOT = "/mnt/sdd/hanwen/visualization"

seqs = [8]
if __name__ == "__main__":
    for i in seqs:
        label_dir = os.path.join(LABEL_ROOT, '%02d' % i, 'imgs')
        origin_dir = os.path.join(ORIGIN_RROT, '%02d' % i, 'imgs')
        sap_dir = os.path.join(SAP_ROOT, '%02d' % i, 'imgs')
        asap_dir = os.path.join(ASAP_ROOT, '%02d' % i, 'imgs')
        save_dir = os.path.join(SAVE_ROOT, '%02d' % i)
        os.makedirs(save_dir, exist_ok=True)

        img_names = os.listdir(label_dir)

        for img in img_names:
            label_img = os.path.join(label_dir, img)
            origin_img = os.path.join(origin_dir, img)
            fn3d_img = os.path.join(sap_dir, img)
            tu2_img = os.path.join(asap_dir, img)
            save_img = os.path.join(save_dir, img)

            target = Image.new('RGB', (FULL_WIDTH, FULL_HEIGHT))
            imgs = [label_img, origin_img, fn3d_img, tu2_img]

            left_y = 0
            right_y = UNIT_HEGHT
            for sub_img in imgs:
                target.paste(Image.open(sub_img), (0, left_y, FULL_WIDTH, right_y))
                left_y += UNIT_HEGHT
                right_y += UNIT_HEGHT
            print('saving img:', save_img)
            target.save(save_img)




