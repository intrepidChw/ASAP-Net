import os
import cv2
import glob

IMG_ROOT = r'G:\DataSet\semanticKITTI\visualization\SqueezeSegV2\TPC-2\sequences'
VIDEO_ROOT = r'G:\DataSet\semanticKITTI\visualization\videos'

begin = [311, 111, 0, 0, 81, 11, 61, 21, 0, 121]
end = [360, 160, 50, 50, 130, 60, 110, 70, 50, 150]

if __name__ == "__main__":
    # for i in range(10):
    i = 3
    img_dir = os.path.join(IMG_ROOT, '%02d' % i, 'imgs')
    img_array = []
    path_array = []
    fps = 10

    filename = os.path.join(img_dir, '%06d' % 0 + '.png')
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)

    for j in range(40, 100):
        filename = os.path.join(img_dir, '%06d' % j + '.png')
        img = cv2.imread(filename)
        # height, width, layers = img.shape
        # size = (width, height)
        img_array.append(img)
        print('Adding ' + filename)
        path_array.append(filename)

    video_dir = os.path.join(VIDEO_ROOT, '%02d' % i + '.avi')
    out = cv2.VideoWriter(video_dir, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for j in range(len(img_array)):
        print('Writing ' + path_array[j])
        out.write(img_array[j])
    out.release()

