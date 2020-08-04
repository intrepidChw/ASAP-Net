import os
import cv2
import glob

IMG_ROOT = r'G:\MyProject\data\BMVC20\Synthia_results\ssg_atten_visu'
VIDEO_ROOT = r'G:\MyProject\data\BMVC20\Synthia_results\ssg_atten_visu\videos'


if __name__ == "__main__":
    # for i in range(10):
    seq_name = 'SYNTHIA-SEQS-06-SPRING'
    img_dir = os.path.join(IMG_ROOT, seq_name, 'predictions')
    img_array = []
    path_array = []
    fps = 10

    filename = os.path.join(img_dir, '%06d' % 0 + '.png')
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)

    for j in range(45, 105):
        filename = os.path.join(img_dir, '%06d' % j + '.png')
        img = cv2.imread(filename)
        
        img_array.append(img)
        print('Adding ' + filename)
        path_array.append(filename)

    video_dir = os.path.join(VIDEO_ROOT, '%02d' % i + '.avi')
    out = cv2.VideoWriter(video_dir, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for j in range(len(img_array)):
        print('Writing ' + path_array[j])
        out.write(img_array[j])
    out.release()

