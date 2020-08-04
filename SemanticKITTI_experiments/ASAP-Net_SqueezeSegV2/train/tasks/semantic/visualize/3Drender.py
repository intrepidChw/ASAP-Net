import argparse
import numpy as np
import vispy.scene
from vispy.scene import visuals
import sys
import vispy.io as io
import yaml
from vispy.gloo.util import _screenshot
import __init__ as booger
import time
# import win32gui, win32ui, win32con, win32api
import _thread
from scipy import misc
from PIL import Image
import os
from common.laserscan import LaserScan, SemLaserScan


parser = argparse.ArgumentParser("./visualize.py")
parser.add_argument(
    '--config', '-c',
    type=str,
    required=False,
    default="config/labels/semantic-kitti.yaml",
    help='Dataset config file. Defaults to %(default)s',
)
FLAGS, unparsed = parser.parse_known_args()


CFG = yaml.safe_load(open(FLAGS.config, 'r'))
color_dict = CFG["color_map"]
scan = SemLaserScan(color_dict, project=False)


seqs = ['01', '03', '07', '09', '08']

scan_root = r'G:\DataSet\semanticKITTI\dataset\sequences'
gt_label_root = r'G:\DataSet\semanticKITTI\dataset\sequences'
gt_img_root = r'G:\DataSet\semanticKITTI\visualization\groundtruth\sequences'

segv2_label_root = r'G:\DataSet\semanticKITTI\prediction\SqueezeSegV2\Static\sequences'
segv2_img_root = r'G:\DataSet\semanticKITTI\visualization\SqueezeSegV2\Static\sequences'

segv2_sap1_label_root = r'G:\DataSet\semanticKITTI\prediction\SqueezeSegV2\SAP-1\sequences'
segv2_sap1_img_root = r'G:\DataSet\semanticKITTI\visualization\SqueezeSegV2\SAP-1\sequences'

segv2_asap1_label_root = r'G:\DataSet\semanticKITTI\prediction\SqueezeSegV2\ASAP-1\sequences'
segv2_asap1_img_root = r'G:\DataSet\semanticKITTI\visualization\SqueezeSegV2\ASAP-1\sequences'

segv2_asap2_label_root = r'G:\DataSet\semanticKITTI\prediction\SqueezeSegV2\ASAP-2\sequences'
segv2_asap2_img_root = r'G:\DataSet\semanticKITTI\visualization\SqueezeSegV2\ASAP-2\sequences'

label_roots = [gt_label_root, segv2_label_root, segv2_sap1_label_root,  segv2_asap1_label_root, segv2_asap2_label_root]
img_roots = [gt_img_root, segv2_img_root, segv2_sap1_img_root, segv2_asap1_img_root, segv2_asap2_img_root]


for i in range(len(label_roots)):
    label_root = label_roots[i]
    img_root = img_roots[i]

    for seq in seqs:
        seq_scan_path = os.path.join(scan_root, seq)
        seq_label_path = os.path.join(label_root, seq)
        seq_img_path = os.path.join(img_root, seq)
        frame_num = len(os.listdir(os.path.join(seq_scan_path, 'velodyne')))
        # print(frame_num)
        for i in range (frame_num):
            if i % 10 ==0:
                continue            
            
            frame_idx = i

            scan_path = os.path.join(seq_scan_path, 'velodyne', '%06d.bin' % frame_idx)
            if seq_label_path == seq_scan_path:
                label_path = os.path.join(seq_label_path, 'labels', '%06d.label' % frame_idx)    
            else:
                label_path = os.path.join(seq_label_path, 'predictions', '%06d.label' % frame_idx)
            
            scan.open_scan(scan_path)
            scan.open_label(label_path)
            scan.colorize()

            # points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)

            # create scatter object and fill in the data
            scatter = visuals.Markers()
            scatter.set_data(scan.points,
                            face_color=scan.sem_label_color[..., ::-1],
                            edge_color=scan.sem_label_color[..., ::-1],
                            size=2, edge_width=2.0)
            canvas = vispy.scene.SceneCanvas(keys='interactive', show=False, bgcolor='w', size=(1980, 1000))
            view = canvas.central_widget.add_view()  
            
            view.add(scatter)
            view.camera = 'turntable'  # 'turntable' or 'arcball'
            view.camera.fov = 60
            # view.camera.rotation = (90, 0, 0)
            # view.camera.rotation1 = (90, 0, 0)
            # view.camera.size_factor=0.5
            view.camera.distance = 12
            x_range = (-25, 25)
            y_range = (-25, 25)
            z_range = (-5, 5)
            view.camera.set_range(x_range, y_range, z_range)

            # add a colored 3D axis for orientation
            # axis = visuals.XYZAxis(parent=view.scene)

            img_dir = os.path.join(seq_img_path, 'imgs')
            os.makedirs(img_dir, exist_ok=True)
            img_path = os.path.join(seq_img_path, 'imgs', '%06d.png' % frame_idx)
            img = canvas.render()
            # img[..., -1] = 100
            print('Processing ' + img_path)
            io.write_png(img_path, img[..., :3])

