import argparse
import numpy as np
import vispy.scene
from vispy.scene import visuals
import sys
import vispy.io as io
import yaml
from vispy.gloo.util import _screenshot
import time
import _thread
from PIL import Image
import os
from laserscan import LaserScan, SemLaserScan


parser = argparse.ArgumentParser("./visualize.py")
parser.add_argument(
    '--config', '-c',
    type=str,
    required=False,
    default="config/labels/semantic-kitti.yaml",
    help='Dataset config file. Defaults to %(default)s',
)
FLAGS, unparsed = parser.parse_known_args()

color_dict = None
scan = SemLaserScan(color_dict, project=False)


data_root = r'G:\MyProject\data\BMVC20\Synthia_results\ssg_atten_results'
save_root = r'G:\MyProject\data\BMVC20\Synthia_results\ssg_atten_results_test'
seqs= ['SYNTHIA-SEQS-01-DAWN', 'SYNTHIA-SEQS-01-FALL', 'SYNTHIA-SEQS-01-NIGHT', 'SYNTHIA-SEQS-01-SUMMER',
        'SYNTHIA-SEQS-01-WINTER', 'SYNTHIA-SEQS-01-WINTERNIGHT', 'SYNTHIA-SEQS-02-DAWN', 'SYNTHIA-SEQS-02-FALL',
        'SYNTHIA-SEQS-02-NIGHT', 'SYNTHIA-SEQS-02-RAINNIGHT', 'SYNTHIA-SEQS-02-SOFTRAIN', 'SYNTHIA-SEQS-02-SUMMER',
        'SYNTHIA-SEQS-02-WINTER', 'SYNTHIA-SEQS-04-DAWN', 'SYNTHIA-SEQS-04-FALL', 'SYNTHIA-SEQS-04-NIGHT',
        'SYNTHIA-SEQS-04-RAINNIGHT', 'SYNTHIA-SEQS-04-SOFTRAIN', 'SYNTHIA-SEQS-04-SUMMER', 'SYNTHIA-SEQS-04-WINTER',
        'SYNTHIA-SEQS-04-WINTERNIGHT', 'SYNTHIA-SEQS-05-FOG', 'SYNTHIA-SEQS-06-SPRING', 'SYNTHIA-SEQS-06-SUNSET']
# 0, 6, 13, 21, 22
# (3, 3), (11, 476), (22, 420)
seq_idx = 22
frame_idx = 420

# save the visualizations of groundtruth
for frame_idx in range(200):
    data_path = os.path.join(data_root, seqs[seq_idx] + '-' + '%06d'%frame_idx + '.npz')
    scan.open_scan(data_path)
    scan.colorize()

    scatter = visuals.Markers()
    scatter.set_data(scan.points,
                    face_color=scan.sem_label_color[..., ::-1],
                    edge_color=scan.sem_label_color[..., ::-1],
                    size=2, edge_width=2.0)
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=False, bgcolor='w', size=(1000, 1000))
    
    camera = vispy.scene.cameras.TurntableCamera(elevation=0, azimuth=-30, roll=0)
    view = canvas.central_widget.add_view()  

    view.add(scatter)
    view.camera = camera

    img_root = os.path.join(save_root, seqs[seq_idx], 'groundtruth')
    os.makedirs(img_root, exist_ok=True)
    img_path = os.path.join(img_root, '%06d' % frame_idx + '.png')
    img = canvas.render()
    print('Processing ' + img_path)
    io.write_png(img_path, img[..., :3])

# save the visualizations of predictions
for frame_idx in range(200):
    data_path = os.path.join(data_root, seqs[seq_idx] + '-' + '%06d'%frame_idx + '.npz')
    scan.open_scan(data_path)
    scan.colorize()

    scatter = visuals.Markers()
    scatter.set_data(scan.points,
                    face_color=scan.sem_pred_color[..., ::-1],
                    edge_color=scan.sem_pred_color[..., ::-1],
                    size=2, edge_width=2.0)
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=False, bgcolor='w', size=(1000, 1000))
    
    camera = vispy.scene.cameras.TurntableCamera(elevation=0, azimuth=-30, roll=0)
    view = canvas.central_widget.add_view()  

    view.add(scatter)
    view.camera = camera

    img_root = os.path.join(save_root, seqs[seq_idx], 'predictions')
    os.makedirs(img_root, exist_ok=True)
    img_path = os.path.join(img_root, '%06d' % frame_idx + '.png')
    img = canvas.render()
    print('Processing ' + img_path)
    io.write_png(img_path, img[..., :3])
