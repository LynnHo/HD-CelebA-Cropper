from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import partial
from multiprocessing import Pool
import os

import cropper
import numpy as np


def get_mean_landmark(landmarks):
    left_eye = landmarks[:, 0, :]
    right_eye = landmarks[:, 1, :]
    nose = landmarks[:, 2, :]
    left_mouth = landmarks[:, 3, :]
    right_mouth = landmarks[:, 4, :]

    left = (left_eye + nose + left_mouth) / 3.0
    right = (right_eye + nose + right_mouth) / 3.0
    top = (left_eye + nose + right_eye) / 3.0
    bottom = (left_mouth + nose + right_mouth) / 3.0
    top_mid = (top + left + right) / 3.0
    bottom_mid = (bottom + left + right) / 3.0
    mid = (top_mid + bottom_mid) / 2.0
    v_size = np.linalg.norm((left_eye + right_eye) / 2.0 - (left_mouth + right_mouth) / 2.0, axis=1)

    mid.shape = -1, 1, 2
    v_size.shape = -1, 1, 1
    norm_lm = (landmarks - mid) / v_size
    mean_lm = np.mean(norm_lm, axis=0)
    mean_lm = mean_lm / max(np.max(mean_lm[:, 0]) - np.min(mean_lm[:, 0]), np.max(mean_lm[:, 1]) - np.min(mean_lm[:, 1]))

    return mean_lm

# ==============================================================================
# =                                      param                                 =
# ==============================================================================
parser = argparse.ArgumentParser()
# main
parser.add_argument('--data_dir', dest='data_dir', required=True)
parser.add_argument('--crop_size', dest='crop_size', type=int, default=512)
parser.add_argument('--save_format', dest='save_format', default='jpg', choices=['jpg', 'png'])
parser.add_argument('--n_worker', dest='n_worker', type=int, default=8)
# others
parser.add_argument('--face_factor', dest='face_factor', type=float, default=0.65, help='The factor of face area relative to the output image.')
parser.add_argument('--landmark_factor', dest='landmark_factor', type=float, default=0.35, help="The factor of landmarks' area relative to the face.")
parser.add_argument('--align_type', dest='align_type', default='similarity', choices=['affine', 'similarity'])
parser.add_argument('--order', dest='order', type=int, default=3, choices=[0, 1, 2, 3, 4, 5], help='The order of interpolation.')
parser.add_argument('--mode', dest='mode', default='edge', choices=['constant', 'edge', 'symmetric', 'reflect', 'wrap'])
parser.add_argument('--compute_mean_landmark', dest='compute_mean_landmark', action='store_true')
args = parser.parse_args()


# ==============================================================================
# =                                opencv first                                =
# ==============================================================================
_DEAFAULT_JPG_QUALITY = 95
try:
    import cv2
    imread = cv2.imread
    imwrite = partial(cv2.imwrite, params=[int(cv2.IMWRITE_JPEG_QUALITY), _DEAFAULT_JPG_QUALITY])
    align_crop = cropper.align_crop_5pts_opencv
    print('Use OpenCV')
except:
    import skimage.io as io
    imread = io.imread
    imwrite = partial(io.imsave, quality=_DEAFAULT_JPG_QUALITY)
    align_crop = cropper.align_crop_5pts_skimage
    print('Importing OpenCv fails. Use scikit-image')


# ==============================================================================
# =                                     run                                    =
# ==============================================================================
save_dir = os.path.join(args.data_dir, 'data_crop_%s_%s' % (args.crop_size, args.save_format))
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

img_dir = os.path.join(args.data_dir, 'data')
landmark_file = os.path.join(args.data_dir, 'list_landmarks_celeba.txt')

img_names = np.loadtxt(landmark_file, skiprows=2, usecols=0, dtype=str)
landmarks = np.loadtxt(landmark_file, skiprows=2, usecols=range(1, 11))
landmarks.shape = -1, 5, 2
if args.compute_mean_landmark:
    mean_lm = get_mean_landmark(landmarks)
else:
    mean_lm = cropper._DEFAULT_MEAN_LANDMARKS


def work(i):  # a single work
    img = imread(os.path.join(img_dir, img_names[i]))
    img_crop = align_crop(img,
                          landmarks[i],
                          mean_lm,
                          crop_size=args.crop_size,
                          face_factor=args.face_factor,
                          landmark_factor=args.landmark_factor,
                          align_type=args.align_type,
                          order=args.order,
                          mode=args.mode)
    imwrite(os.path.join(save_dir, img_names[i].replace('jpg', args.save_format)), img_crop)

pool = Pool(args.n_worker)
pool.map(work, range(len(img_names)))
pool.close()
pool.join()
