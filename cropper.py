from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


_DEFAULT_MEAN_LANDMARKS = np.array([[-0.46911814, -0.51348481],
                                    [0.45750203, -0.53173911],
                                    [-0.00499168, 0.06126145],
                                    [-0.40616926, 0.46826089],
                                    [0.42776873, 0.45444013]])


def align_crop_5pts_opencv(img,
                           src_landmarks,
                           mean_landmarks=_DEFAULT_MEAN_LANDMARKS,
                           crop_size=512,
                           face_factor=0.7,
                           landmark_factor=0.35,
                           align_type='similarity',
                           order=3,
                           mode='edge'):
    """Align and crop a face image by 5 landmarks.

    Arguments:
        img             : Face image to be aligned and cropped.
        src_landmarks   : 5 landmarks:
                              [[left_eye_x, left_eye_y],
                               [right_eye_x, right_eye_y],
                               [nose_x, nose_y],
                               [left_mouth_x, left_mouth_y],
                               [right_mouth_x, right_mouth_y]].
        mean_landmarks  : Mean shape, should be normalized in [-0.5, 0.5].
        crop_size       : Output image size.
        face_factor     : The factor of face area relative to the output image.
        landmark_factor : The factor of landmarks' area relative to the face.
        align_type      : 'similarity' or 'affine'.
        order           : The order of interpolation. The order has to be in the range 0-5:
                              - 0: INTER_NEAREST
                              - 1: INTER_LINEAR
                              - 2: INTER_AREA
                              - 3: INTER_CUBIC
                              - 4: INTER_LANCZOS4
                              - 5: INTER_LANCZOS4
        mode            : One of ['constant', 'edge', 'symmetric', 'reflect', 'wrap'].
                          Points outside the boundaries of the input are filled according
                          to the given mode.
    """
    # set OpenCV
    import cv2
    inter = {0: cv2.INTER_NEAREST, 1: cv2.INTER_LINEAR, 2: cv2.INTER_AREA,
             3: cv2.INTER_CUBIC, 4: cv2.INTER_LANCZOS4, 5: cv2.INTER_LANCZOS4}
    border = {'constant': cv2.BORDER_CONSTANT, 'edge': cv2.BORDER_REPLICATE,
              'symmetric': cv2.BORDER_REFLECT, 'reflect': cv2.BORDER_REFLECT101,
              'wrap': cv2.BORDER_WRAP}

    # check
    assert align_type in ['affine', 'similarity'], 'Invalid `align_type`! Allowed: %s!' % ['affine', 'similarity']
    assert order in [0, 1, 2, 3, 4, 5], 'Invalid `order`! Allowed: %s!' % [0, 1, 2, 3, 4, 5]
    assert mode in ['constant', 'edge', 'symmetric', 'reflect', 'wrap'], 'Invalid `mode`! Allowed: %s!' % ['constant', 'edge', 'symmetric', 'reflect', 'wrap']

    # move
    move = np.array([img.shape[1] // 2, img.shape[0] // 2])

    # pad border
    v_border = img.shape[0] - crop_size
    w_border = img.shape[1] - crop_size
    if v_border < 0:
        v_half = (-v_border + 1) // 2
        img = np.pad(img, ((v_half, v_half), (0, 0), (0, 0)), mode=mode)
        src_landmarks += np.array([0, v_half])
        move += np.array([0, v_half])
    if w_border < 0:
        w_half = (-w_border + 1) // 2
        img = np.pad(img, ((0, 0), (w_half, w_half), (0, 0)), mode=mode)
        src_landmarks += np.array([w_half, 0])
        move += np.array([w_half, 0])

    # estimate transform matrix
    mean_landmarks -= np.array([mean_landmarks[0, :] + mean_landmarks[1, :]]) / 2.0  # middle point of eyes as center
    trg_landmarks = mean_landmarks * (crop_size * face_factor * landmark_factor) + move
    if align_type == 'affine':
        tform = cv2.estimateAffine2D(trg_landmarks, src_landmarks, ransacReprojThreshold=np.Inf)[0]
    else:
        tform = cv2.estimateAffinePartial2D(trg_landmarks, src_landmarks, ransacReprojThreshold=np.Inf)[0]

    # fix the translation to match the middle point of eyes
    trg_mid = (trg_landmarks[0, :] + trg_landmarks[1, :]) / 2.0
    src_mid = (src_landmarks[0, :] + src_landmarks[1, :]) / 2.0
    new_trg_mid = cv2.transform(np.array([[trg_mid]]), tform)[0, 0]
    tform[:, 2] += src_mid - new_trg_mid

    # warp image by given transform
    output_shape = (crop_size // 2 + move[1] + 1, crop_size // 2 + move[0] + 1)
    img_align = cv2.warpAffine(img, tform, output_shape[::-1], flags=cv2.WARP_INVERSE_MAP + inter[order], borderMode=border[mode])

    # crop
    img_crop = img_align[-crop_size:, -crop_size:]

    return img_crop


def align_crop_5pts_skimage(img,
                            src_landmarks,
                            mean_landmarks=_DEFAULT_MEAN_LANDMARKS,
                            crop_size=512,
                            face_factor=0.7,
                            landmark_factor=0.35,
                            align_type='similarity',
                            order=3,
                            mode='edge'):
    """Align and crop a face image by 5 landmarks.

    Arguments:
        img             : Face image to be aligned and cropped.
        src_landmarks   : 5 landmarks:
                              [[left_eye_x, left_eye_y],
                               [right_eye_x, right_eye_y],
                               [nose_x, nose_y],
                               [left_mouth_x, left_mouth_y],
                               [right_mouth_x, right_mouth_y]].
        mean_landmarks  : Mean shape, should be normalized in [-0.5, 0.5].
        crop_size       : Output image size.
        face_factor     : The factor of face area relative to the output image.
        landmark_factor : The factor of landmarks' area relative to the face.
        align_type      : 'similarity' or 'affine'.
        order           : The order of interpolation. The order has to be in the range 0-5:
                              - 0: Nearest-neighbor
                              - 1: Bi-linear
                              - 2: Bi-quadratic
                              - 3: Bi-cubic
                              - 4: Bi-quartic
                              - 5: Bi-quintic
        mode            : One of ['constant', 'edge', 'symmetric', 'reflect', 'wrap'].
                          Points outside the boundaries of the input are filled according
                          to the given mode.
    """
    import skimage.transform as transform

    # check
    assert align_type in ['affine', 'similarity'], 'Invalid `align_type`! Allowed: %s!' % ['affine', 'similarity']
    assert order in [0, 1, 2, 3, 4, 5], 'Invalid `order`! Allowed: %s!' % [0, 1, 2, 3, 4, 5]
    assert mode in ['constant', 'edge', 'symmetric', 'reflect', 'wrap'], 'Invalid `mode`! Allowed: %s!' % ['constant', 'edge', 'symmetric', 'reflect', 'wrap']

    # move
    move = np.array([img.shape[1] // 2, img.shape[0] // 2])

    # pad border
    v_border = img.shape[0] - crop_size
    w_border = img.shape[1] - crop_size
    if v_border < 0:
        v_half = (-v_border + 1) // 2
        img = np.pad(img, ((v_half, v_half), (0, 0), (0, 0)), mode=mode)
        src_landmarks += np.array([0, v_half])
        move += np.array([0, v_half])
    if w_border < 0:
        w_half = (-w_border + 1) // 2
        img = np.pad(img, ((0, 0), (w_half, w_half), (0, 0)), mode=mode)
        src_landmarks += np.array([w_half, 0])
        move += np.array([w_half, 0])

    # estimate transform matrix
    mean_landmarks -= np.array([mean_landmarks[0, :] + mean_landmarks[1, :]]) / 2.0  # middle point of eyes as center
    trg_landmarks = mean_landmarks * (crop_size * face_factor * landmark_factor) + move
    tform = transform.estimate_transform(align_type, trg_landmarks, src_landmarks)

    # fix the translation to match the middle point of eyes
    trg_mid = (trg_landmarks[0, :] + trg_landmarks[1, :]) / 2.0
    src_mid = (src_landmarks[0, :] + src_landmarks[1, :]) / 2.0
    new_trg_mid = transform.matrix_transform([trg_mid], tform.params)[0]
    tform.params[:2, 2] += src_mid - new_trg_mid

    # warp image by given transform
    output_shape = (crop_size // 2 + move[1] + 1, crop_size // 2 + move[0] + 1)
    img_align = transform.warp(img, tform, output_shape=output_shape, order=order, mode=mode)

    # crop
    img_crop = img_align[-crop_size:, -crop_size:]

    return img_crop
