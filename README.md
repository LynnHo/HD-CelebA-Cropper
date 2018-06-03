# <p align="center"> HD Celeba Cropper

[Celeba](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset provides an aligned set `img_align_celeba.zip`. However, the size of each aligned image is **218x178**, so the faces cropped from such images would be even smaller!

Here we provide a code to obtain **higher resolution** face images, by cropping the faces from the original unaligned images via 5 landmarks.

## Cropped Faces (512x512)

***Notice***: There are still some low resolution cropped faces since the coppresonding original images are low resolution.

<p align="center">
<img src="./pics/1.jpg" width="49.5%"> <img src="./pics/2.jpg" width="49.5%">
<img src="./pics/3.jpg" width="49.5%"> <img src="./pics/4.jpg" width="49.5%">
</p>

## Usage

- Prerequisites
    - OpenCV 3 (much faster) or scikit-image
    - Python 2 or 3

- Dataset
    - download the dataset
        - **img_celeba.7z**
            - https://pan.baidu.com/s/1eSNpdRG#list/path=%2FCelebA%2FImg or
            - https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg
        - **list_landmarks_celeba.txt**
            - https://pan.baidu.com/s/1eSNpdRG#list/path=%2FCelebA%2FAnno&parentPath=%2F or
            - https://drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs
    - unzip the data and organize the files as follow

        ```
        path_to_dataset
        ├── data/*.jpg
        └── list_landmarks_celeba.txt
        ```

- Example

    - 512x512 + bicubic + jpg

        ```console
        python hd_celeba.py --data_dir path_to_dataset --crop_size 512 --order 3 --save_format jpg --n_worker 32
        ```

    - 512x512 + lanczos4 (with OpenCV) + png

        ```console
        python hd_celeba.py --data_dir path_to_dataset --crop_size 512 --order 4 --save_format png --n_worker 32
        ```

- Notice
    - order for OpenCV
        - 0: INTER_NEAREST
        - 1: INTER_LINEAR
        - 2: INTER_AREA
        - 3: INTER_CUBIC
        - 4: INTER_LANCZOS4
        - 5: INTER_LANCZOS4
    - order for scikit-image
        - 0: Nearest-neighbor
        - 1: Bi-linear
        - 2: Bi-quadratic
        - 3: Bi-cubic
        - 4: Bi-quartic
        - 5: Bi-quintic
