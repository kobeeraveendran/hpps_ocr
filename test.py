from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, os.path
import re
import sys
import tarfile
import copy
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import textwrap
import numpy as np
from six.moves import urllib
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont


from text_recognition import TextRecognition
from text_detection import TextDetection

from util import *
from shapely.geometry import Polygon, MultiPoint
from shapely.geometry.polygon import orient
from skimage import draw

# !flask/bin/python
'''
from flask import Flask, jsonify, flash, Response
from flask import make_response
from flask import request, render_template
from flask_bootstrap import Bootstrap
from flask import redirect, url_for
from flask import send_from_directory
'''

from werkzeug import secure_filename
from subprocess import call
# from sightengine.client import SightengineClient

'''
UPLOAD_FOLDER = '/ocr/ocr/uploads'
IMAGE_FOLDER = '/ocr/ocr/image'
VIDEO_FOLDER = r'/ocr/ocr/video'
FOND_PATH = '/ocr/ocr/STXINWEI.TTF'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4','avi'])
VIDEO_EXTENSIONS = set(['mp4', 'avi'])
app = Flask(__name__)
bootstrap = Bootstrap(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = os.urandom(24)
'''

FOND_PATH = './STXINWEI.TTF'

# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def init_ocr_model():
    detection_pb = './checkpoint/ICDAR_0.7.pb' # './checkpoint/ICDAR_0.7.pb'
    # recognition_checkpoint='/data/zhangjinjin/icdar2019/LSVT/full/recognition/checkpoint_3x_single_gpu/OCR-443861'
    # recognition_pb = './checkpoint/text_recognition_5435.pb' # 
    recognition_pb = './checkpoint/text_recognition.pb'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "9"
    with tf.device('/cpu:0'):
    #with tf.device('/gpu:0'):
        tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),#, visible_device_list="9"),
                                   allow_soft_placement=True, log_device_placement = True)

        #tf_config = tf.ConfigProto(device_count = {'GPU': 0}, allow_soft_placement=True, log_device_placement = True)

        detection_model = TextDetection(detection_pb, tf_config, max_size=1600)
        recognition_model = TextRecognition(recognition_pb, seq_len=27, config=tf_config)
    
    label_dict = np.load('./reverse_label_dict_with_rects.npy', allow_pickle = True)[()] # reverse_label_dict_with_rects.npy  reverse_label_dict
    return detection_model, recognition_model, label_dict 

#@app.route('/image_ocr/<filename>')
def predict_ocr_image(img_dir, filename, ocr_detection_model, ocr_recognition_model, ocr_label_dict):

    img_path = os.path.join(img_dir, filename)
    save_path = os.path.join('output', filename)

    image, output = detection(img_path, ocr_detection_model, ocr_recognition_model, ocr_label_dict)
    cv2.imwrite(save_path, image)
    #return send_from_directory(app.config['IMAGE_FOLDER'], filename)

    return output

#@app.route("/classify", methods=["POST"])
def classify():
    predictions = detection(request.data)
    print(predictions)
    #return jsonify(predictions=predictions)
    return predictions



#@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


from functools import reduce
import operator
import math


def order_points(pts):
    def centeroidpython(pts):
        x, y = zip(*pts)
        l = len(x)
        return sum(x) / l, sum(y) / l

    centroid_x, centroid_y = centeroidpython(pts)
    pts_sorted = sorted(pts, key=lambda x: math.atan2((x[1] - centroid_y), (x[0] - centroid_x)))
    return pts_sorted


def draw_annotation(image, points, label, horizon=True, vis_color=(30,255,255)):#(30,255,255)
    points = np.asarray(points)
    points = np.reshape(points, [-1, 2])
    cv2.polylines(image, np.int32([points]), 1, (0, 255, 0), 2)

    image = Image.fromarray(image)
    width, height = image.size
    fond_size = int(max(height, width)*0.03)
    FONT = ImageFont.truetype(FOND_PATH, fond_size, encoding='utf-8')
    DRAW = ImageDraw.Draw(image)

    points = order_points(points)
    if horizon:
        DRAW.text((points[0][0], max(points[0][1] - fond_size, 0)), label, vis_color, font=FONT)
    else:
        lines = textwrap.wrap(label, width=1)
        y_text = points[0][1]
        for line in lines:
            width, height = FONT.getsize(line)
            DRAW.text((max(points[0][0] - fond_size, 0), y_text), line, vis_color, font=FONT)
            y_text += height
    image = np.array(image)
    return image


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def mask_with_points(points, h, w):
    vertex_row_coords = [point[1] for point in points]  # y
    vertex_col_coords = [point[0] for point in points]

    mask = poly2mask(vertex_row_coords, vertex_col_coords, (h, w))  # y, x
    mask = np.float32(mask)
    mask = np.expand_dims(mask, axis=-1)
    bbox = [np.amin(vertex_row_coords), np.amin(vertex_col_coords), np.amax(vertex_row_coords),
            np.amax(vertex_col_coords)]
    bbox = list(map(int, bbox))
    return mask, bbox


def detection(img_path, detection_model, recognition_model, label_dict, it_is_video=False):
    if it_is_video:
        bgr_image = img_path
    else:
        bgr_image = cv2.imread(img_path)
    #print('\n\n\n\n\n\n', bgr_image.shape)
    #print('bgr_image path: ', img_path)
    #print('\n\n\n\n\n\n')
    vis_image = copy.deepcopy(bgr_image)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    r_boxes, polygons, scores = detection_model.predict(bgr_image)

    words = []
    confidences = []

    for r_box, polygon, score in zip(r_boxes, polygons, scores):
        mask, bbox = mask_with_points(polygon, vis_image.shape[0], vis_image.shape[1])
        masked_image = rgb_image * mask
        masked_image = np.uint8(masked_image)
        cropped_image = masked_image[max(0, bbox[0]):min(bbox[2], masked_image.shape[0]),
                        max(0, bbox[1]):min(bbox[3], masked_image.shape[1]), :]

        height, width = cropped_image.shape[:2]
        test_size = 299
        if height >= width:
            scale = test_size / height
            resized_image = cv2.resize(cropped_image, (0, 0), fx=scale, fy=scale)
            #print(resized_image.shape)
            left_bordersize = (test_size - resized_image.shape[1]) // 2
            right_bordersize = test_size - resized_image.shape[1] - left_bordersize
            image_padded = cv2.copyMakeBorder(resized_image, top=0, bottom=0, left=left_bordersize,
                                              right=right_bordersize, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            image_padded = np.float32(image_padded) / 255.
        else:
            scale = test_size / width
            resized_image = cv2.resize(cropped_image, (0, 0), fx=scale, fy=scale)
            #print(resized_image.shape)
            top_bordersize = (test_size - resized_image.shape[0]) // 2
            bottom_bordersize = test_size - resized_image.shape[0] - top_bordersize
            image_padded = cv2.copyMakeBorder(resized_image, top=top_bordersize, bottom=bottom_bordersize, left=0,
                                              right=0, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            image_padded = np.float32(image_padded) / 255.

        image_padded = np.expand_dims(image_padded, 0)
        #print(image_padded.shape)

        results, probs = recognition_model.predict(image_padded, label_dict, EOS='EOS')
        #print(results)
        #print(''.join(results))
        #print(probs)

        words.append(''.join(results))
        confidences.append(sum(probs) / len(probs))

        ccw_polygon = orient(Polygon(polygon.tolist()).simplify(5, preserve_topology=True), sign=1.0)
        pts = list(ccw_polygon.exterior.coords)[:-1]
        vis_image = draw_annotation(vis_image, pts, ''.join(results))
        # if height >= width:
        #     vis_image = draw_annotation(vis_image, pts, ''.join(results), False)
        # else:
        #     vis_image = draw_annotation(vis_image, pts, ''.join(results))
    print(' '.join(words))
    retval = (' '.join(words), ' '.join([str(c) for c in confidences]), img_path)
    #print(retval)

    return vis_image, retval

import argparse

from sqlite3 import connect

def update(db, pipeline_output):

    # connect to db
    conn = connect(db)
    c = conn.cursor()

    # update cropped images to include OCR output
    #print(pipeline_output)
    
    for output in pipeline_output:

        #c.execute("SELECT * FROM crop_path WHERE crop_path = {}".format(output[2]))
        c.execute(f'Update PipelineResults set ocr_results = "{output[0]}", confidence = "{output[1]}" where crop_path = "{output[2]}"')
        #c.execute('Update PipelineResults set ocr_results = (?), confidence = (?) where crop_path = (?)'.format(), output)
        conn.commit()
    
    #output = pipeline_output[0]
    #print('INPUT IMAGE PATH: ', output[2])
    #c.execute(f'Update PipelineResults set ocr_results = "{output[0]}", confidence = "{output[1]}" where crop_path = "{output[2]}"')

    #c.executemany('UPDATE PipelineResults SET ocr_results = (?), confidence = (?) WHERE crop_path = ?', pipeline_output)
    #conn.commit()

    # close out db
    c.close()


if __name__ == '__main__':
    # os.environ["TF_ENABLE_CONTROL_FLOW_V2"] = "0"
    #app.run(host='0.0.0.0', debug=True)

    parser = argparse.ArgumentParser()

    parser.add_argument('--img_dir', type = str, help = 'Path to directory of images.', default = './image/')
    parser.add_argument('--db', type = str, help = 'Path to SQLite database file.')
    parser.add_argument('--recognition_model_path', type = str, help = 'Path to the trained model (.pb) file.', default = './checkpoint/text_recognition.pb')
    parser.add_argument('--detection_model_path', type = str, help = 'Path to the trained detection model (.pb) file.', default = './checkpoint/ICDAR_0.7.pb')

    args = parser.parse_args()


    ocr_detection_model, ocr_recognition_model, ocr_label_dict = init_ocr_model()

    os.makedirs('output', exist_ok = True)

    db_output = []

    for filename in os.listdir(args.img_dir):
        db_output.append(predict_ocr_image(args.img_dir, filename, ocr_detection_model, ocr_recognition_model, ocr_label_dict))

    #update(args.db, db_output)