import onnxruntime as ort
import os, sys, time
import numpy as np
import cv2 as cv
import yaml, json
from matplotlib import pyplot as plt


modeldir = 'bean-recognition-v1'
modelfile = os.path.join(modeldir, 'model.onnx')
paramsfile = os.path.join(modeldir, 'param.yaml')
datasetfile = os.path.join(modeldir, 'dataset.yaml')
has_TIDL = False

runtime_options = {
    "tidl_platform": "J7",
    "tidl_version": "7.2",
    "tidl_tools_path": "null",
    "artifacts_folder": os.path.join(modeldir, 'artifacts'),
    "tensor_bits": 8,
    "import": 'no',
}

def get_hist(img, num_bins=16):
    chans = cv.split(img)

    num_bins = min(num_bins, 256) #highest value per pixel is 256 so we can't have more bins than that..
    num_pixels = int(img.shape[0] * img.shape[1])

    hists = []
    for chan in chans:
        hist = cv.calcHist([chan], [0], None, [num_bins], [0,256]) / num_pixels
        hists.append(hist)
        
    hists = np.asarray(hists)
    return hists


def gen_hist_plot(hist_list, plotname):
    print('generate plot of histograms for "%s"' % plotname)
    plt.clf()
    colors = ("b", "g", "r")

    plt.title(plotname)
    for i, color in enumerate(colors):
        hist = hist_list[i,:]
        print(color)
        print(hist)
        plt.plot(hist, color=color)
        print('plotted')

    # plt.show()
    plt.savefig('hist/' + plotname + 'histplot.png')

    # plt.clf() 

def scale_bounding_box(image_size:tuple, x1, y1, x2, y2, model_config):
    '''
    Scale dimensions of the bounding box to the size of the original image 
    so it can be labelled with the classes recognized
    '''
    h = image_size[0]
    w = image_size[1]
    crop_size = model_config['preprocess']['crop']
    scale_factor_x = w/crop_size[1]
    scale_factor_y = h/crop_size[0]
    
    # If padded, then we did so to the longest dimension, and thus the scaling factor is relative to that largest dimension (pad to large square, then resize to smaller)
    if model_config['preprocess'].get('resize_with_pad', None):
        if w > h: scale_factor_y = scale_factor_x
        else: scale_factor_x = scale_factor_y
    
    x1 = int(min(x1 * scale_factor_x, w))
    x2 = int(min(x2 * scale_factor_x, w))
    y1 = int(min(y1 * scale_factor_y, h))
    y2 = int(min(y2 * scale_factor_y, h))

    return x1, y1, x2, y2

def draw_output(in_img, output, classes, original_image=None, model_config=None):
    '''
    Draw bounding boxes and class names onto the image. 
    Ideally, we do this to the original image to avoid interpolating the image
    when scaling up. We need to know the preprocessing parameters to do so
    '''
    conf_thresh = 0.4
    if original_image is None:
        out_img = in_img.copy()
    else: 
        out_img = original_image.copy()

    # if has_TIDL:
    detections = []
    for out in output:
        int_out = out.astype('int')
        x1, y1, x2, y2, _, cls = int_out[:]
        conf = out[-2]
        if conf > conf_thresh:
            print(f'Predicted "{classes[int(cls)]}" with confidence %0.3f' % conf)
            detections.append(out)
            if model_config is not None: 
                x1, y1, x2, y2 = scale_bounding_box(out_img.shape, x1, y1, x2, y2, model_config)
                
            out_img = cv.rectangle(out_img, (x1,y1), (x2,y2), color=(0,255,255), thickness=4)
            classname = classes[int(cls)]
            print(classname)
            cv.putText(out_img, '%.3f, %s' % (conf, classes[int(cls)]), (x1 + 5, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # if len(detections) == 0: 
    #     conf = output[0][-2]
    #     print(f'Highest conf was "{classes[int(cls)]}" with confidence %0.3f' % conf)
    #     x1, y1, x2, y2, _, cls = int_out[:]
    #     conf = out[-2]
    #     if model_config is not None: 
    #         x1, y1, x2, y2 = scale_bounding_box(out_img.shape, x1, y1, x2, y2, model_config)
                
    #     out_img = cv.rectangle(out_img, (x1,y1), (x2,y2), color=(0,255,255), thickness=4)
  
    return out_img, detections
    
def classify_histogram(hist):
    pass

def test_hist():

    img = cv.imread('img.png')
    # img = np.zeros((320,320,3), dtype='uint8')
    # img[:,:,1] = 255
    cv.imshow('img', img)
    cv.waitKey(0)
    hist = get_hist(img)
    hist_plot = gen_hist_plot(hist, 'test')
    cv.imshow('hist', hist_plot)
    cv.waitKey(0)

if __name__ == '__main__':
    # test_hist()
    input_path = 'dataset/training'

    labels_info = json.load(open(os.path.join(input_path,'info.labels')))
    label_files = labels_info['files']

    # image_files = [os.path.join(input_path, imgfile) for imgfile in os.listdir(input_path) if ('png' in imgfile) or ('jpg' in imgfile)]
    classes = [cat['name'] for cat in yaml.safe_load(open(datasetfile, 'r'))['categories']]
    clusters = {c:{'count':0, 'sum_hist':None} for c in classes}
    params = yaml.safe_load(open(paramsfile, 'r'))
    for file in label_files:
        path = os.path.join(input_path, file['path'])
        image = cv.imread(path)
        print(file)
        boxes = file['boundingBoxes']
        for box in boxes:
            print(box)

            x,y,w,h,l = box['x'], box['y'], box['width'], box['height'], box['label']
            # Only focus on the inner 1/4th of the image to avoid pixels at the edge
            cropped_img = image[y+h//4:y+3*h//4, x+w//4:x+3*w//4]
            # cv.imshow('c', cropped_img)
            # cv.waitKey(0)
            hist = get_hist(cropped_img, num_bins=16) #hist is BGR ordered since that's how cv loads images
            # print(hist)

            clusters[l]['count'] += 1
            if clusters[l].get('sum_histogram', None ) is None:
                print('new class ' + l)
                clusters[l]['sum_histogram'] = hist
            else:
                print('additional instance of ' + l)
                clusters[l]['sum_histogram'] += hist
            # print(clusters)

    print('\n\n\n\n\n')
    for c in classes:
        print(clusters[c])
        clusters[c]['average_histogram'] = clusters[c]['sum_histogram'] / clusters[c]['count']
        gen_hist_plot(clusters[c]['average_histogram'] , c)
    import pickle
    pickle.dump(clusters, open('class_clusters.pickle', 'wb'))
