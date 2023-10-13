import cv2 as cv
import numpy as np


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
    from matplotlib import pyplot as plt
    print('generate plot of histograms for "%s"' % plotname)
    plt.clf()
    fig = plt.figure()
    ax = fig.gca()
    canvas = fig.canvas
    colors = ("b", "g", "r")

    plt.title(plotname)
    for (hist, color) in zip(hist_list, colors):
        print(hist)
        print(color)
        plt.plot(hist, color=color)
        print('plotted')

    canvas.draw()  # Draw the canvas, cache the renderer

    image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    # # NOTE: reversed converts (W, H) from get_width_height to (H, W)

    image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)  # (H, W, 3)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    plt.clf()
    return image# plt.savefig('hist/' + plotname + 'histplot.png')

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
    conf_thresh = 0.2
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
  
    return out_img, detections
    
def classify_histogram(hist, clusters):
    distances = {c:np.inf for c in list(clusters.keys())}
    # clusters = {c:{'count':0, 'sum_hist':None} for c in classes}

    for c in list(clusters.keys()):
        cluster_avg_hist = clusters[c]['average_histogram']
        dist = ((hist - cluster_avg_hist) ** 2).sum()
        distances[c] = dist
    
    print(distances)
    # blue_sorted = np.argsortq
    ind = np.argmin([distances[k] for k in clusters.keys()])
    roast_level = list(clusters.keys())[ind]
    print('classed as %s ' % roast_level)
    return roast_level
