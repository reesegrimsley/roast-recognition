#  Copyright (C) 2023 Texas Instruments Incorporated - http://www.ti.com/
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#    Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
#    Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
#
#    Neither the name of Texas Instruments Incorporated nor the names of
#    its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
Intended output display:
+------------------------------+----------+
|                              |          |
|                              |          |
|                              |  stats   |
|                              |  &       |
|  frame w/ post process       |  app     |
|  & visualization             |  info    |
|                              |          |
|                              |          |
|                              |          |
+------------------------------+----------+
|                                         |
|        performance stats/load           |
|                                         |
+-----------------------------------------+
'''


import numpy as np
import cv2 as cv
import pickle
import utils, cv_classifier


import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp, GLib, GObject
Gst.init(None)

RECEIPT_FONT = cv.FONT_HERSHEY_SIMPLEX
HEADING_FONT = cv.FONT_HERSHEY_TRIPLEX


class DisplayDrawer():
    '''
    Class to the output display. This is written primarily for 1920 x 1080 display, but should also scale to other sizes

    Performance stats should take up 20% of the image at the bottom, but have hard limit of height between 50 and 250 pixels. See tiperfoverlay gst plugin for source of this.

    '''
    def __init__(self, display_width=1920, display_height=1080, image_scale=0.675, image_start_y_scale=0.125):
        self.display_width = display_width
        self.display_height = display_height
        self.image_scale = image_scale
        self.image_start_y_scale = image_start_y_scale

        # no odd values allowed!
        self.image_width = int(display_width * image_scale)
        if self.image_width %2 == 1: self.image_width += 1 
        self.image_height = int(display_height * image_scale)
        if self.image_height %2 == 1: self.image_height += 1 


        self.list_width = display_width - self.image_width
        self.list_height = self.image_height
        self.perf_width = display_width
        self.perf_height = display_height - self.image_height


    def set_gst_info(self, app_out, gst_caps): 
        '''
        Set output caps and hold onto a reference for the appsrc plugin that interfaces from here to the final output sink (by default, kmssink.. see gst_configs.py)
        '''
        self.gst_app_out = app_out
        self.gst_caps = gst_caps
        self.gst_app_out.set_caps(self.gst_caps)


    def push_to_display(self, image):
        '''
        Push an image to the display through the appsrc

        param image: and image whose dimensions and pixel format matches self.gst_caps
        '''

        buffer = Gst.Buffer.new_wrapped(image.tobytes())

        ret = self.gst_app_out.push_buffer(buffer)
      
    def make_frame_init(self):
        '''
        Make an initial frame to push immediately. This is intentionally blank
        '''
        return np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
    

    def make_frame(self, input_image, infer_output, categories, model_obj, clusters):
        '''
        Use the output information from the tidlinferer plugin (after reformatting to convenient shape)
            to write some useful information onto various portions of the screen
        
        input image: HxWxC numpy array
        infer_output: tensor/2d array of shape num_boxes x 6, where the 6 values are x1,y1,x2,y2,score, label
        categories: in same format as dataset.yaml, a mapping of class labels to class names (strings)
        model_obj: the ModelRunner object associated with the model being run with tidlinferer
        '''

        processed_image = self.draw_bounding_boxes(input_image, infer_output, categories, clusters, viz_thres=0.35)
        print(processed_image.shape)

        frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        #center in the frame since no list/additiona,ll info area
        start_y = int(self.display_height * self.image_start_y_scale)
        start_x = (self.display_width - self.image_width) // 2
        frame[start_y:self.image_height+start_y, start_x:self.image_width+start_x] = processed_image
        # frame[0:self.image_height, self.image_width:] = 255

        return frame
    
    def draw_bounding_boxes(self, image, boxes_tensor, categories, clusters, viz_thres=0.6):
        '''
        Draw bounding boxes with classnames onto the 
        
        Each box in the tensor expected to be x1,y1,x2,y2,score,class-label.

        '''
        for box in boxes_tensor:
            score = box[4]
            label = box[5]

            if score > viz_thres:
                class_name = categories[int(label)]['name']
                x1,y1,x2,y2 = [int(v) for v in box[:4]]
                cv.rectangle(image, (x1,y1), (x2,y2), color=(0, 255, 128), thickness=3)
                # cv.putText(image, "dl:"+class_name, (x1,y1), cv.FONT_HERSHEY_SIMPLEX, 0.75, color=(0, 255, 255), thickness=2)

                w = x2-x1
                h = y2-y1
                x1 += w//4
                x2 -= w//4
                y1 += h//4
                y2 -= h//4

                cropped_img =  (image[y1:y2,x1:x2] ** 1).astype(np.uint8)
                if any([s==0 for s in cropped_img.shape]): continue

                hist = cv_classifier.get_hist(cropped_img, num_bins=16)
                hist[[2,1,0],:] = hist[[0,1,2],:] #histograms were in BGR but image is in RGB.. swap 
                roast_level = cv_classifier.classify_histogram(hist, clusters)
                print(roast_level)
                # if roast_level.lower() != class_name.lower():
                cv.putText(image, roast_level, (x1-w//4+10,y2), cv.FONT_HERSHEY_SIMPLEX, 1., color=(255, 255, 255), thickness=5)
                cv.putText(image, roast_level, (x1-w//4+10,y2), cv.FONT_HERSHEY_SIMPLEX, 1., color=(128, 64, 64), thickness=3)



        return image

        
        
