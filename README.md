# Roast Recognition demo

This repo contains code for a demostration of coffee roast recognition on the Texas Instruments AM62A. In the spirit of machine vision and inspection systems, this recognizes the roast level of coffee using machine learning (powered by Edge Impulse) and computer vision

The [deep learning model](./bean-recognition-v1) is trained in Edge Impulse using a custom dataset of roasted coffee beans. These include raw / green beans, underroasted (tan colored) beans, a light roast, a dark roast, and a burnt batch. The deep learning model (based on YOLOX) recognizes where in the image coffee beans are, and a simple histogram-based computer vision algorithm tries to match the coloration to the roast levels

The [AM62A](https://www.ti.com/tool/SK-AM62-LP) contains a 2 TOPS deep learning accelerator and 4x Arm(R) Cortex(R) A53 CPUs that are used here to run the application in real time. All code used to develop this project is contained in the repository.

Note that this demo is that not officially supported on by TI

## Running the demo

Setup the AM62A according to its [quick start](https://dev.ti.com/tirex/content/tirex-product-tree/am62ax-devtools/docs/am62ax_skevm_quick_start_guide.html) and the [Edge AI SDK for Linux](https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-AM62A/08.06.00.45).
* Note that I am using the 8.6 SDK. This was the version that Edge Impulse compiled models for at the time of this writing. The compiled deep learning model must be compiled for the same SDK version being used. Onnx or TFlite Models can be recompiled using [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools)

Log into a terminal session on the device via SSH or UART, and clone this repo using 'git'. All dependencies should be preinstalled with the SDK.

Plug in a USB camera capable of 1080p data capture and a 1080p monitor. For development, I used a Logitech c920 camera

From the gst-app-roast-rec directiory, run the "run_demo.sh" script to boot and run the demo. 


## How it's made

Recognizing roast levels turned out to be more difficult than initially imagined. The reason is transfer learning - training deep learning models on small datasets requires pretrained weights as a starting point, which we call transfer learning. Those weights are pretrained on a large dataset like COCO or imagenet1k -- in these datasets, the patterns to recognize are strongly related to shapes (an elephant, a person, a bowl, etc.) and less so to colors or shades. For roast recognition, we're looking for coloration, so solving the problem entirely with transfer learning is a problem. 

Here's where conventional computer vision comes in. Determining colors isn't terribly challenging for RGB images, as long as you know where to look. The object detection model (i.e. the deep learning neural network), recognizes regions of interest (coffee beans), and then we can use some straightforward algorithms to compare the coloration to what the original dataset includes. 

That computer vision algorithm can vary from very simple to very complex, depending on how many corner cases should be considered -- I opted for simple, assuming the environment could be made consistent. Here, I take a histogram of pixel color values for each channel (Red, green, blue) for the region containing coffee beans. Across the training dataset, I calculate what the 'average' histogram looks like, and save that (see [./gst-app-roast-rec/class_clusters-innerbox.pickle](./gst-app-roast-rec/class_clusters-innerbox.pickle). Then on new images, calculate a histogram, and see which 'average' histogram is most similar. My similarity metric was a Euclidean distance. 

From here, it was a matter of constructing an application around the model and computer vision algorithm that collected input from a camera, processed, and displayed to a monitor. Gstreamer is used for the bulk of this pipelining and processing; TI has invested in gstreamer plugins for various hardware accelerators that simplifies this task and gives plenty good performance for an interactive application. In my case, I'm stuck at 15 fps due to camera/USB2.0 throughput limitations. See some of the application notes on the [AM62A7](https://www.ti.com/product/AM62A7) product page or github repos like [edgeai-gst-apps-retail-checkout](https://github.com/TexasInstruments/edgeai-gst-apps-retail-checkout) for more details on building applications for deep learning workflows.
