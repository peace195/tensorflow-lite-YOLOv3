# tensorflow-lite-yolo-v3

Implementation of YOLO v3 object detector in Tensorflow lite

## Setup env
    `docker build -t tflite .`
    `docker run -it -v /home/peace195/tensorflow-lite-yolo-v3:/root/ tflite`
    
## How to run:
To run demo type this in the command line:

1. Download COCO class names file: `wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names`
2. Download and convert model weights:    
    1. Download binary file with desired weights: 
        1. Full weights: `wget https://pjreddie.com/media/files/yolov3.weights`
        1. Tiny weights: `wget https://pjreddie.com/media/files/yolov3-tiny.weights` 
        1. SPP weights: `wget https://pjreddie.com/media/files/yolov3-spp.weights` 
    2. Run `python ./convert_weights.py` and `python ./convert_weights_pb.py`        
3. Run `tflite_convert --saved_model_dir=saved_model/ --output_file yolo_v3.tflite --saved_model_signature_key='predict'`


####Optional Flags
convert_weights_pb.py:
    1. `--class_names`
            1. Path to the class names file
    2. `--weights_file`
        1. Path to the desired weights file    
    3. `--data_format`
        1.  `NCHW` (gpu only) or `NHWC`
    4. `--tiny`
        1. Use yolov3-tiny
    5. `--spp`
        1. Use yolov3-spp
    6. `--output_graph`
        1. Location to write the output .pb graph to
        
Contact me if you have any issues: binhtd.hust@gmail.com / Binh Do