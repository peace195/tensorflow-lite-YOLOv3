# tensorflow-lite-yolo-v3

Convert the weights of YOLO v3 object detector into tensorflow lite format. It can be served for tensorflow serving as well.

## Setup env
    docker build -t tflite .
    docker run -it -v /home/peace195/tensorflow-lite-yolo-v3:/root/ tflite
    
## How to run

1. Download COCO class names file: `wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names`
2. Download binary file with desired weights: 
    - Full weights: `wget https://pjreddie.com/media/files/yolov3.weights`
    - Tiny weights: `wget https://pjreddie.com/media/files/yolov3-tiny.weights` 
    - SPP weights: `wget https://pjreddie.com/media/files/yolov3-spp.weights` 
3. Convert .weights to .pb saved_model `python ./convert_weights_pb.py` (this can be used for tensorflow serving)      
4. Convert .pb to .tflite `tflite_convert --saved_model_dir=saved_model/ --output_file yolo_v3.tflite --saved_model_signature_key='predict'`


Optional Flags

convert_weights_pb.py:

    --class_names
        Path to the class names file
    --weights_file
        Path to the desired weights file    
    --data_format
        `NCHW` (gpu only) or `NHWC`
    --tiny
        Use yolov3-tiny
    --spp
        Use yolov3-spp
    --output_graph
        Location to write the output .pb graph
        
Contact me if you have any issues: binhtd.hust@gmail.com / Binh Do