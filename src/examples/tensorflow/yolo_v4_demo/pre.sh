#!/bin/bash

pip install neuron-cc 'tensorflow-neuron<2' requests pillow matplotlib pycocotools==2.0.1 torch~=1.5.0 --force \
    --extra-index-url=https://pip.repos.neuron.amazonaws.com
curl -LO http://images.cocodataset.org/zips/val2017.zip
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q val2017.zip
unzip annotations_trainval2017.zip
python3 yolo_v4_coco_saved_model.py ./yolo_v4_coco_saved_model
