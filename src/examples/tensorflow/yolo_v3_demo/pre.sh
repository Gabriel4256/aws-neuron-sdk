#!/bin/bash

pip install pillow matplotlib pycocotools==2.0.2 --force --extra-index-url=https://pip.repos.neuron.amazonaws.com
curl -LO http://images.cocodataset.org/zips/val2017.zip
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q val2017.zip
unzip annotations_trainval2017.zip
ls
python yolo_v3_coco_saved_model.py ./yolo_v3_coco_saved_model
