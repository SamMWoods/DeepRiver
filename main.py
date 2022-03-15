import pixellib
from pixellib.instance import custom_segmentation

segment_image = custom_segmentation()
segment_image.inferConfig(num_classes=2, class_names=[
                          "butterfly", "squirrel"])
segment_image.load_model("mask_rcnn_models/mask_rcnn_model.005-1.137427.h5")
segment_image.segmentImage(
    "squirrel.jpg", output_image_name="image_new.jpg", show_bboxes=True)
