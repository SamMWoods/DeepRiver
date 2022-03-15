import pixellib
from pixellib.custom_train import instance_custom_training

# Visualize Dataset
# vis_img = instance_custom_training()
# vis_img.load_dataset("Plastic")
# vis_img.visualize_sample()

# Train a custom model Using your dataset
train_maskrcnn = instance_custom_training()
train_maskrcnn.modelConfig(network_backbone="resnet50",
                           num_classes=2, batch_size=2)
train_maskrcnn.load_pretrained_model("mask_rcnn_coco.h5")
train_maskrcnn.load_dataset("Nature")
train_maskrcnn.train_model(
    num_epochs=300, augmentation=True, path_trained_models="mask_rcnn_models")

# Model Evaluation
# train_maskrcnn = instance_custom_training()
# train_maskrcnn.modelConfig(network_backbone="resnet50", num_classes=2)
# train_maskrcnn.load_dataset("Nature")
# train_maskrcnn.evaluate_model("mask_rcnn_models")
