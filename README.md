# sheet-metal-defect-detection

As part of curriculum, a research paper was written upon detection of defects during the production of steel sheets in factories. The dataset was obtained from Kaggle- Severstal defect detection dataset. 

Extensive EDA was performed, which included conversion of rle to mask image, removal of no defect images and many more mentioned in the notebook

Data augmentation was performed to enhance the models prediction. Initaially 3 models U-net, Xception and mas-rcnn waas chosen.

## U-NET

The architecture looks like a ‘U’ which justifies its name. This architecture consists of three sections: The contraction, The bottleneck, and the expansion section.
As a general convolutional neural network focuses its task on image classification, where input is an image and output is one label, but in our case, it requires us not only to distinguish whether there is a defect, but also to localize the area of abnormality. The reason UNET is able to localize and distinguish borders is by doing classification on every pixel, so the input and output share the same size. 

## MASK-RCNN

Mask RCNN is an instance segmentation model which can perform localization and classification at the same time. FPN (Feature Pyramid Network) extracts the features. RPN (Region Proposal Network) scans over the FPN features to suggest ROI (Region of Interest). The ROI Align classifies the object and is passed on to a mask generator which gives the region of defect. We use the FPN as the backbone of the model. MaskRCNN contains a lot of parameter and comparatively takes huge amount of time to train. 

## XCEPTION-V1

Xception is the extreme version of the Inception model developed by Google. It no longer uses the standard inception modules and instead relies on depth wise separable convolutions. The Xception model has 14 modules consisting of linear residual connections except for the first and last modules. In total it has 36 convolutional layers which function as feature extractors. The output from the last layer in the Xception model is upsampled and passed through a 1x1 convolution filter with sigmoid activation which predicts the mask.

# PROCEDURE

1.DATASET
Our dataset consists of 12,568 training images and 5,506 test images. The dataset had the type of defect category and mask (defect) for the training images. There were 4 types of unique defect. Since, no testing image had any defect mask we discarded them. The training set consisted of 6666 unique images having at least one defect. The images had a resolution of 256x1600x3 pixel.

2.PREPROCESSING
Since the dataset contains data compressed in RLE format we had to convert it into a Boolean mask 2D matrix of dimensions (256,1600,1) The images in the dataset had 3 channels so we converted the image to greyscale which is faster to train due to presence of only a single channel Augmented the dataset to make it more robust for training Used hold out cross validation with 85% training and 15% as validation images


# RESULTS

Xception performed very well in defect type 4 which were substantial, but fared poorly in rare and complex defects. MaskRCNN could handle the rare defects very well. U-NET was a versatile model as it could handle all the defects and most of the predicted masks were close to the ground truth. But it made false predictions for no defect type class. Dice coefficient of each model is shown came out to be:
Xception	- 68.87% in 27 epochs
Mask RCNN	- 71.93% in 19 epochs
U-NET	- 80.21% in 103 epochs










