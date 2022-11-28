import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
from matplotlib import pyplot as plt


def fuse_with_pil(images):
    '''
    Creates a blank image and pastes input images

    Args:
      images (list of numpy arrays) - numpy array representations of the images to paste

    Returns:
      PIL Image object containing the images
    '''

    widths = (image.shape[1] for image in images)
    heights = (image.shape[0] for image in images)
    total_width = sum(widths)
    max_height = max(heights)

    new_im = PIL.Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        pil_image = PIL.Image.fromarray(np.uint8(im))
        new_im.paste(pil_image, (x_offset,0))
        x_offset += im.shape[1]

    return new_im


def give_color_to_annotation(annotation, colors, n_classes=11):
    '''
    Converts a 2-D annotation to a numpy array with shape (height, width, 3) where
    the third axis represents the color channel. The label values are multiplied by
    255 and placed in this axis to give color to the annotation

    Args:
      annotation (numpy array) - label map array

    Returns:
      the annotation array with an additional color channel/axis
    '''
    seg_img = np.zeros( (annotation.shape[0],annotation.shape[1], 3) ).astype('float')

    # colors = [tuple(np.random.randint(256, size=3) / 255.0) for i in range(n_classes)]

    for c in range(n_classes):
        segc = (annotation == c)
        seg_img[:,:,0] += segc*( colors[c][0] * 255.0)
        seg_img[:,:,1] += segc*( colors[c][1] * 255.0)
        seg_img[:,:,2] += segc*( colors[c][2] * 255.0)

    return seg_img


def show_annotation_and_prediction(image, annotation, prediction, iou_list, dice_score_list, colors):
    '''
    Displays the images with the ground truth and predicted label maps. Also overlays the metrics.

    Args:
      image (numpy array) -- the input image
      annotation (numpy array) -- the ground truth label map
      prediction (numpy array) -- the predicted label map
      iou_list (list of floats) -- the IOU values for each class
      dice_score_list (list of floats) -- the Dice Score for each class
    '''

    new_ann = np.argmax(annotation, axis=2)
    true_img = give_color_to_annotation(new_ann, colors)
    pred_img = give_color_to_annotation(prediction, colors)

    image = image + 1
    image = image * 127.5
    image = np.reshape(image, (image.shape[0], image.shape[1],))
    image = np.uint8(image)
    images = [image, np.uint8(pred_img), np.uint8(true_img)]

    metrics_by_id = [(idx, iou, dice_score) for idx, (iou, dice_score) in enumerate(zip(iou_list, dice_score_list)) if iou > 0.0 and idx < 10]
    metrics_by_id.sort(key=lambda tup: tup[1], reverse=True)  # sorts in place

    display_string_list = [f"{idx}: IOU: {iou:.4f} Dice Score: {dice_score:.4f}" for idx, iou, dice_score in metrics_by_id]
    display_string = "\n".join(display_string_list)

    plt.figure(figsize=(15, 4))

    titles = {0: "Original", 1: "Prediction", 2: "Annotation"}
    for idx, im in enumerate(images):
        plt.subplot(1, 3, idx+1)
        if idx == 1:
            plt.xlabel(display_string)
        
        plt.title(titles[idx])
        plt.xticks([])
        plt.yticks([])
        plt.imshow(im)


def show_annotation_and_image(image, annotation, colors):
    '''
    Displays the image and its annotation side by side

    Args:
      image (numpy array) -- the input image
      annotation (numpy array) -- the label map
    '''
    new_ann = np.argmax(annotation, axis=2)
    seg_img = give_color_to_annotation(new_ann, colors)

    image = image + 1
    image = image * 127.5
    image = np.reshape(image, (image.shape[0], image.shape[1],))

    image = np.uint8(image)
    images = [image, seg_img]

    images = [image, seg_img]
    fused_img = fuse_with_pil(images)
    plt.imshow(fused_img)


def list_show_annotation(dataset, num_images, colors):
    '''
    Displays images and its annotations side by side

    Args:
      dataset (tf Dataset) -- batch of images and annotations
      num_images (int) -- number of images to display
    '''
    ds = dataset.unbatch()

    plt.figure(figsize=(20, 15))
    plt.title("Images And Annotations")
    plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.05)

    for idx, (image, annotation) in enumerate(ds.take(num_images)):
        plt.subplot(5, 5, idx + 1)
        plt.yticks([])
        plt.xticks([])
        show_annotation_and_image(image.numpy(), annotation.numpy(), colors)

