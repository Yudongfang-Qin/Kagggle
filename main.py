import os
import random
import re
from PIL import Image
from sklearn.cluster import KMeans
import cv2
import numpy as np
import pandas as pd
import spicy
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
batch_size = 1
lambda_scale = 1.
dataset_path = "./train_images/video_"
EPOCHS = 8
STEPS = int(18582/batch_size)

#
# def annotations():
#     file = pd.read_csv('train.csv')
#     df = pd.DataFrame(file)
#     print(len(df))
#     data = df['annotations']
#     weight = []
#     height = []
#     weight_height = []
#     annotations = []
#     for i in range(0, len(df)):
#         annotations.append(list(map(int, re.findall(r"\d+\.?\d*", data[i]))))
#     #     for j in range(0, int(len(a) / 4)):
#     #         weight.append(int(a[4 * j + 2]))
#     #         height.append(int(a[4 * j + 3]))
#     #         weight_height.append([int(a[4 * j + 2]), int(a[4 * j + 3])])
#     # print(weight)
#
#     return annotations
def annotations():
    file = pd.read_csv('train.csv')
    df = pd.DataFrame(file)
    print(len(df))
    data = df['annotations']
    annotations = []
    for i in range(0, len(df)):
        annotations.append(list(map(int, re.findall(r"\d+\.?\d*", data[i]))))
    annotation_j = []
    print("ovo")
    annotations_video_id = [annotations[0:sum(df.video_id==0)],annotations[sum(df.video_id==0):sum(df.video_id==0)+sum(df.video_id==1)],annotations[sum(df.video_id==0)+sum(df.video_id==1):sum(df.video_id==0)+sum(df.video_id==1)+sum(df.video_id==2)]]
    video_frame_video_id = [df.video_frame[0:sum(df.video_id==0)],df.video_frame[sum(df.video_id==0):sum(df.video_id==0)+sum(df.video_id==1)],df.video_frame[sum(df.video_id==0)+sum(df.video_id==1):sum(df.video_id==0)+sum(df.video_id==1)+sum(df.video_id==2)]]
    return annotations_video_id, video_frame_video_id

[real_frame_3,video_frame_3] = annotations()


def K_mean(real_frame, num_shape):
    weight_height = []
    for h in range(0,3):
        for i in range(0, len(real_frame[h])):
            for j in range(0, int(len(real_frame[h][i]) / 4)):
                weight_height.append([int(real_frame[h][i][4 * j + 2]), int(real_frame[h][i][4 * j + 3])])
    km = KMeans(n_clusters=num_shape, init='k-means++', max_iter=60)

    km.fit(weight_height)
    centroids = km.cluster_centers_
    centroids_int = []
    for lines in centroids:
        centroids_int.append(list(map(round, lines)))
    # y_kmean = km.predict(weight_height)
    # plt.scatter(x = weight,y = height,c = y_kmean)
    # plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=10, alpha=0.7)
    # plt.xlabel('weight')
    # plt.ylabel('height')
    # plt.show()
    # print(centroids)
    return centroids_int


num_shape = 20
likely_size_frame = K_mean(real_frame_3, num_shape)


def plot_boxes_on_image(show_image_with_boxes, real_frame, color=[0, 0, 255], thickness=2):
    for i in range(0, len(real_frame)):
        cv2.rectangle(show_image_with_boxes,
                      pt1=(int(real_frame[i][1]), int(real_frame[i][0])),
                      pt2=(int(real_frame[i][1]) + int(real_frame[i][2]),
                           int(real_frame[i][0]) + int(real_frame[i][3])), color=color, thickness=thickness)
    show_image_with_boxes = cv2.cvtColor(show_image_with_boxes, cv2.COLOR_BGR2RGB)
    return show_image_with_boxes


def sorce(frame_0, frame_1):
    inter_width = np.minimum(frame_0[0] + frame_0[2], frame_1[0] + frame_1[2]) - np.maximum(frame_0[0], frame_1[0])
    inter_height = np.minimum(frame_0[1] + frame_0[3], frame_1[1] + frame_1[3]) - np.maximum(frame_0[1], frame_1[1])
    if inter_width <= 0 or inter_height <= 0:
        return -1
    else:
        intersection = inter_width * inter_height

        frame_0_area = frame_0[2] * frame_0[3]
        frame_1_area = frame_1[2] * frame_1[3]

        union = frame_0_area + frame_1_area - intersection  # 并集的面积
        return intersection / union


def bounding_box_regression(frame, truth):
    target_reg = np.zeros(shape=4)
    target_reg[0] = (truth[0] - frame[0]) / frame[2]
    target_reg[1] = (truth[1] - frame[1]) / frame[3]
    target_reg[2] = np.log(truth[2] / frame[2])
    target_reg[3] = np.log(truth[3] / frame[3])

    return target_reg


def decode_output(pred_bboxes, pred_scores, score_thresh=0.5):
    grid_x, grid_y = tf.range(80, dtype=tf.int32), tf.range(45, dtype=tf.int32)
    grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
    grid_x, grid_y = tf.expand_dims(grid_x, -1), tf.expand_dims(grid_y, -1)
    grid_xy = tf.stack([grid_x, grid_y], axis=-1)
    center_xy = grid_xy * 16
    center_xy = tf.cast(center_xy, tf.float32)
    quarter_likely_size_frame = []
    # for i in range(len(likely_size_frame)):
    #     quarter_likely_size_frame.append([int(likely_size_frame[i][0]*0.5),int(likely_size_frame[i][1]*0.5)])

    # anchor_xymin = center_xy - quarter_likely_size_frame
    size = likely_size_frame * tf.exp(pred_bboxes[..., 2:4])
    xy_min = tf.exp(pred_bboxes[..., 0:1]) + center_xy

    pred_bboxes = tf.concat([xy_min, size], axis=-1)
    pred_scores = pred_scores[..., 1]
    score_mask = pred_scores > score_thresh
    ovo_bboxes = pred_bboxes[score_mask]
    ovo_scores = pred_scores[score_mask]
    pred_bboxes_0 = tf.reshape(pred_bboxes[score_mask], shape=[-1, 4]).numpy()
    pred_scores_0 = tf.reshape(pred_scores[score_mask], shape=[-1, ]).numpy()
    return pred_scores_0, pred_bboxes_0


def nms(pred_boxes, pred_score, iou_thresh):
    selected_boxes = []
    n = 0
    while len(pred_boxes) > 0 and n < 10:
        n += 1
        max_idx = np.argmax(pred_score)
        selected_box = pred_boxes[max_idx]
        selected_boxes.append(selected_box)

        pred_boxes = np.concatenate([pred_boxes[:max_idx], pred_boxes[max_idx + 1:]])
        pred_score = np.concatenate([pred_score[:max_idx], pred_score[max_idx + 1:]])
        next_pred_boxes = []
        next_pred_score = []
        for i in range(len(pred_boxes)):
            ious = sorce(selected_box, pred_boxes[i])

            if ious >= iou_thresh:
                next_pred_boxes.append(pred_boxes[i])
                next_pred_score.append(pred_score[i])

        pred_boxes = next_pred_boxes
        pred_score = next_pred_score

    selected_boxes = np.array(selected_boxes)
    return selected_boxes


class RPNplus(tf.keras.Model):
    # VGG_MEAN = [103.939, 116.779, 123.68]
    def __init__(self):
        super(RPNplus, self).__init__()
        # conv1
        self.conv1_1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        #self.conv1_2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')

        # conv2
        self.conv2_1 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')
        #self.conv2_2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')

        # conv3
        self.conv3_1 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        #self.conv3_2 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        #self.conv3_3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        self.pool3 = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')

        # conv4
        self.conv4_1 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        #self.conv4_2 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        #self.conv4_3 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.pool4 = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')

        # conv5
        self.conv5_1 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        #self.conv5_2 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        #self.conv5_3 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.pool5 = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same')

        ## region_proposal_conv
        self.region_proposal_conv1 = tf.keras.layers.Conv2D(256, kernel_size=[5, 2],
                                                            activation=tf.nn.relu,
                                                            padding='same', use_bias=False)
        self.region_proposal_conv2 = tf.keras.layers.Conv2D(512, kernel_size=[5, 2],
                                                            activation=tf.nn.relu,
                                                            padding='same', use_bias=False)
        self.region_proposal_conv3 = tf.keras.layers.Conv2D(512, kernel_size=[5, 2],
                                                            activation=tf.nn.relu,
                                                            padding='same', use_bias=False)
        ## Bounding Boxes Regression layer
        self.bboxes_conv = tf.keras.layers.Conv2D(80, kernel_size=[1, 1],
                                                  padding='same', use_bias=False)
        ## Output Scores layer
        self.scores_conv = tf.keras.layers.Conv2D(40, kernel_size=[1, 1],
                                                  padding='same', use_bias=False)

    def call(self, x, training=False):
        h = self.conv1_1(x)
        #h = self.conv1_2(h)
        h = self.pool1(h)

        h = self.conv2_1(h)
        #h = self.conv2_2(h)
        h = self.pool2(h)

        h = self.conv3_1(h)
        #h = self.conv3_2(h)
        #h = self.conv3_3(h)
        h = self.pool3(h)
        # Pooling to same size
        pool3_p = tf.nn.max_pool2d(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pool3_proposal')
        pool3_p = self.region_proposal_conv1(pool3_p)  # [1, 45, 60, 256]

        h = self.conv4_1(h)
        #h = self.conv4_2(h)
        #h = self.conv4_3(h)
        h = self.pool4(h)
        pool4_p = self.region_proposal_conv2(h)  # [1, 45, 60, 512]

        h = self.conv5_1(h)
        #h = self.conv5_2(h)
        #h = self.conv5_3(h)
        pool5_p = self.region_proposal_conv2(h)  # [1, 45, 60, 512]

        region_proposal = tf.concat([pool3_p, pool4_p, pool5_p], axis=-1)  # [1, 45, 60, 1280]

        conv_cls_scores = self.scores_conv(region_proposal)  # [1, 45, 60, 18]
        conv_cls_bboxes = self.bboxes_conv(region_proposal)  # [1, 45, 60, 36]

        cls_scores = tf.reshape(conv_cls_scores, [batch_size, 45, 80, 20, 2])
        cls_bboxes = tf.reshape(conv_cls_bboxes, [batch_size, 45, 80, 20, 4])

        return cls_scores, cls_bboxes


def step_1(test_frame_0):
    pos_thresh = 0.5
    neg_thresh = 0.1
    iou_thresh = 0.5
    grid_width = 16
    grid_height = 16
    image_height = 720
    image_width = 1280
    num_shape = len(likely_size_frame)

    target_scores = np.zeros(shape=[45, 80, num_shape, 2])  # 0: background, 1: foreground, ,
    target_bboxes = np.zeros(shape=[45, 80, num_shape, 4])  # t_x, t_y, t_w, t_h
    target_masks = np.zeros(shape=[45, 80, num_shape])  # negative_samples: -1, positive_samples: 1
    for i in range(45):
        for j in range(80):
            for k in range(num_shape):
                for l in range(int(len(test_frame_0) / 4)):
                    test_frame = test_frame_0[0 + 4 * l:4 + 4 * l]
                    center_x = j * grid_width + grid_width * 0.5
                    center_y = i * grid_height + grid_height * 0.5
                    xmin = center_x - likely_size_frame[k][0] * 0.5
                    ymin = center_y - likely_size_frame[k][1] * 0.5
                    xmax = center_x + likely_size_frame[k][0] * 0.5
                    ymax = center_y + likely_size_frame[k][1] * 0.5
                    # ignore cross-boundary anchors
                    if (xmin > -5) & (ymin > -5) & (xmax < (image_width + 5)) & (ymax < (image_height + 5)):
                        anchor_boxes = np.array([xmin, ymin, xmax, ymax])
                        # anchor_boxes = np.expand_dims(anchor_boxes, axis=0)
                        # compute iou between this anchor and all ground-truth boxes in image.
                        anchor_boxes_0 = [anchor_boxes[0], anchor_boxes[1], anchor_boxes[2] - anchor_boxes[0],
                                          anchor_boxes[3] - anchor_boxes[1]]
                        ious = sorce(anchor_boxes_0, test_frame)
                        positive_masks = ious > pos_thresh
                        negative_masks = ious < neg_thresh
                        if np.any(positive_masks):
                            target_scores[i, j, k, 1] = 1.
                            target_masks[i, j, k] = 1  # labeled as a positive sample
                            # find out which ground-truth box matches this anchor
                            max_iou_idx = np.argmax(ious)
                            selected_gt_boxes = test_frame[max_iou_idx:max_iou_idx + 4]
                            target_bboxes[i, j, k] = bounding_box_regression(anchor_boxes_0, selected_gt_boxes)

                        if np.all(negative_masks):
                            target_scores[i, j, k, 0] = 1.
                            target_masks[i, j, k] = -1  # labeled as a negative sample

    return target_scores, target_bboxes, target_masks


def process_image_label(image_path, test_frame_0):
    raw_image = cv2.imread(image_path)

    target = step_1(test_frame_0)
    return raw_image, target


def create_image_path_generator():

    while True:
        for i in [0, 1, 2]:
            for j in range(len(real_frame_3[i])):
                try:
                    image_paths = dataset_path + str(i) + "/" + str(video_frame_3[i][j]) + ".jpg"
                    test_frame_0 = real_frame_3[i][j]
                    if test_frame_0 == []:
                        continue
                    yield image_paths, test_frame_0
                except StopIteration:
                    continue


def DataGenerator():
    image_label_path_generator = create_image_path_generator()
    while True:
        try:

            images = np.zeros(shape=[batch_size, 720, 1280, 3], dtype=float)
            target_scores = np.zeros(shape=[batch_size, 45, 80, num_shape, 2], dtype=float)
            target_bboxes = np.zeros(shape=[batch_size, 45, 80, num_shape, 4], dtype=float)
            target_masks = np.zeros(shape=[batch_size, 45, 80, num_shape], dtype=int)
            for i in range(batch_size):
                image_path, test_frame_0 = next(image_label_path_generator)
                image, target = process_image_label(image_path, test_frame_0)
                images[i] = image
                target_scores[i] = target[0]
                target_bboxes[i] = target[1]
                target_masks[i] = target[2]
            yield images, target_scores, target_bboxes, target_masks
        except StopIteration:
            return


def compute_loss(target_scores, target_bboxes, target_masks, pred_scores, pred_bboxes):
    """
    target_scores shape: [1, 45, 60, 9, 2],  pred_scores shape: [1, 45, 60, 9, 2]
    target_bboxes shape: [1, 45, 60, 9, 4],  pred_bboxes shape: [1, 45, 60, 9, 4]
    target_masks  shape: [1, 45, 60, 9]
    """
    score_loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_scores, logits=pred_scores)
    foreground_background_mask = (np.abs(target_masks) == 1).astype(int)
    score_loss = tf.reduce_sum(score_loss * foreground_background_mask, axis=[1, 2, 3]) / (
                np.sum(foreground_background_mask) + 0.00001)
    score_loss = tf.reduce_mean(score_loss)

    boxes_loss = tf.abs(target_bboxes - pred_bboxes)
    boxes_loss = 0.5 * tf.pow(boxes_loss, 2) * tf.cast(boxes_loss < 1, tf.float32) + (boxes_loss - 0.5) * tf.cast(
        boxes_loss >= 1, tf.float32)
    boxes_loss = tf.reduce_sum(boxes_loss, axis=-1)
    foreground_mask = (target_masks > 0).astype(np.float32)
    boxes_loss = tf.reduce_sum(boxes_loss * foreground_mask, axis=[1, 2, 3]) / (np.sum(foreground_mask) + 0.00001)
    boxes_loss = tf.reduce_mean(boxes_loss)

    return score_loss, boxes_loss


def main():
    # real_frame = annotations()
    # num_shape = 30
    # likely_size_frame = np.array(K_mean(real_frame, num_shape))
    print("hello world")

    TrainSet = DataGenerator()

    model = RPNplus()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    writer = tf.summary.create_file_writer("./log")
    global_steps = tf.Variable(0, trainable=False, dtype=tf.int64)

    for epoch in range(EPOCHS):
        for step in range(STEPS):
            global_steps.assign_add(1)
            image_data, target_scores, target_bboxes, target_masks = next(TrainSet)
            with tf.GradientTape() as tape:
                pred_scores, pred_bboxes = model(image_data)
                # print("ovo")
                score_loss, boxes_loss = compute_loss(target_scores, target_bboxes, target_masks, pred_scores,
                                                      pred_bboxes)
                total_loss = score_loss + lambda_scale * boxes_loss
                gradients = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                print(
                    "=> epoch %d  step %d  total_loss: %.6f  score_loss: %.6f  boxes_loss: %.6f" % (epoch + 1, step + 1,
                                                                                                    total_loss.numpy(),
                                                                                                    score_loss.numpy(),
                                                                                                    boxes_loss.numpy()))
            # writing summary data
            with writer.as_default():
                tf.summary.scalar("total_loss", total_loss, step=global_steps)
                tf.summary.scalar("score_loss", score_loss, step=global_steps)
                tf.summary.scalar("boxes_loss", boxes_loss, step=global_steps)
            writer.flush()
        model.save_weights("RPN.npy")


if __name__ == '__main__':
    main()
