import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from time import gmtime, strftime
import torch.nn.functional as F

import cvbaselineutils.config as conf
from cvdatasetutils.pascalvoc import PascalVOCOR

from cvbaselineutils.yolodataset import YoloTrainingDataset

from mltrainingtools.cmdlogging import section_logger
from mltrainingtools.dnnutils import load_pretrained, create_lr_policy
from mltrainingtools.cvutils import IoU

from scipy.special import expit

import numpy as np
import copy
import os
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches


EPSILON = 1e-9
GLOBAL_EVALUATIONS_FILE="yolo_eval.csv"
GLOBAL_STATS_FILE="yolo_perimage_stats.csv"

def get_absolute_coordinates(anchor_box_prior, cell_coords, tw, th, tx, ty, tp):
    p = expit(tp)
    bx = expit(tx) + cell_coords[0]
    by = expit(ty) + cell_coords[1]
    bh = anchor_box_prior[0] * np.exp(th)
    bw = anchor_box_prior[1] * np.exp(tw)
    return bw, bh, bx, by, p


class Yolo(nn.Module):
    def __init__(self, S, B, num_classes, lambda_noobj, lambda_coord, anchors, backbone="VGG16"):
        super(Yolo, self).__init__()
        self.middle_layer_size = 4096
        input_size, feature_extractor = load_pretrained(self.middle_layer_size, backbone, True, False)

        self.input_size = input_size
        self.S = S
        self.B = B
        self.num_classes = num_classes
        self.lambda_noobj = lambda_noobj
        self.lambda_coord = lambda_coord

        self.anchors = anchors

        self.feature_extractor = feature_extractor
        self.anchor_size = 5
        self.cell_size = self.anchor_size * self.B + self.num_classes
        self.output_size = S * S * self.cell_size

        detection_layers = [
            nn.BatchNorm1d(self.middle_layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.middle_layer_size, self.output_size),
            nn.BatchNorm1d(self.output_size),
            nn.ReLU(inplace=True)
        ]

        self.detection_net = nn.Sequential(*detection_layers)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.detection_net(x)
        return x

    def get_input_size(self):
        return self.input_size

    def responsible_anchor(self, i_x, i_y, j, x, y, w, h):
        max_anchor = 0
        max_IoU = 0

        for i, anchor in enumerate(self.anchors):
            new_IoU = IoU(x, y, h, w, (i_x + 0.5) * self.cell_size, (i_y + 0.5) * self.cell_size, anchor[0], anchor[1])

            if new_IoU > max_IoU:
                max_anchor = i
                max_IoU = new_IoU

        return max_anchor == j

    def responsible_cell(self, i_x, i_y, x, y):
        return (i_x * self.cell_size < x) and (x < (i_x + 1) * self.cell_size) and \
               (i_y * self.cell_size < y) and (y < (i_y + 1) * self.cell_size)

    def anchor_without_objects(self, i_x, i_y, j, x, y, w, h):
        return not self.responsible_anchor(i_x, i_y, j, x, y, w, h)

    def get_anchor_values(self, anchor, anchor_box_index, cell_coords, anchor_box_prior):
        tx = anchor[anchor_box_index + 1].cpu().detach().numpy().item()
        ty = anchor[anchor_box_index + 2].cpu().detach().numpy().item()
        th = anchor[anchor_box_index + 3].cpu().detach().numpy().item()
        tw = anchor[anchor_box_index + 4].cpu().detach().numpy().item()
        tp = anchor[anchor_box_index].cpu().detach().numpy().item()

        bw, bh, bx, by, p = get_absolute_coordinates(anchor_box_prior, cell_coords, tw, th, tx, ty, tp)

        return p, bx, by, bh, bw

    def get_target_values(self, anchor, anchor_box_index):
        p = anchor[anchor_box_index]
        tx = anchor[anchor_box_index + 1].cpu().detach().numpy().item()
        ty = anchor[anchor_box_index + 2].cpu().detach().numpy().item()
        th = anchor[anchor_box_index + 3].cpu().detach().numpy().item()
        tw = anchor[anchor_box_index + 4].cpu().detach().numpy().item()

        return p, tx, ty, th, tw

    def get_cell_coords(self, i_x, i_y):
        return i_x / self.S, i_y / self.S

    def get_box_coords(self, anchor_box_index):
        return self.anchors[anchor_box_index]

    def get_objectness(self, x, y, h, w, x_h, y_h, h_h, w_h):
        if sum([x, y, h, w]) < EPSILON:
            return 0
        else:
            return 1 - IoU(x, y, h, w, x_h, y_h, h_h, w_h)

    def create_loss_function(self):
        def loss_function(output, target):
            accumulator = 0
            M, _ = target.size()

            for i_x in range(self.S):
                for i_y in range(self.S):
                    cell_coords, cell_index, class_end, class_init = self.extract_cell_coords(i_x, i_y)

                    for e in range(M):
                        for j in range(self.B):
                            anchor_box_index = cell_index + j * self.anchor_size

                            p, x, y, h, w = self.get_target_values(target[e], anchor_box_index)
                            p_h, x_h, y_h, h_h, w_h = self.get_anchor_values(output[e], anchor_box_index,
                                                                             cell_coords,
                                                                             self.get_box_coords(j))

                            objectness = (p_h - p * self.get_objectness(x, y, h, w, x_h, y_h, h_h, w_h))**2

                            accumulator += self.responsible_anchor(i_x, i_y, j, x, y, w, h) * objectness

                            if p < EPSILON:
                                continue

                            box_agreement = self.lambda_coord * (
                                (x - x_h) ** 2 + (y - y_h) ** 2 +
                                (w - w_h) ** 2 + (w - w_h) ** 2)

                            accumulator += self.responsible_anchor(i_x, i_y, j, x, y, w, h) * box_agreement
                            accumulator += self.lambda_noobj * self.anchor_without_objects(i_x, i_y, j, x, y, w, h) * objectness

                            c = target[e, class_init:class_end]
                            c_h = output[e, class_init:class_end]

                            class_agreement = torch.sum((c - c_h) ** 2)

                            accumulator += self.responsible_cell(i_x, i_y, x, y) * class_agreement

            accumulator /= M

            return accumulator

        return loss_function

    def extract_cell_coords(self, i_x, i_y):
        cell_index = ((i_x * self.S) + i_y) * self.cell_size
        class_init = cell_index + self.B * self.anchor_size
        class_end = class_init + self.num_classes
        cell_coords = self.get_cell_coords(i_x, i_y)
        return cell_coords, cell_index, class_end, class_init


def train_model(model, name, output_path, lr, epoch_blocks, num_epochs, dataloaders, optimizer, scheduler, loss_criterion,
                device, epoch_loss_history, epoch_size=conf.STEPS_PER_EPOCH, batch_log_freq=conf.BATCH_LOG_FREQUENCY,
                skip_eval=False):

    log = section_logger(1)
    step = section_logger(2)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_eae = float("inf")

    for epoch_block in range(epoch_blocks):
        for epoch in range(num_epochs):
            log('Processing epoch {}'.format(epoch))

            for phase in ['train', 'eval']:
                log('Starting [{}] phase'.format(phase))

                if phase == 'train':
                    model.train()
                elif skip_eval:
                    continue
                else:
                    model.eval()

                running_loss = 0.0
                batch_n = 0

                log('Starting optimization')
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = loss_criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item()

                    if batch_n % batch_log_freq == 0:
                        step('[{}]: {}'.format(batch_n, loss.item()))

                    if batch_n > epoch_size:
                        break

                    batch_n += 1

                log('Calculating epoch stats')
                epoch_loss = running_loss / batch_n

                log('Saving best model')
                if phase == 'val' and epoch_loss < best_eae:
                    best_eae = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

                epoch_loss_history[phase].append(epoch_loss)

            scheduler.step(epoch)

        save_results(name, output_path, model, epoch_loss_history, lr)

    log('Finishing epochs block')
    model.load_state_dict(best_model_wts)
    return model


def save_results(name, output_path, model, epoch_log_history, lr):
    loss_df = pd.DataFrame(epoch_log_history)
    loss_df.loc[:, 'name'] = name
    loss_df.loc[:, 'lr'] = lr
    loss_df.to_csv(os.path.join(output_path, name + '.csv'))
    torch.save(model.state_dict(), os.path.join(output_path, name + '.pt'))
    #TODO: Save the list of classes, or they won't be recoverable at runtime


def execute_single_model(name, output_path, dataloaders, S, B, num_classes, anchors, lr=0.0001, lambda_coord=0.5,
                         lambda_noobj=0.5, epoch_blocks=conf.EPOCH_BLOCKS, num_epochs=conf.NUM_EPOCHS,
                         model_file=None):

    if model_file is not None:
        model = load_model(model_file, S, B, num_classes, anchors)
    else:
        model = Yolo(S, B, num_classes, lambda_noobj, lambda_coord, anchors)

    model.cuda()
    model_parameters = model.parameters()

    optimizer = optim.SGD(params=model_parameters, lr=lr, momentum=0.9, weight_decay=0.001)
    scheduler = LambdaLR(optimizer, lr_lambda=create_lr_policy([10, 20, 30]))

    loss_criterion = model.create_loss_function()

    device = torch.device("cuda:" + conf.CUDA_DEVICE if torch.cuda.is_available() else "cpu")

    epoch_loss_history = {
        "eval": [],
        "train": []
    }

    model = train_model(model, name, output_path, lr, epoch_blocks, num_epochs, dataloaders, optimizer, scheduler,
                        loss_criterion, device, epoch_loss_history)


def execute(model_file=None):
    S = 7
    B = 5
    batch_size = conf.BATCH_SIZE
    dataset, yolo_dataloaders, yoloset = prepare_datasets(B, S, batch_size)

    name = strftime("%Y%m%d%H%M", gmtime())

    execute_single_model(name, conf.DATA_FOLDER, yolo_dataloaders, S, B, len(dataset.classes), yoloset.anchorbs,
                         model_file=model_file)


def prepare_datasets(B, S, batch_size):
    dataset = PascalVOCOR("/home/dani/Documentos/Proyectos/Doctorado/Datasets/VOC2012/VOCdevkit/VOC2012")
    yoloset = YoloTrainingDataset(dataset, S, B)
    train_size = int(0.8 * len(yoloset))
    test_size = len(yoloset) - train_size
    yoloset_train, yoloset_val = torch.utils.data.random_split(yoloset, [train_size, test_size])
    yolo_dataloaders = {
        'train': DataLoader(yoloset_train, batch_size=batch_size, shuffle=True, num_workers=7),
        'eval': DataLoader(yoloset_val, batch_size=batch_size, shuffle=True, num_workers=7)
    }
    return dataset, yolo_dataloaders, yoloset


def load_model(path, S, B, num_classes, anchors, lambda_coord=0.5, lambda_noobj=0.5):
    model = Yolo(S, B, num_classes, lambda_noobj, lambda_coord, anchors)
    model.load_state_dict(torch.load(path))
    model.eval()

    return model


def get_box_values(objects, w, h):
    boxes = []

    for obj in objects:
        box = {
            'x': int((obj['x'] - obj['w']/2) * w),
            'y': int((obj['y'] - obj['h']/2) * h),
            'h': int(obj['h'] * h),
            'w': int(obj['w'] * w),
            'class': obj['class']
        }

        boxes.append(box)

    return boxes


def paint_image(evaluation_path, m, type, image, pixel_objects, classes, colormap=None):
    title = "Yolo OR"

    fig = plt.figure(frameon=False)
    plt.imshow(image)
    ax = plt.Axes(fig, [0., 0., 1., 0.9])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.title(title)

    for obj in pixel_objects:
        box = patches.Rectangle((obj['x'], obj['y']), obj['w'], obj['h'],
                                linewidth=1, edgecolor='g', facecolor='none')

        ax.add_patch(box)

        ax.text(obj['x'], obj['y'], classes[obj['class']],
                horizontalalignment='left',
                verticalalignment='bottom',
                bbox=dict(facecolor='green', alpha=0.5))

    if colormap:
        ax.imshow(image, cmap=colormap)
    else:
        ax.imshow(image)

    ax.set_axis_off()
    fig.savefig(os.path.join(evaluation_path, "example_{}_{}.png".format(type, m)), dpi=120)
    plt.close(fig)


def paint_class_map(evaluation_path, m, type, image, detections, classes, width, height, colormap=None):
    title = "Yolo OR"

    fig = plt.figure(frameon=False)
    plt.imshow(image)
    ax = plt.Axes(fig, [0., 0., 1., 0.9])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.title(title)
    class_color = plt.get_cmap('prism')

    anchor_width = int(width / np.sqrt(len(detections['classes'])))
    anchor_height = anchor_width

    for anchor in detections['classes']:
        color = class_color(anchor['class'])
        box = patches.Rectangle((anchor['x'] * width, anchor['y'] * height), anchor_width, anchor_height,
                                linewidth=1, edgecolor='black', facecolor=color, fill=True, alpha=0.30)

        ax.add_patch(box)

        ax.text(anchor['x'] * width + anchor_width/2, anchor['y'] * height + anchor_height/2, classes[anchor['class']],
                horizontalalignment='center',
                verticalalignment='center',
                size='smaller',
                bbox=None)

    if colormap:
        ax.imshow(image, cmap=colormap)
    else:
        ax.imshow(image)

    ax.set_axis_off()
    fig.savefig(os.path.join(evaluation_path, "example_{}_{}_classmap.png".format(type, m)), dpi=120)
    plt.close(fig)


def show_image_results(model_name, detections, images, labels, classes):
    evaluation_path = os.path.join(conf.EVALUATIONS_FOLDER, model_name)
    os.makedirs(evaluation_path, exist_ok=True)

    M, c, w, h = images.shape

    current_batch_name = strftime("%H%M%S", gmtime())

    for m in range(M):
        # Get the ground truth objects
        boxes_gt = get_box_values(labels[m]['objects'], w, h)

        # Get the pixel values of all the objects in the image
        boxes = get_box_values(detections[m]['objects'], w, h)

        # Paint the image with the detected objects
        paint_image(evaluation_path, m, "{}_detected".format(current_batch_name),
                    images[m, :, :, :].permute(1, 2, 0).numpy(), boxes, classes)

        # Paint the image with the ground truth
        paint_image(evaluation_path, m, "{}_gt".format(current_batch_name),
                    images[m, :, :, :].permute(1, 2, 0).numpy(), boxes_gt, classes)

        # Paint the class map for the image
        paint_class_map(evaluation_path, m, "{}_detected".format(current_batch_name),
                        images[m, :, :, :].permute(1, 2, 0).numpy(), detections[m], classes, w, h)


def get_map_confusion_matrix(detected_objects, true_objects, target_class=None, threshold=0.5):
    matrix = {
        'tp': 0,
        'fp': 0,
        'fn': 0
    }

    for detected in detected_objects:
        for ground_truth in true_objects:
            if target_class is not None and ground_truth['class'] != target_class:
                continue

            overlap = IoU(detected['x'], detected['y'], detected['h'], detected['w'],
                          ground_truth['x'], ground_truth['y'], ground_truth['h'], ground_truth['w'])

            if overlap > threshold:
                if detected['class'] == ground_truth['class']:
                    matrix['tp'] += 1
                else:
                    matrix['fn'] += 1
            else:
                matrix['fp'] += 1

    return matrix


def evaluate_model(detections, ground_truths, target_class=None):
    stats = {
        'precision': [],
        'recall': []
    }

    for i, detection in enumerate(detections):
        ground_truth = ground_truths[i]

        confusion_matrix = get_map_confusion_matrix(detection['objects'], ground_truth['objects'], target_class)

        total_positives = confusion_matrix['tp'] + confusion_matrix['fp']

        if total_positives > 0:
            precision = confusion_matrix['tp'] / total_positives
        else:
            precision = 0.0

        real_positives = confusion_matrix['tp'] + confusion_matrix['fn']

        if real_positives > 0:
            recall = confusion_matrix['tp'] / real_positives
        else:
            recall = 0.0

        stats['precision'].append(precision)
        stats['recall'].append(recall)

    return pd.DataFrame(stats)


def save_evaluations(model_name, mAP, per_class_map, global_stats, classes):
    evaluation_path = os.path.join(conf.EVALUATIONS_FOLDER, GLOBAL_EVALUATIONS_FILE)
    evaluation_folder = os.path.join(conf.EVALUATIONS_FOLDER, model_name)
    os.makedirs(evaluation_folder, exist_ok=True)
    stats_path = os.path.join(evaluation_folder, GLOBAL_EVALUATIONS_FILE)

    data_raw = {("map_" + classes[key]): [value] for key, value in enumerate(per_class_map)}
    data_raw['map_global'] = [mAP]
    data_raw['name'] = [model_name]

    df = pd.DataFrame(data_raw)

    if os.path.isfile(evaluation_path):
        df.to_csv(evaluation_path, header=False, mode='a')
    else:
        df.to_csv(evaluation_path, header=True)

    global_stats.to_csv(stats_path, header=True)


def extract_object_candidates(model, batch, threshold=0.55, ground_truth=False):
    M, _ = batch.size()
    softmax = lambda x: np.exp(x)/sum(np.exp(x))

    examples = []

    for m in range(M):
        example = {
            'id': m,
            'cells': []
        }

        for i in range(model.S ** 2):
            i_x = i // model.S
            i_y = i % model.S

            cell = batch[m, (i * model.cell_size):(i + 1) * model.cell_size].detach().numpy()

            cell_obj = {
                'classes': softmax(cell[-model.num_classes:]),
                'index': i,
                'boxes': []
            }

            boxes = np.split(cell[:-model.num_classes], model.B)

            for j, box in enumerate(boxes):
                anchor_box_prior = model.get_box_coords(j)
                cell_coords, cell_index, class_end, class_init = model.extract_cell_coords(i_x, i_y)

                if ground_truth:
                    bw, bh, bx, by, p = box[3], box[4], box[1], box[2], box[0]
                else:
                    bw, bh, bx, by, p = get_absolute_coordinates(anchor_box_prior, cell_coords,
                                                                 box[3], box[4], box[1], box[2], box[0])

                candidate_obj = dict(p=p, x=bx, y=by, h=bh, w=bw)

                if candidate_obj['p'] > threshold:
                    cell_obj['boxes'].append(candidate_obj)

            example['cells'].append(cell_obj)

        examples.append(example)

    return examples


def process_candidates(model, batch):
    examples = []

    for index, example in enumerate(batch):
        objects = []
        classes = []

        for cell in example['cells']:
            cell_x = (cell['index'] // model.S) / model.S
            cell_y = (cell['index'] % model.S) / model.S
            cell_class = np.argmax(cell['classes'])

            for box in cell['boxes']:
                box['classes'] = cell['classes'] * box['p']
                box['class'] = cell_class
                objects.append(box)

            classes.append({
                'x': cell_x,
                'y': cell_y,
                'class': cell_class
            })

        examples.append({
            'index': index,
            'objects': objects,
            'classes': classes
        })

    return examples


def NMS(objects):
    cleaned_objects = []
    objects.sort(reverse=True, key=lambda x: x['p'])

    # While there are still objects

        # Pick the object with most p

        # From the remaining objects, get those that have an IoU greater than a threshold with the selected one


    return objects


def refine_detections(candidates):
    for candidate in candidates:
        candidate['objects'] = NMS(candidate['objects'])

    return candidates


def calculate_map(results, recall_points=11):
    results['IP'] = results.groupby('recall')['precision'].transform('max')
    results = results.sort_values(by=['IP'], ascending=False)

    precision_recall = []

    for recall_level in np.linspace(0.0, 1.0, recall_points):
        try:
            x = results[results['recall'] >= recall_level]['IP']
            prec = max(x)
        except:
            prec = 0.0

        precision_recall.append(prec)

    map = (1/recall_points) * sum(precision_recall)

    return map


def calculate_stats(detections, detections_gt, classes):
    keys = ['p', 'x', 'y', 'w', 'h']

    global_stats = []

    for i, detection in enumerate(detections):
        stats = {
            'p': [],
            'x': [],
            'y': [],
            'h': [],
            'w': [],
            'gtp': [],
            'gtx': [],
            'gty': [],
            'gth': [],
            'gtw': [],
            'classdist_max': [],
            'classdist_min': [],
            'classdist_mean': [],
            'classdist_sd': []
        }

        gt_detection = detections_gt[i]

        class_histogram = [0] * len(classes)
        gt_histogram = [0] * len(classes)

        for object in detection['objects']:
            for key in keys:
                stats[key].append(object[key])

            class_histogram[object['class']] += 1
            stats['classdist_max'].append(np.max(object['classes']))
            stats['classdist_min'].append(np.min(object['classes']))
            stats['classdist_mean'].append(np.mean(object['classes']))
            stats['classdist_sd'].append(np.std(object['classes']))

        for gt_object in gt_detection['objects']:
            for key in keys:
                stats['gt' + key].append(gt_object[key])

            gt_histogram[gt_object['class']] += 1

        image_stats = {key + "_mean": [np.mean(stats[key])] for key in keys}
        image_stats.update({key + "_sd": [np.std(stats[key])] for key in keys})
        image_stats.update({"detected_" + classes[key]: [value] for key, value in enumerate(class_histogram)})
        image_stats.update({"gtclass_" + classes[key]: [value] for key, value in enumerate(gt_histogram)})
        image_stats['classdist_max'] = max(stats['classdist_max'])
        image_stats['classdist_min'] = min(stats['classdist_min'])
        image_stats['classdist_mean'] = np.mean(stats['classdist_mean'])
        image_stats['classdist_sd'] = np.mean(stats['classdist_sd'])

        global_stats.append(pd.DataFrame(image_stats))

    return pd.concat(global_stats)


def evaluate(path, batch_number=10, batch_images=5):
    S = 7
    B = 5
    batch_size = 15
    model_name = (path.split("/")[-1]).split('.')[0]

    dataset, yolo_dataloaders, yoloset = prepare_datasets(B, S, batch_size)
    num_classes = len(dataset.classes)

    model = load_model(path, S, B, num_classes, yoloset.anchorbs)

    global_results = []
    global_stats = []
    per_class_results = [[]] * num_classes

    current_batch = 0

    for images, labels in yolo_dataloaders['eval']:
        results = model(images)

        candidates = extract_object_candidates(model, results)
        candidates_gt = extract_object_candidates(model, labels, ground_truth=True)

        detections = process_candidates(model, candidates)
        detections = refine_detections(detections)

        detections_gt = process_candidates(model, candidates_gt)

        if current_batch < batch_images:
            show_image_results(model_name, detections, images, detections_gt, dataset.classes)

        global_results.append(evaluate_model(detections, detections_gt))
        [per_class_results[class_id].append(evaluate_model(detections, detections_gt, class_id)) for class_id in range(num_classes)]
        global_stats.append(calculate_stats(detections, detections_gt, dataset.classes))

        current_batch += 1

        if current_batch % 10 == 0:
            print('Evaluating batch {}'.format(current_batch))

        if current_batch > batch_number:
            break

    global_stats = pd.concat(global_stats)
    global_results = pd.concat(global_results)
    per_class_results = [pd.concat(class_results) for class_results in per_class_results]

    map = calculate_map(global_results)
    per_class_map = [calculate_map(class_result) for class_result in per_class_results]

    save_evaluations(model_name, map, per_class_map, global_stats, dataset.classes)


