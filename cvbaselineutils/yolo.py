import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from time import gmtime, strftime

import cvbaselineutils.config as conf
from cvdatasetutils.dnnutils import load_pretrained
from cvdatasetutils.pascalvoc import PascalVOCOR

from cvbaselineutils.yolodataset import YoloTrainingDataset
from mltrainingtools.cmdlogging import section_logger

from scipy.special import expit

import numpy as np
import copy
import os
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches


EPSILON = 1e-9


def segment_IoU(x, w, x_h, w_h):
    if x < x_h:
        return ordered_IoU(x, w, x_h, w_h)
    else:
        return ordered_IoU(x_h, w_h, x, w)


def ordered_IoU(x, w, x_h, w_h):
    if (x_h > x + w) or (x_h + w_h < x):
        return 0
    else:
        return min(x + w, x_h + w_h) - max(x, x_h)


def IoU(x, y, h, w, x_h, y_h, h_h, w_h):
    x_inter = segment_IoU(x, w, x_h, w_h)
    y_inter = segment_IoU(y, h, y_h, h_h)

    intersection = x_inter * y_inter
    union = (h * w) + (h_h * w_h) - intersection

    return intersection/union


def get_absolute_coordinates(anchor_box_prior, cell_coords, th, tp, tw, tx, ty):
    p = expit(tp)
    bx = expit(tx) + cell_coords[0]
    by = expit(ty) + cell_coords[1]
    bh = anchor_box_prior[0] * np.exp(th)
    bw = anchor_box_prior[1] * np.exp(tw)
    return bh, bw, bx, by, p


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
        self.detection = nn.Linear(self.middle_layer_size, S * S * self.cell_size)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.detection(x)
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

        bh, bw, bx, by, p = get_absolute_coordinates(anchor_box_prior, cell_coords, th, tp, tw, tx, ty)

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


def create_lr_policy(milestones):
    multipliers = [1, 10, 1, 0.1]

    def policy(epoch):
        for i, val in enumerate(milestones):
            if epoch < val and i < len(multipliers):
                return multipliers[i]

        return multipliers[-1]

    return policy


def execute_single_model(name, output_path, dataloaders, S, B, num_classes, anchors, lr=0.001, lambda_coord=0.5,
                         lambda_noobj=0.5, epoch_blocks=conf.EPOCH_BLOCKS, num_epochs=conf.NUM_EPOCHS):
    model = Yolo(S, B, num_classes, lambda_noobj, lambda_coord, anchors)
    model.cuda()
    model_parameters = model.parameters()

    optimizer = optim.SGD(params=model_parameters, lr=lr, momentum=0.9, weight_decay=0.001)
    scheduler = LambdaLR(optimizer, lr_lambda=create_lr_policy([10, 50, 80]))

    loss_criterion = model.create_loss_function()

    device = torch.device("cuda:" + conf.CUDA_DEVICE if torch.cuda.is_available() else "cpu")

    epoch_loss_history = {
        "eval": [],
        "train": []
    }

    model = train_model(model, name, output_path, lr, epoch_blocks, num_epochs, dataloaders, optimizer, scheduler,
                        loss_criterion, device, epoch_loss_history)


def execute():
    S = 7
    B = 5
    batch_size = conf.BATCH_SIZE
    dataset, yolo_dataloaders, yoloset = prepare_datasets(B, S, batch_size)

    name = strftime("%Y%m%d%H%M", gmtime())

    execute_single_model(name, conf.DATA_FOLDER, yolo_dataloaders, S, B, len(dataset.classes), yoloset.anchorbs)


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

        # Pain the class map for the image


def evaluate_model(candidates, labels):
    return []


def show_evaluations(global_results):
    return []


def extract_object_candidates(model, batch, threshold=0.51, ground_truth=False):
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
                    bh, bw, bx, by, p = box[3], box[4], box[1], box[2], box[0]
                else:
                    bh, bw, bx, by, p = get_absolute_coordinates(anchor_box_prior, cell_coords,
                                                                 box[3], box[0], box[4], box[1], box[2])

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


def evaluate(path, batch_number=1):
    S = 7
    B = 5
    batch_size = 10
    model_name = (path.split("/")[-1]).split('.')[0]

    dataset, yolo_dataloaders, yoloset = prepare_datasets(B, S, batch_size)
    num_classes = len(dataset.classes)

    model = load_model(path, S, B, num_classes, yoloset.anchorbs)

    global_results = []
    current_batch = 0

    for images, labels in yolo_dataloaders['eval']:
        results = model(images)

        candidates = extract_object_candidates(model, results)
        candidates_gt = extract_object_candidates(model, labels, ground_truth=True)
        detections = process_candidates(model, candidates)
        detections_gt = process_candidates(model, candidates_gt)

        show_image_results(model_name, detections, images, detections_gt, dataset.classes)

        evaluation_results = evaluate_model(candidates, labels)

        global_results.append(evaluation_results)

        current_batch += 1

        if current_batch > batch_number:
            break

    show_evaluations(global_results)

    
if __name__== "__main__":
    evaluate("./data/201908200516.pt")
    #execute()
