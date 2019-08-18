import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from time import gmtime, strftime

import cvdatasetutils.config as conf
from cvdatasetutils.dnnutils import load_pretrained
from cvdatasetutils.pascalvoc import PascalVOCOR

from cvbaselineutils.yolodataset import YoloTrainingDataset
from mltrainingtools.cmdlogging import section_logger

import numpy as np
import copy
import os
import pandas as pd


EPSILON = 1e-9


def segment_IoU(x, w, x_h, w_h):
    if x < x_h:
        return ordered_IoU(x, w, x_h, w_h)
    else:
        return ordered_IoU(x_h, w_h, x, w)


def ordered_IoU(x, w, x_h, w_h):
    if (x_h < x + w) or (x_h + w_h < x):
        return 0
    else:
        return min(x + w, x_h + w_h) - max(x, x_h)


def IoU(x, y, h, w, x_h, y_h, h_h, w_h):
    x_inter = segment_IoU(x, w, x_h, w_h)
    y_inter = segment_IoU(y, h, y_h, h_h)

    intersection = x_inter * y_inter
    union = (h * w) + (h_h * w_h) - intersection

    return intersection/union


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

    def get_anchor_values(self, anchor, anchor_box_index):
        p = anchor[anchor_box_index]
        x = max(EPSILON, anchor[anchor_box_index + 1].cpu().detach().numpy().item())
        y = max(EPSILON, anchor[anchor_box_index + 2].cpu().detach().numpy().item())
        h = max(EPSILON, anchor[anchor_box_index + 3].cpu().detach().numpy().item())
        w = max(EPSILON, anchor[anchor_box_index + 4].cpu().detach().numpy().item())

        return p, x, y, h, w

    def create_loss_function(self):
        def loss_function(output, target):
            accumulator = 0
            M, _ = target.size()

            for i_x in range(self.S):
                for i_y in range(self.S):
                    cell_index = ((i_x * self.S) + i_y) * self.cell_size

                    class_init = cell_index + self.B * self.anchor_size
                    class_end = class_init + self.num_classes

                    c = target[:, class_init:class_end]
                    c_h = output[:, class_init:class_end]

                    class_agreement = torch.sum((c - c_h) ** 2)

                    for e in range(M):
                        for j in range(self.B):
                            anchor_box_index = cell_index + j * self.anchor_size

                            p, x, y, h, w = self.get_anchor_values(target[e], anchor_box_index)
                            p_h, x_h, y_h, h_h, w_h = self.get_anchor_values(output[e], anchor_box_index)

                            objectness = 1 - IoU(x, y, h, w, x_h, y_h, h_h, w_h)

                            box_agreement = self.lambda_coord * (
                                (x - x_h) ** 2 + (y - y_h) ** 2 +
                                (w - w_h) ** 2 + (w - w_h) ** 2 + objectness)

                            accumulator += self.responsible_anchor(i_x, i_y, j, x, y, w, h) * box_agreement + \
                                           self.lambda_noobj * self.anchor_without_objects(i_x, i_y, j, x, y, w, h) * objectness

                            accumulator += self.responsible_cell(i_x, i_y, x, y) * class_agreement

                    accumulator /= M

            return accumulator

        return loss_function


def train_model(model, num_epochs, dataloaders, optimizer, loss_criterion, device, epoch_loss_history,
                epoch_size=conf.STEPS_PER_EPOCH, batch_log_freq=conf.BATCH_LOG_FREQUENCY, skip_eval=False):
    log = section_logger(1)
    step = section_logger(2)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_eae = float("inf")

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

    log('Finishing epochs block')
    model.load_state_dict(best_model_wts)
    return model


def save_results(name, output_path, model, epoch_log_history, lr):
    loss_df = pd.DataFrame(epoch_log_history)
    loss_df.loc[:, 'name'] = name
    loss_df.loc[:, 'lr'] = lr
    loss_df.to_csv(os.path.join(output_path, name + '.csv'))
    torch.save(model.state_dict(), os.path.join(output_path, name + '.pt'))


def execute_single_model(name, output_path, dataloaders, S, B, num_classes, anchors, lr=0.0005, lambda_coord=0.5,
                         lambda_noobj=0.5, epoch_blocks=10, num_epochs=10):
    model = Yolo(S, B, num_classes, lambda_noobj, lambda_coord, anchors)
    model.cuda()
    model_parameters = model.parameters()

    optimizer = optim.Adam(model_parameters, lr=lr)
    loss_criterion = model.create_loss_function()

    device = torch.device("cuda:" + conf.CUDA_DEVICE if torch.cuda.is_available() else "cpu")

    epoch_loss_history = {
        "eval": [],
        "train": []
    }

    for e in range(epoch_blocks):
        model = train_model(model, num_epochs, dataloaders, optimizer, loss_criterion, device, epoch_loss_history)

        save_results(name, output_path, model, epoch_loss_history, lr)


def execute():
    S = 9
    B = 5
    batch_size = 15
    dataset, yolo_dataloaders, yoloset = prepare_datasets(B, S, batch_size)

    name = strftime("%Y%m%d%H%M", gmtime())

    execute_single_model(name, conf.VG_DATA, yolo_dataloaders, S, B, len(dataset.classes), yoloset.anchorbs)


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


def show_results(detections):
    return None


def extract_object_candidates(model, batch, threshold=0.5):
    M, _ = batch.size()
    softmax = lambda x: np.exp(x)/sum(np.exp(x))

    examples = []

    for m in range(M):
        example = {
            'id': m,
            'cells': []
        }

        for i in range(model.S ** 2):
            cell = batch[m, (i * model.cell_size):(i + 1) * model.cell_size].detach().numpy()

            cell_obj = {
                'classes': softmax(cell[-model.num_classes:]),
                'index': i,
                'boxes': []
            }

            boxes = np.split(cell[:-model.num_classes], model.B)

            for box in boxes:
                candidate_obj = dict(p=box[0], x=box[1], y=box[2], h=box[3], w=box[4])

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
            cell_x = (cell['index'] // model.S) * model.cell_size
            cell_y = (cell['index'] % model.S) * model.cell_size
            cell_class = np.argmax(cell['classes'])

            for box in cell['boxes']:
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


def evaluate(path):
    S = 9
    B = 5
    batch_size = 20

    dataset, yolo_dataloaders, yoloset = prepare_datasets(B, S, batch_size)
    num_classes = len(dataset.classes)

    model = load_model(path, S, B, num_classes, yoloset.anchorbs)

    for image, labels in yolo_dataloaders['eval']:
        results = model(image)

        candidates = extract_object_candidates(model, results)
        detections = process_candidates(model, candidates)

        show_results(detections)


if __name__== "__main__":
    evaluate("../data/201908161720.pt")
    #execute()
