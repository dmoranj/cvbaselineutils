from torch.utils.data import Dataset
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from torchvision import transforms
from cvdatasetutils.pascalvoc import PascalVOCOR

YOLO_SIZE=(448, 448)
BBOX_COORDS = 4


class YoloTrainingDataset(Dataset):
    """
    Dataset for training YOLO v1 models. This dataset's labels are calculated based on
    the distribution of anchor boxes for a YOLO model. The anchor boxes are calculated with clustering
    algorithms based on the bounding boxes found in the given dataset.


    """
    def __init__(self, dataset, S, B):
        self.dataset = dataset
        self.S = S
        self.B = B
        self.cell_size = 1/S
        self.anchorbs = generate_anchor_boxes(self.dataset.get_raw(), B)
        self.num_classes = len(dataset.get_class_list())
        self.anchor_size = BBOX_COORDS + 1
        self.cell_length = self.B * self.anchor_size + self.num_classes
        self.label_size = self.S**2 * self.cell_length
        self.transform = transforms.Compose([
            transforms.Resize(YOLO_SIZE),
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        image, annotations = self.dataset.__getitem__(idx)

        image = self.transform(image)

        labels = np.zeros(self.label_size)

        for obj in annotations['objects']:
            iou_scores = np.zeros((self.S, self.S, self.B))

            for i in range(self.S):
                for j in range(self.S):
                    for p in range(self.B):
                        anchor = {
                            'bx': (i + 0.5) * self.cell_size,
                            'by': (j + 0.5) * self.cell_size,
                            'h': self.anchorbs[p][0],
                            'w': self.anchorbs[p][1]
                        }

                        iou_scores[i, j, p] = IoU(obj, anchor)

            grid_x, grid_y, anchor_id = np.unravel_index(np.argmax(iou_scores), iou_scores.shape)

            label_index = ((grid_x * self.S) + grid_y) * self.cell_length + anchor_id * self.anchor_size
            labels[label_index] = 1.0
            labels[label_index + 1] = obj['bx']
            labels[label_index + 2] = obj['by']
            labels[label_index + 3] = obj['h']
            labels[label_index + 4] = obj['w']

            class_index = self.dataset.get_class_list().index(obj['class'])

            label_classes_index = ((grid_x * self.S) + grid_y) * self.cell_length + self.B * self.anchor_size

            labels[label_classes_index + class_index] = 1.0

        return image, labels.astype(np.float32)


def get_intersection(obj1, obj2, pos, size):
    order = lambda o1, o2, pos, size: (o1, o2) if o1[pos] - o1[size]/2 < o2[pos] - o2[size]/2 else (o2, o1)

    obj1, obj2 = order(obj1, obj2, pos, size)
    length = min((obj1[pos] + obj1[size]/2), (obj2[pos] + obj1[size]/2)) - (obj2[pos] - obj2[size]/2)

    return max(length, 0)


def IoU(obj1, obj2):
    area = lambda o: o['h']*o['w']

    intersection_w = get_intersection(obj1, obj2, 'bx', 'w')
    intersection_h = get_intersection(obj1, obj2, 'by', 'h')

    intersection = intersection_h * intersection_w

    union = max(area(obj1) - area(obj2) - intersection, 1)

    return intersection/union


def generate_anchor_boxes(dataset, B):
    boxes = [{
        'h': obj['h'],
        'w': obj['w']
    } for image in dataset for obj in image['objects']]

    box_data = pd.DataFrame(boxes)
    clusterer = KMeans(B)
    clusterer.fit(box_data)

    return clusterer.cluster_centers_



if __name__ == "__main__":
    dataset = PascalVOCOR("/home/dani/Documentos/Proyectos/Doctorado/Datasets/VOC2012/VOCdevkit/VOC2012")

    yoloset = YoloTrainingDataset(dataset, 9, 5)

    for image, labels in yoloset:
        print("Yolo")

