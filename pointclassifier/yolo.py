from slam3r.utils.recon_utils import estimate_intrinsics
from ultralytics import YOLO
import numpy as np
import torch

class YOLOPointClassifier():

    def __init__(self,weights_path):
        self.model = YOLO(weights_path)
        self.yolo_results = []
        self.global_point_classes = {}

    def predict_view(self, image):
        print(f"Image shape no predict {image.shape}")
        results = self.model.predict(source=image, save=False, save_txt=False)
        self.yolo_results.append(results)

    def classify_points(self, registred_pcds):

        point_classes = np.full((registred_pcds.shape[0],), -1, dtype=int)

        for i in range(len(self.yolo_results)): 
            yolo_result = self.yolo_results[i]

            N, XYZ = registred_pcds.shape
            H, W = yolo_result[0].orig_shape[:2]
            if N != H * W:
                raise ValueError(f"Mismatch: Expected {H}*{W}={H*W} points, but got {N}")
            

            print(N, H*W)
            
            if yolo_result[0].boxes:

                for box, mask in zip(yolo_result[0].boxes, yolo_result[0].masks.data):
                    class_id = int(box.cls)

                    if mask is None:
                        x_min, y_min, x_max, y_max = map(int, box.xyxy.cpu().numpy())
                        mask = np.zeros((registred_pcds.shape[0], registred_pcds.shape[1]), dtype=bool)
                        mask[y_min:y_max, x_min:x_max] = True
                    else:
                        mask = mask.cpu().numpy()

                mask_flat = mask.flatten().astype(bool)

                for idx in np.where(mask_flat)[0]:
                    point = tuple(registred_pcds[idx])
                    if point in self.global_point_classes:
                        point_classes[idx] = self.global_point_classes[point]
                    else:
                        point_classes[idx] = class_id
                        self.global_point_classes[point] = class_id

        for idx, point in enumerate(registred_pcds):
            if tuple(point) in self.global_point_classes:
                point_classes[idx] = self.global_point_classes[tuple(point)]

        return point_classes

    def classify_points_reprojection(self, registred_pcds):
        """
        Classifica pontos 3D usando YOLO e propaga as classificações entre views.
        """
        H, W = self.yolo_results[0][0].orig_shape[:2]
        N, XYZ = registred_pcds.shape

        if N != H * W:
            raise ValueError(f"Expected {H}*{W}={H*W} points, but got {N}")

        intrinsics = estimate_intrinsics(torch.tensor(registred_pcds.reshape(1, H, W, XYZ)))
        points_3d = registred_pcds.T
        points_2d_h = np.dot(intrinsics, points_3d)
        points_2d = points_2d_h[:2] / points_2d_h[2]
        points_2d = points_2d.T.astype(int)

        valid_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H)
        points_2d = points_2d[valid_mask]
        registred_pcds = registred_pcds[valid_mask]

        point_classes = np.zeros((registred_pcds.shape[0],), dtype=int)

        for i in range(len(self.yolo_results)):
            yolo_result = self.yolo_results[i]

            if yolo_result[0].masks:
                masks = yolo_result[0].masks.data.cpu().numpy()
                for idx, mask in enumerate(masks):
                    class_id = yolo_result[0].boxes.cls[idx].item()

                    mask_indices = np.argwhere(mask)
                    mask_set = set(map(tuple, mask_indices))

                    mask_points = np.array([(p[1], p[0]) in mask_set for p in points_2d])
                    point_classes[mask_points] = class_id

                    for idx, point in enumerate(registred_pcds[mask_points]):
                        self.global_point_classes[tuple(point)] = class_id

        for idx, point in enumerate(registred_pcds):
            if tuple(point) in self.global_point_classes:
                point_classes[idx] = self.global_point_classes[tuple(point)]

        return point_classes