import cv2
import numpy as np
import torch
import math
import argparse

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from model.MobileNetV4 import EventDetector_mb4  # 학습된 모델 클래스


class SampleVideo(Dataset):
    def __init__(self, path, input_size=160, transform=None):
        self.path = path
        self.input_size = input_size
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.path)
        frame_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
        ratio = self.input_size / max(frame_size)
        new_size = tuple([int(x * ratio) for x in frame_size])
        delta_w = self.input_size - new_size[1]
        delta_h = self.input_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # preprocess and return frames
        images = []
        for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, img = cap.read()
            resized = cv2.resize(img, (new_size[1], new_size[0]))
            b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                       value=[0.406 * 255, 0.456 * 255, 0.485 * 255])  # ImageNet means (BGR)

            b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
            images.append(b_img_rgb)
        cap.release()
        # labels = np.zeros(len(images)) # only for compatibility with transforms
        sample = {'images': np.asarray(images)}
        if self.transform:
            sample = self.transform(sample)
        return sample
        

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        images= sample['images']
        images = images.transpose((0, 3, 1, 2))
        return {'images': torch.from_numpy(images).float().div(255.)}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        images = sample['images']
        images.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        return {'images': images}


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def calculate_angle( a, b, c):
        radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
        angle = math.degrees(radians)
        angle = abs(angle)
        if angle > 180:
            angle = 360 - angle
        return angle

# Calculates the angle between two points and one of the axes
def calculate_angle2( x1, y1, x2, y2, axis='x', orientation='right'):
    if (math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * x1) != 0:
        if axis == 'x':
            theta = math.acos((x2 - x1) * (-x1) / (math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * x1))
        elif axis == 'y':
            theta = math.acos((y2 - y1) * (-y1) / (math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
        else:
            raise ValueError("Invalid axis, use 'x' or 'y'")

        if orientation == 'right':
            angle = int(180 / math.pi) * theta
        elif orientation == 'left':
            angle = 180 - int(180 / math.pi) * theta
        else:
            raise ValueError("Invalid orientation, use 'left' or 'right'")
    else:
        return 0

    return angle

# Calculates the midpoint between two points
def middle_point(a, b):
    midpoint_x = (a.x + b.x) / 2
    midpoint_y = (a.y + b.y) / 2
    return midpoint_x, midpoint_y

