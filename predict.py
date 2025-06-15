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
from util import ToTensor, Normalize
from dataloader import SampleVideo


# 이벤트 이름 매핑
event_names = {
    0: 'Address',
    1: 'Toe-up',
    2: 'Mid-backswing (arm parallel)',
    3: 'Top',
    4: 'Mid-downswing (arm parallel)',
    5: 'Impact',
    6: 'Mid-follow-through (shaft parallel)',
    7: 'Finish'
}

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

