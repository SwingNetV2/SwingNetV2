import os
import os.path as osp
import argparse
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import mediapipe as mp
from predict_utils import *
from model import EventDetector_clstm_lr
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


seq_length = 64
model = EventDetector_clstm_lr(n_conv=seq_length, num_classes=9, ).cuda()
class SampleVideo(Dataset):
    def __init__(self, path, input_size=160, transform=None):
        self.path = path
        self.input_size = input_size
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise ValueError(f"Video file {self.path} could not be opened.")
        frame_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
        ratio = self.input_size / max(frame_size)
        new_size = tuple([int(x * ratio) for x in frame_size])
        delta_h = self.input_size - new_size[0]
        delta_w = self.input_size - new_size[1]
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

        sample = {'images': np.asarray(images)}
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Path to video that you want to test', default='test_video.mp4')
    parser.add_argument('-s', '--seq-length', type=int, help='Number of frames to use per forward pass', default=64)
    args = parser.parse_args()
    args, _ = parser.parse_known_args()
    seq_length = args.seq_length

    print('Preparing video: {}'.format(args.path))

    ds = SampleVideo(args.path, transform=transforms.Compose([ ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))
    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)


    save_dict = torch.load('models/best_conv_lr.pth', weights_only=True)
    model.load_state_dict(save_dict)
    model.eval()
    model.cuda()
    print("Loaded model weights")


    print('Testing...')
    for sample in dl:
        images = sample['images']
        B, total_frames, C, H, W = images.shape
        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        batch = 0
        num_batches = math.ceil(total_frames / seq_length)
        for i in range(num_batches):
            start = i * seq_length
            end   = start + seq_length
    
            if end <= total_frames:
                batch_imgs = images[:, start:end]               # [B, 64, C, H, W]
            else:
                rem  = total_frames - start                     # ex) 8
                last = images[:, start:total_frames]            # [B, rem, C, H, W]
                pad  = images[:, -1:].repeat(1, seq_length-rem, 1, 1, 1)
                batch_imgs = torch.cat([last, pad], dim=1)      # [B, 64, C, H, W]
    
            logits = model(batch_imgs.cuda())
            if i == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), axis=0)

    events = np.argmax(probs, axis=0)[:-1]
    print('Predicted event frames: {},'.format(events))
    cap = cv2.VideoCapture(args.path)

    for i, e in enumerate(events):
        cap.set(cv2.CAP_PROP_POS_FRAMES, e)
        ret, img = cap.read()
        if not ret or img is None:
            print(f"[Warning] Frame {e} could not be read, skipping.")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.save(f'output/image{e}.jpg')
    
        cv2.putText(img, f'{event_names[i]}, frame {e}', (20, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 2)
      
        image = mp.Image.create_from_file(f'output/image{e}.jpg')

        #Create an PoseLandmarker object.
        base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True)
        detector = vision.PoseLandmarker.create_from_options(options)

        #Detect pose landmarks from the input image.
        detection_result = detector.detect(image)

        landmarks = detection_result.pose_landmarks
        mp_pose = mp.solutions.pose
        image_np = image.numpy_view().copy()

        h, w = image_np.shape[:2]

        if landmarks:
            left_shoulder = landmarks[0][mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[0][mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[0][mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[0][mp_pose.PoseLandmark.RIGHT_HIP]
            right_wrist = landmarks[0][mp_pose.PoseLandmark.RIGHT_WRIST]
            left_wrist = landmarks[0][mp_pose.PoseLandmark.LEFT_WRIST]
            nose = landmarks[0][mp_pose.PoseLandmark.NOSE]
            right_knee = landmarks[0][mp_pose.PoseLandmark.RIGHT_KNEE]
            left_knee = landmarks[0][mp_pose.PoseLandmark.LEFT_KNEE]
            right_ankle = landmarks[0][mp_pose.PoseLandmark.RIGHT_ANKLE]
            left_ankle = landmarks[0][mp_pose.PoseLandmark.LEFT_ANKLE]
            left_elbow = landmarks[0][mp_pose.PoseLandmark.LEFT_ELBOW]

            knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            pelvis_angle = calculate_angle(left_ankle, left_hip, right_shoulder)
            arm_angle = calculate_angle(left_wrist, left_elbow, left_shoulder)
            shoulders_inclination = calculate_angle2(int(right_shoulder.x * w), int(right_shoulder.y * h),
                                                        int(left_shoulder.x * w), int(left_shoulder.y * h), 'x', 'left')
            hips_inclination = calculate_angle2(int(left_hip.x * w), int(left_hip.y * h), int(right_hip.x * w), int(right_hip.y * h), 'x')

            midpoint_x, midpoint_y = middle_point(right_ankle, left_ankle)

            # Display points
            cv2.circle(image_np, (int(right_shoulder.x * w), int(right_shoulder.y * h)), 2, (0, 255, 0), -1)
            cv2.circle(image_np, (int(left_shoulder.x * w), int(left_shoulder.y * h)), 2, (0, 255, 0), -1)
            cv2.circle(image_np, (int(right_hip.x * w), int(right_hip.y * h)), 2, (255, 255, 0), -1)
            cv2.circle(image_np, (int(left_hip.x * w), int(left_hip.y * h)), 2, (0, 150, 255), -1)
            cv2.circle(image_np, (int(right_knee.x * w), int(right_knee.y * h)), 2, (255, 0, 255), -1)
            cv2.circle(image_np, (int(left_knee.x * w), int(left_knee.y * h)), 2, (255, 0, 255), -1)
            cv2.circle(image_np, (int(left_ankle.x * w), int(left_ankle.y * h)), 2, (255, 0, 0), -1)
            cv2.circle(image_np, (int(left_wrist.x * w), int(left_wrist.y * h)), 2, (0, 255, 255), -1)
            cv2.circle(image_np, (int(nose.x * w), int(nose.y * h)), 2, (0, 0, 255), -1)
            cv2.circle(image_np, (int(left_elbow.x * w), int(left_elbow.y * h)), 2, (128, 0, 128), -1)
            cv2.circle(image_np, (int(right_ankle.x * w), int(right_ankle.y * h)), 2, (255, 0, 0), -1)
            cv2.circle(image_np, (int(midpoint_x * w), int(midpoint_y * h)), 2, (255, 255, 255), -1)

            # Display angle and lines on the image
                # shoulders_inclination
            cv2.line(image_np, (int(right_shoulder.x * w), int(right_shoulder.y * h)), (int(right_shoulder.x * w), int(right_shoulder.y * h)), (0, 255, 0), 1)
            cv2.line(image_np, (int(left_shoulder.x * w), int(left_shoulder.y * h)), (int(right_shoulder.x * w), int(right_shoulder.y * h)), (0, 255, 0), 1)
                # hips_inclination
            cv2.line(image_np, (int(left_hip.x * w), int(left_hip.y * h)), (int(left_hip.x * w), int(left_hip.y * h)), (255, 255, 0), 2)
            cv2.line(image_np, (int(left_hip.x * w), int(left_hip.y * h)), (int(right_hip.x * w), int(right_hip.y * h)), (255, 255, 0), 1)
                # knee_angle
            cv2.line(image_np, (int(left_hip.x * w), int(left_hip.y * h)), (int(left_knee.x * w), int(left_knee.y * h)), (255, 0, 255), 1)
            cv2.line(image_np, (int(left_knee.x * w), int(left_knee.y * h)), (int(left_ankle.x * w), int(left_ankle.y * h)), (255, 0, 255), 1)
                # pelvis_angle
            cv2.line(image_np, (int(left_hip.x * w), int(left_hip.y * h)), (int(left_ankle.x * w), int(left_ankle.y * h)), (0, 150, 255), 1)
            cv2.line(image_np, (int(left_hip.x * w), int(left_hip.y * h)), (int(right_shoulder.x * w), int(right_shoulder.y * h)), (0, 150, 255), 1)
                # arm_angle
            cv2.line(image_np, (int(left_shoulder.x * w), int(left_shoulder.y * h)), (int(left_elbow.x * w), int(left_elbow.y * h)), (128, 0, 128), 1)
            cv2.line(image_np, (int(left_elbow.x * w), int(left_elbow.y * h)), (int(left_wrist.x * w), int(left_wrist.y * h)), (128, 0, 128), 1)

            cv2.line(image_np, (int(left_ankle.x * w), int(left_ankle.y * h)), (int(left_ankle.x * w), int(left_ankle.y * h)), (255, 150, 0), 1)
            cv2.line(image_np, (int(right_ankle.x * w), int(right_ankle.y * h)), (int(right_ankle.x * w), int(right_ankle.y * h)), (255, 150, 0), 1)

            cv2.line(image_np, (int(midpoint_x * w), int(midpoint_y * h)), (int(midpoint_x * w), int(midpoint_y * h)), (255, 255, 255), 1)

            # image_np = cv2.resize(image_np, None, fx=4, fy=4) : 160*160
            cv2.putText(image_np, '{} , {}'.format(event_names[i], format(e)), (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 2)
            cv2.putText(image_np, f'Shoulders inclination: {shoulders_inclination:.2f}', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image_np, f'Hips inclination: {hips_inclination:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(image_np, f'Knee Angle: {knee_angle:.2f}', (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(image_np, f'Pelvis Angle: {pelvis_angle:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2)
            cv2.putText(image_np, f'Arm Angle: {arm_angle:.2f}', (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2)

            pil_image = Image.fromarray(image_np.astype(np.uint8))
            pil_image.save(f'output/output{e}.jpg') # Save with event frame number
            display(pil_image)
