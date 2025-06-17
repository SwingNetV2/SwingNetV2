import os
import pandas as pd
import cv2
import albumentations as A


df = pd.read_pickle('golfDB.pkl')
yt_video_dir = '../../database/videos/'

# Albumentations transform 
transform_video_frames = A.Compose([
    A.Resize(height=256, width=256),
    A.RandomCrop(height=224, width=224),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(p=0.2),
])

def preprocess_videos(anno_id, dim=160):
    a = df.loc[df['id'] == anno_id]
    bbox = a['bbox'].iloc[0]  
    events = a['events'].iloc[0]  
    path = 'videos_{}/'.format(dim)

    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.isfile(os.path.join(path, "{}.mp4".format(anno_id))):
        print('Processing annotation id {}'.format(anno_id))

        # 입력 비디오 열기
        cap = cv2.VideoCapture(os.path.join(yt_video_dir, '{}.mp4'.format(a['youtube_id'].iloc[0])))

        # 출력 비디오 설정
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(os.path.join(path, "{}.mp4".format(anno_id)),
                              fourcc, cap.get(cv2.CAP_PROP_FPS), (dim, dim))

        # Bounding box 좌표 계산
        x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * bbox[0])
        y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * bbox[1])
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * bbox[2])
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * bbox[3])

        count = 0
        success, image = cap.read()

        while success:
            count += 1
            if count >= events[0] and count <= events[-1]:
                # Bounding box로 crop
                crop_img = image[y:y + h, x:x + w]

                # BGR → RGB 변환 (Albumentations용)
                crop_img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

                # Albumentations transform 적용
                transformed = transform_video_frames(image=crop_img_rgb)
                processed_img = transformed['image']

                # RGB → BGR 변환 (VideoWriter용)
                processed_img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)

                # 최종 크기 조정 
                final_img = cv2.resize(processed_img_bgr, (dim, dim))

                out.write(final_img)

            if count > events[-1]:
                break
            success, image = cap.read()

        cap.release()
        out.release()
        print('Completed processing annotation id {}'.format(anno_id))

    else:
        print('Annotation id {} already completed for size {}'.format(anno_id, dim))


if __name__ == "__main__":
    preprocess_videos(anno_id=1, dim=160)
