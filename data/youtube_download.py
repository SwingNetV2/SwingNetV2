import os
from yt_dlp import YoutubeDL

def download_with_ytdlp(
    url: str,
    output_dir: str = "/content/drive/MyDrive/youtube_download")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ydl_opts = {
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "merge_output_format": "mp4",
    }

    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_title = info_dict.get("title", None)
        video_ext = info_dict.get("ext", "mp4")
        saved_filename = f"{video_title}.{video_ext}"
        saved_path = os.path.join(output_dir, saved_filename)
        return saved_path


if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/shorts/7bTPupYIXtM"
    try:
        print(f"[INFO] 다운로드 시작: {youtube_url}")
        saved_file = download_with_ytdlp(youtube_url)
        print(f"[완료] 파일 저장 성공: {saved_file}")
    except Exception as e:
        print(f"[ERROR] 다운로드 중 오류 발생: {e}")
