import os
from pydub import AudioSegment
from pathlib import Path
import speech_recognition as sr
import time
import cv2
import json
import pandas as pd
os.environ["PATH"] += os.pathsep + '/opt/homebrew/bin'


def wrap_text(text, width):
    words = text.split()
    wrapped_lines = []
    current_line = ""

    for word in words:
        if cv2.getTextSize(current_line + word, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0] <= width:
            current_line += word + " "
        else:
            wrapped_lines.append(current_line.strip())
            current_line = word + " "

    if current_line:
        wrapped_lines.append(current_line.strip())

    return wrapped_lines


def display_text_with_wrapping(frame, text, x, y, font_scale, color, thickness, width, h_padding=1.7):
    lines = wrap_text(text, width)
    y_offset = 0

    for line in lines:
        (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        h = round(h*h_padding)
        cv2.putText(frame, line, (x, y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        y_offset += h

    return frame


def video_caption(video_path, start_time=0, end_time=30):
    mp3_path = video_path.replace('.mp4', '.mp3')
    audio = AudioSegment.from_file(video_path)  # convert to mp3
    audio.export(mp3_path, format='mp3')

    output_wav_path = Path(mp3_path).with_suffix(".wav")

    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(output_wav_path, format="wav")

    input_wav_path = str(output_wav_path)

    os.system(f"""ffmpeg -i '{mp3_path}' -acodec pcm_s16le -ar 16000 {input_wav_path}""")

    # loading audio file using SpeechRecognition
    r = sr.Recognizer()
    with sr.AudioFile(input_wav_path) as source:
        audio = r.record(source, offset=start_time, duration=end_time-start_time)

    try:
        result = r.recognize_whisper(audio, model="base", show_dict=True, load_options={}, no_speech_threshold=0.2)
        text = result["text"]
        print("Detected speech:", text)
    except sr.UnknownValueError:
        print("Error")

    # video captioning
    cap = cv2.VideoCapture(video_path)  # opens the video
    fps = cap.get(cv2.CAP_PROP_FPS)  # frames per second
    current_frame = 0
    output_path = video_path.replace('.mp4', ' captioned.mp4')

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # saves transcript as csv
    data = {
        'start': [segment['start'] for segment in result['segments']],
        'end': [segment['end'] for segment in result['segments']],
        'text': [segment['text'] for segment in result['segments']]
    }

    df = pd.DataFrame(data)
    with open(video_path.replace('.mp4', ' transcript.json'), "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    result_csv_path = video_path.replace('.mp4', ' transcript.csv')
    df.to_csv(result_csv_path, index=False)

    # VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Codec for MP4 file
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()  # infinite loop through all the frames
        current_frame += 1
        if not ret:
            break

    # Find the appropriate caption for the current frame
        caption = None
        for segment in result['segments']:
            start_time = segment["start"]
            end_time = segment["end"]
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            if start_frame <= current_frame <= end_frame:
                caption = segment["text"]
                break

        if caption is not None:
            shadow = display_text_with_wrapping(frame, caption, x=52, y=52, font_scale=1, color=(0, 0, 0), thickness=5,
                                                width=frame.shape[1] - 100, h_padding=1.6)
            frame_with_wrapped = display_text_with_wrapping(frame, caption, x=50, y=50, font_scale=1, color=(255, 255, 255),
                                                            thickness=2, width=frame.shape[1] - 100)
            cv2.imwrite('test_image.jpg', frame_with_wrapped)
            cv2.imwrite('test_image.jpg', shadow)

        output_video.write(frame)
    # release the cap object
    cap.release()
    output_video.release()
    # close all windows
    cv2.destroyAllWindows()

    # get audio
    output_video_path = output_path.replace('.mp4', ' with audio.mp4')

    os.system(f'''ffmpeg -y -i "{output_path}" -i "{video_path}" -c:v copy -c:a copy -shortest "{output_video_path}"''')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    t0 = time.time()
    video_caption('Data (for testing)/Tim Urban_ Inside the mind of a master procrastinator _ TED trimmed.mp4')
    print(f'Time used (seconds): {round(time.time() - t0, 2)}')
