from moviepy.editor import VideoClip, ImageSequenceClip, CompositeVideoClip, TextClip, concatenate_videoclips, \
    ImageClip, clips_array
from glob import glob
from PIL import Image
import numpy as np
from moviepy.editor import VideoFileClip


def from_folder(folder, file_name):
    frames = []
    duration_pre_image = 0.1
    drive_records = glob(folder)
    total = len(drive_records)
    for index in range(0, total, 2):
        print("working {}/{}".format(index + 1, total))
        record = drive_records[index]
        image = np.array(Image.open(record))
        text = TextClip(txt="steps_{}".format(basename(record)),
                        method="caption", align="North",
                        color="white", stroke_width=3, fontsize=18)
        text = text.set_duration(duration_pre_image)

        center_image_clip = ImageClip(image, duration=duration_pre_image)
        all_images_clip = clips_array([[center_image_clip]])
        frames.append(CompositeVideoClip([
            all_images_clip,
            text
        ]))
    final = concatenate_videoclips(frames, method="compose")
    final.write_videofile(file_name, fps=10)


def remove_mp4_extension(file_name):
    return file_name.replace(".mp4", "")

global_image_index = 0


def save_video_as_images():
    def save_to_file(image_array):
        image = Image.fromarray(image_array)
        global global_image_index
        image.save('./videos/original/{}.png'.format(global_image_index), 'PNG')
        global_image_index = global_image_index + 1
        return np.array(image)

    global_image_index = 0
    video_file = './videos/back_home_fast.mp4'
    clip = VideoFileClip(video_file, audio=False)
    t_start = 0
    t_end = 5
    if t_end > 0.0:
        clip = clip.subclip(t_start=t_start, t_end=t_end)
    else:
        clip = clip.subclip(t_start=t_start)

    clip = clip.fl_image(save_to_file)
    clip.write_videofile("{}_test.mp4".format(remove_mp4_extension(video_file)), audio=False)


if __name__ == '__main__':
    save_video_as_images()
