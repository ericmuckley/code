from glob import glob
import cv2

def create_video(image_list, video_name, fps=8, reverse=False):
    # create video out of images saved in a folder
    # frames per second (fps) and order of the images can be reversed 
    # using the **kwargs.
    if reverse: image_list = image_list[::-1]
    frame = cv2.imread(image_list[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, -1, fps, (width,height))
    for image in image_list:
        video.write(cv2.imread(image))
    cv2.destroyAllWindows()
    video.release()
    return video





image_list = sorted(glob(r'C:\Users\a6q\voigt_surf_plots\/*.jpg'))
#image_list = [i for i in image_list if 'error' in i]


video = create_video(image_list,
                     r'C:\Users\a6q\Desktop\voigt_surf_plots100.avi',                     
                     fps=6)
