from glob import glob
import cv2

def create_video(imagelist, video_name='vid.avi', fps=8, reverse=False):
    """
    Create video out of a list of images saved in a folder.
    Specify name of the video in 'video_name'.
    Frames per second (fps) and order of the images can be reversed.
    """
    imagelist = sorted(imagelist)
    if reverse:
        imagelist = imagelist[::-1]
    # get size of the video frame
    frame = cv2.imread(imagelist[0])
    height, width, layers = frame.shape
    # initiate video
    video = cv2.VideoWriter(video_name, -1, fps, (width,height))
    for i, img in enumerate(imagelist):
        print('writing frame %i / %i' %(i+1, len(imagelist)))
        img = cv2.imread(img)
        video.write(img)
    cv2.destroyAllWindows()
    video.release()
    return video


imagelist = glob(r'C:\Users\a6q\exp_data\spline_predictions\/*.jpg')
#image_list = [i for i in image_list if 'error' in i]


video = create_video(
        imagelist,
        fps=5,
        video_name = r'C:\Users\a6q\Desktop\spline_predictions.avi')
