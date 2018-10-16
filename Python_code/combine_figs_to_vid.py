# -*- coding: utf-8 -*-

def create_video(image_folder, video_name, fps=8, reverse=False):
    #create video out of images saved in a folder
    import cv2
    import os
    
    images = [img for img in os.listdir(image_folder) if img.endswith('.jpg')]
    if reverse: images = images[::-1]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, -1, fps, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()


image_folder = r'C:\Users\a6q\Desktop\lab on a QCM\2018-10-05_pedotpss_qcm_images_fast_50X_2sec_6'
video_name = 'C:\\Users\\a6q\\Desktop\\pp_swelling.avi'
fps = 2

create_video(image_folder, video_name, fps)

