import numpy as np
import cA2v

def segment_image(im):
    # implement saliency
    # call grabcut implementation
    mask = np.zeros(im.shape, np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    cv2.grabCut(im, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK) 

def segment_video(frames):
    masks = []
    for frame in frames:
        np.append(segment_image(frame))
    
    # implement smoothing
    SMOOTH_FACTOR = 3
    final_masks = []
    for cframe in arange(len(frames)):
        curr_mask = np.zeros(frames[0].shape)
        diff = (SMOOTH_FACTOR - 1) * 0.5
        num = 0
        for idx in arange(cframe - diff, cframe + diff):
            if idx >= 0 and idx < len(frames):
                num += 1
                curr_mask += masks[idx]
        final_masks.append(curr_mask / num)

    smoothed_masks = [frames[i] + frames[i + 1] + frames[i + 2]) for i in 

if __name__ == '__main__':
    img = cv2.imread('lol.jpg')
    mask = segment_image(im)
