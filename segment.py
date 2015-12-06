import numpy as np
import cv2
import cv
from saliency import get_saliency
import os

"""
Based on Goferman et al's Context-Aware Saliency Detection (2011).
def get_saliency(img):
    # convert im to CIELAB color space
    im         = cv2.cvtColor(img, cv.CV_RGB2Lab)
    K          = 64
    PATCH_SIZE = 7
    offset     = (PATCH_SIZE - 1) / 2
    numel      = im.shape[0] * im.shape[1]
    maxdim     = max(im.shape[0], im.shape[1])
    C          = 3

    saliency   = np.zeros((im.shape[0], im.shape[1]))
    print saliency.shape

    # calculate initial saliency
    sub2ind    = np.unravel_index(np.arange(numel), (im.shape[0], im.shape[1]))
    maxcolor   = -np.inf
    for i in range(3000,3001):
    #for i in range(numel):
        dcolor = np.zeros(numel)
        dpos = np.zeros(numel)
        ri,ci = sub2ind[0][i], sub2ind[1][i]
        for j in range(numel):
            if j % 100 == 0:
                print "at", j
            rj,cj = sub2ind[0][j], sub2ind[1][j]
            if ri - offset < 0 or ri + offset >= im.shape[0] or \
               rj - offset < 0 or rj + offset >= im.shape[0] or \
               ci - offset < 0 or ci + offset >= im.shape[1] or \
               cj - offset < 0 or cj + offset >= im.shape[1]:
                continue

            patchi = im[(ri-offset):(ri+offset+1), (ci-offset):(ci+offset+1), :]
            patchj = im[(rj-offset):(rj+offset+1), (cj-offset):(cj+offset+1), :]
            dcolor[j] = np.linalg.norm(patchi - patchj)
            dpos[j] = ((ri - rj)**2 + (ci - cj)**2)**0.5 / maxdim

            maxcolor = max(maxcolor, np.max(dcolor))

            ds = dcolor / (1 + C * dpos)
            saliency[ri,ci] = 1 - np.exp(-np.mean(np.sort(ds)[:K]))

    print saliency.shape
    print maxcolor
    saliency /= maxcolor

    # find nearest attendant
    THRESH = 0.8
    att_r, att_c = np.where(ds > THRESH)
    for r,c in np.ndenumerate(saliency):
        d = np.min([sqrt((r-att_r[idx])^2 + (c-att_c[idx])^2) for idx in range(len(att_r))])
        saliency[r,c] = d

    # weight towards center with gaussian
    SIGMA_R = im.shape[0] / 6
    SIGMA_C = im.shape[1] / 6
    rr, cc = np.meshgrid(range(im.shape[0]), range(im.shape[1]), ordering='ij')
    midr, midc = im.shape / 2
    const = 1 / (2 * np.pi * sqrt(SIGMA_R * SIGMA_C))
    gauss = np.exp(-0.5 * (xx^2 / SIGMA_R^2 + yy^2 / SIGMA_C^2))
    return const * guass * saliency
"""

def segment_image(img, forimage=True):
    RESIZE = 250
    h,w,ch = img.shape
    maxdim = max(h, w)
    newh = int(float(h) / maxdim * RESIZE)
    neww = int(float(w) / maxdim * RESIZE)
    imresize = cv2.resize(img, (neww, newh))

    mask = get_saliency(imresize)
    if forimage:
        # for images
        print "for image"
        THRESH_FG = 0.7
        THRESH_PFG = 0.5
        THRESH_PBG = 0.1
    else:
        # for videos
        print "for video"
        THRESH_FG = 0.5
        THRESH_PFG = 0.4
        THRESH_PBG = 0.4

    mask2 = np.array(np.zeros(mask.shape), dtype=np.uint8)
    mask2[mask > THRESH_PBG] = cv2.GC_PR_BGD
    mask2[mask > THRESH_PFG] = cv2.GC_PR_FGD
    mask2[mask > THRESH_FG]  = cv2.GC_FGD

    # call grabcut implementation
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    cv2.grabCut(imresize,
                mask2,
                (0,0,neww,newh),
                bgdModel,
                fgdModel,
                5,
                cv2.GC_INIT_WITH_MASK) 

    maskshow = np.where((mask2==0)|(mask2==2),0,1).astype('uint8')
    newim = imresize*maskshow[:,:,np.newaxis]

    mask2 = mask2.astype(np.float32)
    mask2[(mask2==0)|(mask2==2)] = 0.
    mask2[mask2==3] = 0.5

    newim = cv2.resize(newim, (w, h))
    mask2 = cv2.resize(mask2, (w, h))

    # grabcut with no initialization
    origmask = np.array(np.zeros(mask.shape), dtype=np.uint8)

    # call grabcut implementation
    bgdModel2 = np.zeros((1,65), np.float64)
    fgdModel2 = np.zeros((1,65), np.float64)
    BORDER = 10
    cv2.grabCut(imresize,
                origmask,
                (BORDER,BORDER,neww-2*BORDER,newh-2*BORDER),
                bgdModel2,
                fgdModel2,
                5,
                cv2.GC_INIT_WITH_RECT) 

    origmaskshow = np.where((origmask==0)|(origmask==2),0,1).astype('uint8')
    orignewim = imresize*origmaskshow[:,:,np.newaxis]

    origmask = origmask.astype(np.float32)
    origmask[(origmask==0)|(origmask==2)] = 0.
    origmask[origmask==3] = 0.5

    orignewim = cv2.resize(orignewim, (w, h))
    origmask  = cv2.resize(origmask, (w, h))

    return newim, mask2, orignewim, origmask

def segment_video(vc):
    print "LOL"
    masks = []
    frames = []
    numframes = 0
    while vc.isOpened():
        ret, frame = vc.read()
        if ret == True:
            numframes += 1
            print "on frame", numframes
            frames.append(frame)
            im, mask, _, _ = segment_image(frame)
            masks.append(mask)
        else:
            break

    vc.release()
    
    # implement smoothing
    SMOOTH_FACTOR = 3
    final_masks = []
    for cframe in np.arange(len(masks)):
        curr_mask = np.zeros(masks[0].shape)
        diff = (SMOOTH_FACTOR - 1) / 2
        num = 0
        for idx in np.arange(cframe - diff, cframe + diff + 1):
            if idx >= 0 and idx < len(masks):
                num += 1
                curr_mask += masks[idx]
        final_masks.append(curr_mask / num)

    return frames, final_masks

if __name__ == '__main__':
    # read in segmentation ground-truths
    print "ground truth extraction"
    gts = {}
    GT_PATH = 'iccv09Data/labels'
    for subdir, dirs, files in os.walk(GT_PATH):
        for file in files:
            if 'regions' in file:
                key = file.split('.')[0]
                GT_FG = 7
                gts[key] = np.loadtxt(GT_PATH + '/' + file)
                gts[key][gts[key] != GT_FG] = 0
                gts[key][gts[key] == GT_FG] = 1

    errors  = []
    errors0 = []
    count = 0
    IM_PATH = 'iccv09Data/special_images'
    for subdir, dirs, files in os.walk(IM_PATH):
        for file in files:
            print "on image", file
            img = cv2.imread(IM_PATH + '/' + file)
            newim, maskfinal, newim0, mask0final = segment_image(img)

            h, w, ch = img.shape
            key = file.split('.')[0]
            gt = gts[key]
            error = np.sum(np.abs(maskfinal - gt)) / (w * h)
            error0 = np.sum(np.abs(mask0final - gt)) / (w * h)
            print "Error:", error
            print "Error, no init:", error0

            errors.append(error)
            errors0.append(error0)

    errors = np.array(errors)
    errors0 = np.array(errors0)

    # main for videos
    # video = cv2.VideoCapture('video/ido_walk.avi')
    # masks = segment_video(video)

    print "average diff:", np.mean(errors0 - errors)
    print "sd diff:", np.std(errors0 - errors)
    print "mean, init:", np.mean(errors)
    print "mean, no init:", np.mean(errors0)
    print errors
    print errors0

