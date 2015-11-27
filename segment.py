import numpy as np
import cA2v

"""
Based on Goferman et al's Context-AWare Saliency Detection (2011).
"""
def get_saliency(im):
    # convert im to CIELAB color space
    im_cie     = cv2.cvtColor(im, CV_RGB2Lab)
    K          = 64
    PATCH_SIZE = 7
    offset     = (PATCH_SIZE - 1) / 2
    numel      = np.zeros(im.shape[0], im.shape[1])
    maxdim = max(im.shape[0], im.shape[1])
    C = 3

    dcolor     = np.zeros(numel)
    dpos       = np.zeros(numel)
    saliency   = np.zeros(im.shape)

    # calculate initial saliency
    sub2ind    = np.unravel_index(np.arange(ds.shape[0]), (im.shape[0], im.shape[1]))
    for (i, j) in np.ndenumerate(ds):
        ri,ci = sub2ind[i]
        rj,cj = sub2ind[j]
        if ri - offset < 0 or ri + offset >= im.shape[0] or \
           rj - offset < 0 or rj + offset >= im.shape[0] or \
           ci - offset < 0 or ci + offset >= im.shape[1] or \
           cj - offset < 0 or cj + offset >= im.shape[1]:
            continue
        patchi = im_cie[(ri-offset):(ri+offset), (ci-offset):(ci+offset), :]
        patchj = im_cie[(rj-offset):(rj+offset), (cj-offset):(cj+offset), :]
        dcolor[i,j] = np.linalg.norm(patchi, patchj)
        dpos[i,j] = ((ri - rj)^2 + (ci - cj)^2)**0.5 / maxdim

    dcolor /= np.max(np.max(dcolor))
    ds = dcolor / (1 + C * dpos)
    for idx in range(len(ds)):
        imidx = sub2ind[idx]
        saliency[imidx] = 1 - np.exp(-np.mean(np.sort(ds[idx])[:K]))

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


def segment_image(im):
    im = get_saliency(im)
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
