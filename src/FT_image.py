from multiprocessing import Pool
from prepare_data import *
def FT(path):
    for path_iter in path:
        img = cv2.imread(path_iter,0)
        img = cv2.resize(img, (256,256))
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = (20*np.log(np.abs(fshift))).astype("uint8")
        # cv2.imshow("img", magnitude_spectrum)
        cv2.imwrite(path_iter[:-4]+"FT.jpg", magnitude_spectrum)
if __name__ == "__main__":
    root = "/home/longnguyen/data/CelebA_Spoof_/CelebA_Spoof"
    path, value = getPathValue()
    path = [os.path.join(root, path_iter) for path_iter in path ]
    split = 12
    split_data = [path[i::split] for i in range(split)]
    p = Pool(split)
    results = p.map(FT, split_data)

