import cv2
import numpy as np

image_hsv = None   # global ;(
pixel = (20,60,80) # some stupid default
data =[]
# mouse callback function
def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y,x]
        data.append(pixel)
        #you might want to adjust the ranges(+-10, etc):
        upper =  np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
        lower =  np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])
        #print(pixel, lower, upper)
        data_np = np.array(data)
        print("min", np.min(data_np, axis=0))
        print("max", np.max(data_np, axis=0))

def main():
    import sys
    train = 1
    if len(sys.argv) > 1:
        if "test" in sys.argv[1]:
            train = 0

    global image_hsv, pixel # so we can use it in mouse callback
    image_hsv = np.load("data/hsv_sample2.npy")
    image_bgr = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    if train: 
        ## NEW ##
        cv2.namedWindow('hsv')
        cv2.setMouseCallback('hsv', pick_color)

        # now click into the hsv img , and look at values:
        #image_hsv = cv2.cvtColor(image_src,cv2.COLOR_BGR2HSV)
        cv2.imshow("hsv",image_hsv)

    else:
        lower = np.array([114,107,93])
        upper = np.array([121, 175, 119])
        image_mask = cv2.inRange(image_hsv,lower,upper)
        inv_mask = np.bitwise_not(image_mask)
        res = cv2.bitwise_and(image_hsv,image_hsv,mask = inv_mask)
        from PIL import Image
        invert_img = Image.fromarray(res)
        invert_img.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
