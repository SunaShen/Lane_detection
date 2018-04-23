import numpy as np  
import cv2 


kern_L = np.array(np.mat('2 3 0; 3 0 -3; 0 -3 -2'), subok=True)

kern_R = np.array(np.mat('0 -3 -2; 3 0 -3; 2 3 0'), subok=True)


if __name__ == "__main__":
    cap = cv2.VideoCapture('data_set/IMG_7563.mp4')

    if cap.isOpened() == 0 :
        print'video read failed.'
    else:
        print'video read success'

    fps = cap.get(5)  # CAP_PROP_FPS
    total_frame = cap.get(7)  # CAP_PROP_FRAME_COUNT
    print 'The fps is',fps
    print 'Total frames are',total_frame
    

    #i　= 0

    while (cv2.waitKey(500) & 0xFF != ord('q')): #ord()--return ASCII

        ret, src_img = cap.read()
        if ret == 0:
            break
        
        src_img_gray = cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)

        cv2.imshow('src_img_gray', src_img_gray)


        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

        temp_img_erode = cv2.erode(src_img_gray,element)

        cv2.imshow('temp_img_erode', temp_img_erode)

        tem_img_L = cv2.filter2D(temp_img_erode, -1, kern_L)
        tem_img_R = cv2.filter2D(temp_img_erode, -1, kern_R)

        cv2.imshow('tem_img_L', tem_img_L)
        cv2.imshow('tem_img_R', tem_img_R)

        ret1, dst_img_L = cv2.threshold(tem_img_L, 230, 255, cv2.THRESH_TOZERO)
        ret2, dst_img_R = cv2.threshold(tem_img_R, 230, 255, cv2.THRESH_TOZERO)

        cv2.imshow('dst_img_L', dst_img_L)
        cv2.imshow('dst_img_R', dst_img_R)

        res_img = dst_img_L + dst_img_R

        res_img_rows, res_img_cols = res_img.shape

        #先取ROI，再画线
        #det_img会随着下面res_img的变换而变化.....难受啊.....并不是copy？
        det_img = res_img[res_img_rows / 2: res_img_rows - 10, 0 : res_img_cols]

        cv2.imshow('det_img', det_img)

        # i = i + 1

        # print 'frame:',i
        # # if i % 30 == 0 :
        # #     filename = 'data_set/ROI' + str(i) + '.jpg'
        # #     cv2.imwrite(filename, det_img)


        # cv2.line(res_img, (0, res_img_rows/2),
        #          (res_img_cols, res_img_rows/2), (255, 255, 255), 3)
        # cv2.line(res_img, (0, res_img_rows - 10),
        #          (res_img_cols, res_img_rows - 10), (255, 255, 255), 3)

        cv2.imshow('res_img', res_img)



        #逐帧查看
        cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()






