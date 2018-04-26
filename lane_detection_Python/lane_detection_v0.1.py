import numpy as np
import cv2



# 返回最大连续值,长度相等时,优先返回大的值
def max_seq(data):

    result = []
    tresult = []

    tr = 0
    r = 0
    i = 0

    while True:
        if tr == 0:
            tr = tr + 1
            tresult.append(data[i])
        else:
            if data[i] == (data[i-1]+1):
                tr = tr + 1
                tresult.append(data[i])
            else:
                if tr >= r:
                    r = tr
                    result = tresult
                tr = 1
                tresult = []  # 清空list
                tresult.append(data[i])
        i = i + 1
        if i == len(data):
            break
    if tr >= r:  # 比较最后一次
        result = tresult

    return result



# 合成识别出车线和原图
def add_mask(img1, img2):
    
    # I want to put logo on top-left corner, So I create a ROI
    rows, cols ,channels= img1.shape
    roi = img1[rows / 2: rows - 25, 0: cols]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', img2gray)
    # cv2.waitKey(0)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    # cv2.imshow('mask_inv', mask_inv)
    # cv2.waitKey(0)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    # cv2.imshow('bg', img1_bg)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
    # cv2.imshow('fg', img2_fg)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[rows / 2: rows -25, 0: cols] = dst

    return img1
    # cv2.waitKey(0)
    # cv2.imshow('result', img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()






# 消除小联通区 (路灯干扰,虽然前面已经有腐蚀了。)
# num = 25
def eliminate_noise_point(src_img, num):

    # gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    temp_img = src_img.copy()

    # img = binary 不行，，那是什么？
    dst_img = src_img.copy()

    # 该函数会改变src_img使用temp_img代替吧
    contours, hierarchy = cv2.findContours(
        temp_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # contours.sort(cmp)  # 按照联通区面积排序，升序

    # 找出小联通区，并全部置0,清楚干扰
    # 置0的只是连通区的轮廓,有可能会有中心的点不会被处理掉
    # 例如IMG_7563.mp4中302帧,小点(413,203)和(413,227)
    for arr in contours:
        if arr.size < num:
            for pos in arr:
                dst_img[pos[:, 1], pos[:, 0]] = 0

    # cv2.imwrite('data_set/roi120.png', img)

    return dst_img

# 左右分两半逐行扫描,求平均坐标(后期改进),认为是目标车线
# thickness 线宽 = thickness*2
# thickness = 8
def detec_point(src_img, thickness):
    
    size = src_img.shape  # [rows, cols]

    res_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)


    mask_img = np.zeros(res_img.shape,np.uint8)

    # gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    gray_img = src_img

    i = 0

    # 由车道线特性,从下往上扫描，逐行x值向中间收敛！！
    index_l = 20
    index_r = size[1] - 35

    while i < size[0] - 50:

        i = i + 1

        # 取一行
        # 由车道线特性,从下往上扫描，逐行x值向中间收敛！！
        # ROW_L = gray_img[- i, index_l:size[1]/2]
        # ROW_R = gray_img[- i, size[1]/2:size[1]]

        left = int(index_l - 20)
        right = int(index_r + 35)

        # print left,right

        #############################
        # 此处切分注意left,size[1]/2.# 
        #############################
        ROW_L = gray_img[- i, left:size[1]/2]
        ROW_R = gray_img[- i, size[1]/2:right]

        #遍历寻找符合表达式的坐标
        # idx = [idx for (idx, val) in enumerate(col) if val == 255]

        idx_L = np.where(ROW_L == 255)
        if idx_L[0].size != 0:
            tidx_l = max_seq(idx_L[0])  # 使用max_seq,可以过滤每行中小连续
            index_l = round(np.mean(tidx_l))
            index_l = index_l + left  # 注意left,由于切分问题,换算成原始坐标
            res_img[- i, int(index_l)-thickness: int(index_l) +
                    thickness] = (255, 0, 0)
            mask_img[- i, int(index_l)-thickness: int(index_l) +
                     thickness] = (255, 0, 0)

        # print idx_L[0], index_l

        idx_R = np.where(ROW_R == 255)
        if idx_R[0].size != 0:
            tidx_r = max_seq(idx_R[0])  # 使用max_seq,可以过滤每行中小连续
            index_r = round(np.mean(tidx_r))
            index_r = index_r + size[1]/2  # 注意size[1]/2,由于切分问题,换算成原始坐标
            res_img[- i, int(index_r)-thickness: int(index_r) +
                    thickness] = (0, 255, 0)
            mask_img[- i, int(index_r)-thickness: int(index_r) +
                     thickness] = (0, 255, 0)
        
    return res_img, mask_img



kern_L = np.array(np.mat('2 3 0; 3 0 -3; 0 -3 -2'), subok=True)

kern_R = np.array(np.mat('0 -3 -2; 3 0 -3; 2 3 0'), subok=True)





if __name__ == "__main__":
    cap = cv2.VideoCapture('data_set/IMG_7563.mp4')

    if cap.isOpened() == 0:
        print'video read failed.'
    else:
        print'video read success'

    fps = cap.get(5)  # CAP_PROP_FPS
    total_frame = cap.get(7)  # CAP_PROP_FRAME_COUNT
    print 'The fps is', fps
    print 'Total frames are', total_frame

    i = 0

    while (cv2.waitKey(33) & 0xFF != ord('q')):  # ord()--return ASCII

        ret, src_img = cap.read()
        if ret == 0:
            break

        cv2.imshow('src_img', src_img)

        src_img_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

        # cv2.imshow('src_img_gray', src_img_gray)

        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

        temp_img_erode = cv2.erode(src_img_gray, element)

        # cv2.imshow('temp_img_erode', temp_img_erode)

        tem_img_L = cv2.filter2D(temp_img_erode, -1, kern_L)
        tem_img_R = cv2.filter2D(temp_img_erode, -1, kern_R)

        # cv2.imshow('tem_img_L', tem_img_L)
        # cv2.imshow('tem_img_R', tem_img_R)

        # cv2.THRESH_TOZERO   ------>  cv2.THRESH_BINARY
        # 之前想测下效果,就忘记这件事了,导致后面老是出现不该出现的非0,255的数值,
        # 一度以为是图片压缩格式的问题,后来用bmp还是不行,后来发现这里忘记改回来o(╥﹏╥)o
        ret1, dst_img_L = cv2.threshold(tem_img_L, 230, 255, cv2.THRESH_BINARY)
        ret2, dst_img_R = cv2.threshold(tem_img_R, 230, 255, cv2.THRESH_BINARY)

        # cv2.imshow('dst_img_L', dst_img_L)
        # cv2.imshow('dst_img_R', dst_img_R)

        res_img = dst_img_L + dst_img_R

        img = res_img.copy()

        res_img_rows, res_img_cols = res_img.shape

        cv2.imshow('res_img', res_img)
        #先取ROI，再画线
        #det_img会随着下面res_img的变换而变化.....难受啊.....并不是copy？  这就是ROI把
        # det_img = res_img[res_img_rows / 2: res_img_rows - 10, 0: res_img_cols]
        # 修改切片,由从倒数10行增加到20,不然中间有被情况下会被雨刷干扰.....
        det_img = res_img[res_img_rows / 2: res_img_rows - 25, 0: res_img_cols]

        # 消除小联通区
        det_img_1 = eliminate_noise_point(det_img, 25)

        cv2.imshow('det_img_1', det_img_1)

        det_img_2, det_mask = detec_point(det_img_1, 8)
        
        cv2.imshow('det_img_2', det_img_2)

        cv2.imshow('det_mask', det_mask)

        added_img = add_mask(src_img, det_mask)

        cv2.imshow('added_img', added_img)

        i = i + 1

        print 'frame:',i
        # # if i % 30 == 0 :
        # #     filename = 'data_set/ROI' + str(i) + '.jpg'
        # #     cv2.imwrite(filename, det_img)

        # cv2.line(res_img, (0, res_img_rows/2),
        #          (res_img_cols, res_img_rows/2), (255, 255, 255), 3)
        # cv2.line(res_img, (0, res_img_rows - 10),
        #          (res_img_cols, res_img_rows - 10), (255, 255, 255), 3)

        # cv2.imshow('res_img', res_img)

        # 逐帧查看
        # cv2.waitKey(0)

        # 保存问题图像
        if cv2.waitKey(0) & 0xFF == ord('s'):
            filename = 'data_set/erro_img/det_img_' + str(i) + '.png'
            cv2.imwrite(filename, det_img_1)
            filename = 'data_set/erro_img/res_img_' + str(i) + '.png'
            cv2.imwrite(filename, img)
            

    cap.release()
    cv2.destroyAllWindows()
