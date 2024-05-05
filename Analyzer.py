import cv2 as cv
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
import time

def Test(path, names):
    img_name = path.split('/')[-1][:-4]
    ds=pd.read_csv('/home/pi/Omixdetect/UI/temp.csv')
    print('ds')
    print(list(ds.iloc[0]))
    img_path_to_save = '/home/pi/Omixdetect/UI/OmiXCVDData/AnalyzedImage' + img_name + '.png'
    #comps = [img_name[:8], img_name[8:19], img_name[19:27], img_name[27:30], img_name[30:33], img_name[33:]]
    comps = list(ds.iloc[0])
    print(comps)
    Colors = [
        ((0, 0, 255), (0, 0, 145), 'Pink', 'Positive', (np.uint8([0, 30, 100]), np.uint8([10, 255, 255])),
         (np.uint8([160, 30, 100]), np.uint8([179, 255, 255]))),
        ((0, 255, 0), (255, 0, 0), 'Orange', 'Negative', (np.uint8([11, 30, 100]), np.uint8([27, 255, 255])))
    ]
    boundaries = [(0 + 150 * i, 149 + 150 * i) for i in range(8)]

    def BuildSpace(cols):
        f = np.vectorize(int)
        col1, col2 = f(cols[0]), f(cols[1])
        space = []
        for i in np.arange(col1[0], (col2[0] + 1)):
            for j in np.arange(col1[1], (col2[1] + 1)):
                for k in np.arange(col1[2], (col2[2] + 1)):
                    space.append([i, j, k])
        return np.asarray(space)

    Low_Pink = BuildSpace(Colors[0][4])
    High_Pink = BuildSpace(Colors[0][5])
    Orange = BuildSpace(Colors[1][4])
    Mean_Low_Pink = np.mean(Low_Pink, axis=0)
    Mean_High_Pink = np.mean(High_Pink, axis=0)
    Mean_Orange = np.mean(Orange, axis=0)
    ICov_Low_Pink = np.linalg.inv(np.cov(Low_Pink.T))
    ICov_High_Pink = np.linalg.inv(np.cov(High_Pink.T))
    ICov_Orange = np.linalg.inv(np.cov(Orange.T))
    print('spaces defined')

    def Mahanalobis(v1, v2, icovar):
        return np.sqrt(((v1 - v2).T) @ icovar @ (v1 - v2))

    def FindTube(x):
        for i in range(8):
            if boundaries[i][0] <= x <= boundaries[i][1]:
                return i + 1

    def dist(tup, x):
        return min(abs(x - tup[0]), abs(x - tup[1]))

    def ValidContour(contour):
        return cv.contourArea(contour) > 300

    def Score(color):
        if color[0] in np.arange(0, 11):
            p_dist = Mahanalobis(color, Mean_Low_Pink, ICov_Low_Pink)
            o_dist = Mahanalobis(color, Mean_Orange, ICov_Orange)
            l = 0.579
        elif color[0] in np.arange(160, 180):
            p_dist = Mahanalobis(color, Mean_High_Pink, ICov_High_Pink)
            o_dist = Mahanalobis(color, Mean_Orange, ICov_Orange)
            l = 0.579
        elif color[0] in np.arange(11, 28):
            o_dist = Mahanalobis(color, Mean_Orange, ICov_Orange)
            f = np.vectorize(int)
            cols = [Colors[0][4], Colors[0][5]]
            intcols = []
            for col in cols:
                intcols.append((f(col[0])[0], f(col[1])[0]))
            distance = [dist(tup, color[0]) for tup in intcols]
            m = min(distance)
            i = distance.index(m)
            if i == 0:
                p_dist = Mahanalobis(color, Mean_Low_Pink, ICov_Low_Pink)
            else:
                p_dist = Mahanalobis(color, Mean_High_Pink, ICov_High_Pink)
            l = 0.585
        else:
            f = np.vectorize(int)
            cols = [Colors[0][4], Colors[0][5], Colors[1][4]]
            intcols = []
            for col in cols:
                intcols.append((f(col[0])[0], f(col[1])[0]))
            distance = [dist(tup, color[0]) for tup in intcols]
            m1, m2 = tuple(sorted(distance)[:2])
            i1, i2 = distance.index(m1), distance.index(m2)
            if i1 == 0:
                p_dist = Mahanalobis(color, Mean_Low_Pink, ICov_Low_Pink)
                o_dist = Mahanalobis(color, Mean_Orange, ICov_Orange)
                l = 0.579
            elif i1 == 1:
                p_dist = Mahanalobis(color, Mean_High_Pink, ICov_High_Pink)
                o_dist = Mahanalobis(color, Mean_Orange, ICov_Orange)
                l = 0.579
            else:
                o_dist = Mahanalobis(color, Mean_Orange, ICov_Orange)
                if i2 == 1:
                    p_dist = Mahanalobis(color, Mean_Low_Pink, ICov_Low_Pink)
                else:
                    p_dist = Mahanalobis(color, Mean_High_Pink, ICov_High_Pink)
                l = 0.585
        return (round(l * np.exp(-l * p_dist), 4), round(l * np.exp(-l * o_dist), 4))

    def GetExpectedPixel(img):
        colors_count = {}
        (channel_h, channel_s, channel_v) = cv.split(img)
        channel_h = channel_h.flatten()
        channel_s = channel_s.flatten()
        channel_v = channel_v.flatten()
        for i in range(len(channel_h)):
            HSV = str(channel_h[i]) + " " + str(channel_s[i]) + " " + str(channel_v[i])
            if HSV in colors_count:
                colors_count[HSV] += 1
            else:
                colors_count[HSV] = 1
        del colors_count['0 0 0']
        n_pixels = sum([colors_count[i] for i in colors_count.keys()])
        colors_count_prob = {key: (colors_count[key] / n_pixels) for key in colors_count.keys()}
        colors_count_prob = {tuple(int(t) for t in key.split(' ')): colors_count_prob[key] for key in colors_count_prob}
        expected_pixel_value = [int(round(sum([i[j] * colors_count_prob[i] for i in colors_count_prob.keys()]), 0)) for
                                j in range(3)]
        return expected_pixel_value

    def Results(samples, names):
        print('analyzing images')
        out1 = {}
        out2 = {}
        out3 = {}
        for i in samples.keys():
            key = 'Sample.ID.' + str(i)
            if len(samples[i]) == 1:
                p_score, n_score = samples[i][0]
            else:
                p_score = max(samples[i], key=lambda x: x[0])[0]
                n_score = max(samples[i], key=lambda x: x[1])[1]
            out1[key] = names[i - 1]
            out2[key] = "P Score : " + str(p_score)
            out3[key] = "N Score : " + str(n_score)
        cols = ['Test.Name', 'Kit.Lot.No', 'Date', 'Run.ID', 'Strip.No', 'Gene'] + list(out1.keys())
        data_dict = {cols[i]: comps[i] for i in range(6)}
        data_dict.update(out1)
        print('dictionaries made')
        print('writing csv')
        with open('/home/pi/Omixdetect/UI/OmiXCVDData/OmiXCVDResults1.csv', 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=cols)
            writer.writeheader()
            writer.writerows([data_dict, out2, out3])
        print("csv made")

    print('supporting functions defined')

    def Analyze(path, names):
        img = cv.imread(path)
        img1, img2 = img[550:620, 400:1550], img[580:620, 400:1550]
        img_bi_fil = cv.bilateralFilter(img2, 15, 30, 30)
        img_hsv = cv.cvtColor(img_bi_fil, cv.COLOR_BGR2HSV)
        print('cropped image')

        samples = {}

        for color in Colors:
            mask = cv.inRange(img_hsv, color[4][0], color[4][1])
            if color[2] == 'Pink':
                mask2 = cv.inRange(img_hsv, color[5][0], color[5][1])
                mask = cv.bitwise_or(mask2, mask, mask)
            ret, thresh = cv.threshold(mask, 200, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            contours = list(contours)
            for i in range((len(contours) - 1), -1, -1):
                contour = contours[i]
                if ValidContour(contour):
                    black_strip = np.zeros(img2.shape, dtype='uint8')
                    cv.drawContours(black_strip, [contour], -1, (255, 255, 255), -1)
                    region = cv.bitwise_and(img_hsv, black_strip)
                    (x, y, w, h) = cv.boundingRect(contour)
                    mean_color = GetExpectedPixel(region)
                    score = Score(mean_color)
                    tube = FindTube(x)
                    if tube not in samples.keys():
                        samples[tube] = [score]
                    else:
                        samples[tube].append(score)
                else:
                    contours.pop(i)
                cv.drawContours(img2, contours, contourIdx=-1, color=color[0], thickness=2)
        plt.imshow(img2[:, :, ::-1])
        plt.savefig(img_path_to_save)
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        sorted_samples = sorted(samples.keys())
        samples = {key: samples[key] for key in sorted_samples}
        print(samples)
        print('contours drawn, proceeding to analysis')
        return Results(samples, names)

    return Analyze(path, names)
                                               