import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import pandas as pd


class FPMatch:
    def __init__(self, screenshot, displays=False):
        self.target_offset = self.output = None
        self.displays = displays
        self.cdict = dict()
        self.img_touse = self.preproc(screenshot).astype('uint8')

        self.clone_target = self.get_regions(self.img_touse)

        if self.displays:
            width = 4
            length = int(np.ceil(len(self.cdict.keys()) / 4))
            fig, ax = plt.subplots(length, width, figsize=(3 * width, 3 * length))
            for i, key in enumerate(self.cdict.keys()):
                ax[i // 4, i % 4].imshow(self.cdict[key]['figure'], cmap='gray')
                ax[i // 4, i % 4].set_xticks([])
                ax[i // 4, i % 4].set_yticks([])
            plt.show()
            plt.close()
            self.disp(self.clone_target, cmap='gray')

        self.get_matches()
        self.get_results()

    def preproc(self, img):
        input_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        input_scaled = cv2.resize(input_gray, (1366, 768), interpolation=cv2.INTER_AREA)
        res = input_scaled

        if self.displays:
            self.disp(res, title='preprocessed')

        return res

    def get_regions(self, img):
        horizontal = np.copy(img)
        vertical = np.copy(img)

        # Extract horizontal and vertical lines by using morphological operations
        # https://docs.opencv.org/3.4/dd/dd7/tutorial_morph_lines_detection.html

        # Specify size on horizontal axis
        cols = horizontal.shape[1]
        horizontal_size = cols // 20
        # Create structure element for extracting horizontal lines through morphology operations
        horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        # Apply morphology operations
        horizontal = cv2.erode(horizontal, horizontal_structure)
        horizontal = cv2.dilate(horizontal, horizontal_structure)

        # Specify size on vertical axis
        rows = vertical.shape[0]
        verticalsize = rows // 20
        # Create structure element for extracting vertical lines through morphology operations
        vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
        # Apply morphology operations
        vertical = cv2.erode(vertical, vertical_structure)
        vertical = cv2.dilate(vertical, vertical_structure)

        imghv = vertical + horizontal
        if self.displays:
            self.disp(imghv)

        imgbin = (imghv > 10).astype('uint8') * 255

        # FIND COMPONENTS
        cntrs, _ = cv2.findContours(imgbin[:, :600], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if self.displays:
            self.disp(
                cv2.drawContours(np.zeros(imgbin.shape), cntrs, -1, 255, 2),
                title='components contours'
            )
        cntrs_area = [cntr for cntr in cntrs if 7_500 < cv2.contourArea(cntr) < 8_000]
        if len(cntrs_area) == 0:
            raise Exception('Could not find components box')

        for i, cntr in enumerate(cntrs_area):
            self.cdict[i] = dict()
            self.cdict[i]['contour'] = cntr
            self.cdict[i]['area'] = cv2.contourArea(cntr)
            self.cdict[i]['mincoords'] = (cntr[:, 0].min(), cntr[:, 0].max())

        bcrop = 10
        for key in self.cdict.keys():
            self.cdict[key]['figure'] = (self.apply_bbox(
                fig=img,
                bbox=self.get_bbox(
                    self.cdict[key]['contour']
                )
            )[bcrop:-bcrop, bcrop:-bcrop].astype('uint8'))

        # FIND CLONE TARGET
        cntrs, _ = cv2.findContours(imgbin[:, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if self.displays:
            self.disp(
                cv2.drawContours(np.zeros(imgbin.shape), cntrs, -1, 255, 2),
                title='clone target contours'
            )

        try:
            cntrs_area = [cntr for cntr in cntrs if 200_000 < cv2.contourArea(cntr) < 240_000]
            bbox = self.get_bbox(cntrs_area[0])
            clone_target = self.apply_bbox(
                fig=img,
                bbox=bbox
            )[:-50, :-85]
            self.target_offset = np.array(bbox[0, :])
        except IndexError:
            raise Exception('Could not find clone target box')

        return clone_target.astype('uint8')

    def get_matches(self):
        for key in self.cdict.keys():
            component = self.cdict[key]['figure']
            self.cdict[key]['evals'] = dict()
            component_canny = cv2.Canny(component, 50, 200)
            (tH, tW) = component_canny.shape[:2]

            # Multi-scale Template Matching using Python and OpenCV
            # https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/

            # load the image, convert it to grayscale, and initialize the
            # bookkeeping variable to keep track of the matched region
            gray = self.clone_target.astype('uint8')
            found = None

            # loop over the scales of the image
            for scale in np.linspace(0.8, 1.0, 21, endpoint=True)[::-1]:
                # resize the image according to the scale, and keep track
                # of the ratio of the resizing
                resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
                r = gray.shape[1] / float(resized.shape[1])

                # if the resized image is smaller than the template, then break
                # from the loop
                if resized.shape[0] < tH or resized.shape[1] < tW:
                    break
                # detect edges in the resized, grayscale image and apply template
                # matching to find the template in the image
                edged = cv2.Canny(resized, 50, 200)
                result = cv2.matchTemplate(edged, component_canny, cv2.TM_CCOEFF)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

                # if we have found a new maximum correlation value, then update
                # the bookkeeping variable
                if found is None or maxVal > found[0]:
                    found = (maxVal, maxLoc, r)
                    self.cdict[key]['evals'] = {
                        'scale': scale,
                        'maxVal': maxVal
                    }

            # unpack the bookkeeping variable and compute the (x, y) coordinates
            # of the bounding box based on the resized ratio
            (_, maxLoc, r) = found
            (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
            (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
            self.cdict[key]['best_loc'] = {
                'start': np.array([int(startX), int(startY)]),
                'end': np.array([int(endX), int(endY)]),
            }

    def get_results(self):
        self.output = np.zeros(self.img_touse.shape + (4,), 'uint8')
        colors = [
            (255, 0, 0, 255),
            (0, 255, 0, 255),
            (0, 0, 255, 255),
            (255, 255, 0, 255),
        ]

        data = [(key, elem['evals']['scale'], elem['evals']['maxVal'], elem['mincoords'])
                for key, elem in self.cdict.items()]
        df = pd.DataFrame(data=data, columns=['key', 'scale', 'maxVal', 'mincoords'])
        selections = df.sort_values('maxVal', ascending=False).iloc[:4]

        for i, key in enumerate(selections['key']):
            cv2.drawContours(
                self.output,
                [self.cdict[key]['contour']],
                -1, colors[i], 2
            )
            st = tuple(self.cdict[key]['best_loc']['start'] + self.target_offset)
            en = tuple(self.cdict[key]['best_loc']['end'] + self.target_offset)
            cv2.rectangle(self.output, st, en, colors[i], 2)

        plt.figure(figsize=(16, 9))
        plt.imshow(self.img_touse, cmap='gray')
        plt.imshow(self.output, alpha=.5)
        plt.xticks([])
        plt.yticks([])
        plt.show()
        plt.close()

    @staticmethod
    def get_bbox(contour):
        min_x = np.min(contour[:, :, 0])
        max_x = np.max(contour[:, :, 0])

        min_y = np.min(contour[:, :, 1])
        max_y = np.max(contour[:, :, 1])

        return np.array([[min_x, min_y], [max_x, max_y]])

    @staticmethod
    def fill_bbox(shape, bbox, value=1):
        zeros = np.zeros(shape)
        zeros[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]] = value
        return zeros

    @staticmethod
    def apply_bbox(fig, bbox):
        return fig[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]]

    @staticmethod
    def disp(fig, cmap='gray', figsize=(8, 5), title=''):
        plt.figure(figsize=figsize)
        plt.imshow(fig, cmap=cmap, vmin=0, vmax=255)
        plt.xticks([])
        plt.yticks([])
        plt.title(title)
        plt.show()
        plt.close()
