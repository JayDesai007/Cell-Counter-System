import numpy as np #pip install numpy
import tensorflow as tf  #pip install tensorflow
import collections
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import cv2  #pip install python-opencv
from PIL import Image #pip install pillow
import matplotlib.pyplot as plt #pip install matplotlib

class ImageOperation:
    def __init__(self, filePath):
        self.filePath = filePath

    def loadImage(self):
        self.frame = cv2.imread(self.filePath)
        self.ansFrame = np.copy(self.frame)
        self.height = self.frame.shape[0]
        self.width = self.frame.shape[1]

    def cannyConvert(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        self.canny = cv2.Canny(blur, 7, 7)

    def findHoughLines(self):
        self.lines = cv2.HoughLines(self.canny, 1, np.pi / 90, 250, np.array([]), 0, 0)

    def find_avg_hline(self, temp):
        t = 0
        sum = 0.0
        for i in temp:
            sum += i[1]
            t += 1
        avg = int(sum / t)
        return [temp[0][0], avg, temp[0][2], temp[0][3], avg, temp[0][5], avg]

    def find_avg_vline(self, temp):
        t = 0
        sum = 0.0
        for i in temp:
            sum += i[1]
            t += 1
        avg = int(sum / t)
        return [temp[0][0], avg, temp[0][2], avg, temp[0][4], avg, temp[0][6]]

    def sort_lines(self):
        temp1 = []
        temp2 = []
        if self.lines is not None:
            for i in self.lines:
                rho = i[0][0]
                if rho >= 0:
                    theta = i[0][1]
                    a = int(round(np.cos(theta)))
                    b = int(round(np.sin(theta)))
                    if a == 0:
                        temp1.append([True, rho, theta, 0, rho, self.width, rho])
                    else:
                        temp2.append([True, rho, theta, rho, 0, rho, self.height])

        temp = [[False, 0, 0, 0, 0, self.width, 0], [False, 0, 0, 0, self.height, self.width, self.height]]
        temp_ = [[False, 0, 0, 0, 0, 0, self.height], [False, 0, 0, self.width, 0, self.width, self.height]]
        for j in range(0, len(temp1)):
            t = []
            for i in range(j, len(temp1)):
                if temp1[i][0] and temp1[i][2] == temp1[j][2] and np.abs(temp1[j][1] - temp1[i][1]) < 80:
                    t.append(temp1[i])
                    temp1[i][0] = False
            if t:
                temp.append(self.find_avg_hline(t))
        for j in range(0, len(temp2)):
            t = []
            for i in range(j, len(temp2)):
                if temp2[i][0] and temp2[i][2] == temp2[j][2] and np.abs(temp2[j][1] - temp2[i][1]) < 80:
                    t.append(temp2[i])
                    temp2[i][0] = False
            if t:
                temp_.append(self.find_avg_vline(t))
        intersection_point = []
        for j in temp:
            for i in temp_:
                intersection_point.append(self.lineIntersection([j[3], j[4]], [j[5], j[6]], [i[3], i[4]], [i[5], i[6]]))
        intersection_point_ = []
        intersection_point = sorted(intersection_point)
        for i in range(len(temp_)):
            intersection_point_.append(intersection_point[i * len(temp):i * len(temp) + len(temp)])
        for i in temp_:
            temp.append(i)
        self.lines = temp
        self.intersection_point = intersection_point_

    def lineIntersection(self, A, B, C, D):
        a1 = B[1] - A[1]
        b1 = A[0] - B[0]
        c1 = a1 * A[0] + b1 * A[1]
        a2 = D[1] - C[1]
        b2 = C[0] - D[0]
        c2 = a2 * C[0] + b2 * C[1]
        determinant = a1 * b2 - a2 * b1
        if determinant == 0:
            return [np.Inf, np.Inf]
        else:
            x = (b2 * c1 - b1 * c2) / determinant
            y = (a1 * c2 - a2 * c1) / determinant
            return (int(x), int(y))

    def find_area(self, A, D):
        return (D[0]-A[0])*(D[1]-A[1])

    def saveImage(self):
        cv2.imwrite("test_images/temp.jpg", self.frame)

    def fixedIntersectionPoint(self):
        self.intersection_point = []
        tempList = []
        hSpace = int(self.height/4)
        vSpace = int(self.width/4)
        for i in range(5):
            for j in range(5):
                tempList.append((vSpace*i, hSpace*j))
            self.intersection_point.append(tempList)
            tempList = []

    def display_lines(self):
        for i in self.lines:
            rho = i[1]
            theta = i[2]
            a = int(round(np.cos(theta)))
            if a == 0:
                cv2.line(self.frame, (0, rho), (self.width, rho), (255, 0, 0), 2)
            else:
                cv2.line(self.frame, (rho, 0), (rho, self.height), (255, 0, 0), 2)
        plt.imshow(self.frame)
        plt.show()
        plt.imshow(self.ansFrame)
        plt.show()

class Spore:
    def __init__(self, filePath, image):
        self.detection_graph = tf.Graph()
        self.category_index = {1: {'id': 1, 'name': 'spore'}}
        self.filePath = filePath
        self.image = image
        self.count = 0

    def loadModel(self):
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.filePath, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def draw_bounding_box_on_image(self, image,
                                   ymin,
                                   xmin,
                                   ymax,
                                   xmax,
                                   color,
                                   thickness=4,
                                   display_str_list=()):
        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=thickness, fill=color)
        font = ImageFont.load_default()
        display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = bottom + total_display_str_height
        for display_str in display_str_list[::-1]:
            text_width, text_height = font.getsize(display_str)
            margin = np.ceil(0.05 * text_height)
            draw.rectangle(
                [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                                  text_bottom)],
                fill=color)
            draw.text(
                (left + margin, text_bottom - text_height - margin),
                display_str,
                fill='black',
                font=font)
            text_bottom -= text_height - 2 * margin

    def visualize_boxes_and_labels_on_image_array(self, image, boxes, scores, line_thickness):
        box_to_display_str_map = collections.defaultdict(list)
        box_to_color_map = collections.defaultdict(str)
        min_score_thresh = 0.4
        for i in range(boxes.shape[0]):
            if scores is None or scores[i] > min_score_thresh:
                box = tuple(boxes[i].tolist())
                display_str = ''
                display_str = '{}%'.format(int(100 * scores[i]))
                box_to_display_str_map[box].append(display_str)
                box_to_color_map[box] = 'Chartreuse'
        self.count = self.count + len(box_to_display_str_map)
        for box, color in box_to_color_map.items():
            ymin, xmin, ymax, xmax = box
            image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
            self.draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                                       line_thickness, box_to_display_str_map[box])
            np.copyto(image, np.array(image_pil))
        return len(box_to_display_str_map)

    def run_inference_for_single_image(self, image, graph):
        with graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes',
                            'detection_masks']:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
                # Run inference
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def find_spore(self, A, B, C, D):
        image_np = np.copy(self.image.frame[A[1]:D[1], B[0]:A[0]])
        height = image_np.shape[0]
        width = image_np.shape[1]
        scale_percent = min(375 / width, 370 / height)
        if scale_percent > 1:
            scale_percent = 1
        image_np = cv2.resize(image_np, (int(width * scale_percent), int(height * scale_percent)),
                             interpolation=cv2.INTER_AREA)
        output_dict = self.run_inference_for_single_image(image_np, self.detection_graph)
        count = self.visualize_boxes_and_labels_on_image_array(image_np,
                                                  output_dict['detection_boxes'],
                                                  output_dict['detection_scores'],
                                                  line_thickness=8)
        image_np = cv2.resize(image_np, (width, height), interpolation=cv2.INTER_AREA)
        # print(count)
        self.image.ansFrame[A[1]:D[1], B[0]:A[0]] = image_np

        # cv2.imshow("frame", self.image.frame)
        # cv2.imshow("newFrame", self.image.ansFrame)
        # cv2.waitKey()
        return count

    # def find_valid_area(self):
    #     countList = []
    #     for j in range(len(self.image.intersection_point) - 1):
    #         for i in range(len(self.image.intersection_point[0]) - 1):
    #             countList.append(self.find_spore(self.image.intersection_point[j + 1][i], self.image.intersection_point[j][i],
    #                            self.image.intersection_point[j][i + 1], self.image.intersection_point[j + 1][i + 1]))
    #     print(countList)
# image = ImageOperation("test_images/img.jpg")
# image = ImageOperation("5min191001_010_ovl.jpg")
# image.loadImage()
# image.cannyConvert()
# image.findHoughLines()
# print(image.lines.shape)
# image.sort_lines()
# image.display_lines()

#
# sporeCount = Spore("inference_graph/frozen_inference_graph.pb", image)
# sporeCount.loadModel()
# sporeCount.find_valid_area()
