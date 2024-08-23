import os
import cv2
import numpy as np

# 定义两个文件夹的路径
binary_folder = "out_path.txt"
ground_truth_folder = "gt_path.txt"

binary_files = []
with open(binary_folder, "r") as file:
    for line in file:
        binary_files.append(line.strip())
ground_truth_files = []
with open(ground_truth_folder, "r") as file:
    for line in file:
        ground_truth_files.append(line.strip())


accuracies = []
recalls = []
precisions = []
f1_scores = []
ious = []
specificities = []
# 遍历每一对图像
for binary_file, gt_file in zip(binary_files, ground_truth_files):
    # 读取二元图像和对应的地面真值（ground truth）图像
    binary_image = cv2.imread(os.path.join(
        binary_folder, binary_file), cv2.IMREAD_GRAYSCALE)

    ground_truth = cv2.imread(os.path.join(
        ground_truth_folder, gt_file), cv2.IMREAD_GRAYSCALE)
    x, y = binary_image.shape
    g_x, g_y = ground_truth.shape
    if x != g_x and y != g_y:
        binary_image = binary_image.astype(np.uint8)
        binary_image = cv2.resize(binary_image, (g_x, g_y))
    binary_image = np.where(binary_image < 1, 0, 255)
    # 计算真正例、假正例、真负例和假负例的数量
    true_positive = np.logical_and(
        binary_image == 255, ground_truth == 255).sum()
    false_positive = np.logical_and(
        binary_image == 255, ground_truth == 0).sum()
    true_negative = np.logical_and(binary_image == 0, ground_truth == 0).sum()
    false_negative = np.logical_and(
        binary_image == 0, ground_truth == 255).sum()
    # print(binary_file)
    # print(true_positive+false_positive+true_negative+false_negative)
    # 计算精确度
    accuracy = (true_positive + true_negative) / (true_positive +
                                                  false_positive + true_negative + false_negative)
    accuracies.append(accuracy)
    # print(true_positive,false_positive)
    # 计算召回率
    recall = true_positive / (true_positive + false_negative)
    recalls.append(recall)

    # 计算精确率
    precision = true_positive / (true_positive + false_positive)
    precisions.append(precision)

    # 计算F1分数
    f1 = 2 * (precision * recall) / (precision + recall)
    f1_scores.append(f1)
    print(f1)
    # 计算IoU
    iou = true_positive / (true_positive + false_positive + false_negative)
    ious.append(iou)
    # print(iou)
    specificity = true_negative / (true_negative + false_positive)
    specificities.append(specificity)
# 计算平均值
average_accuracy = np.mean(accuracies)
average_recall = np.mean(recalls)
average_precision = np.mean(precisions)
average_f1_score = np.mean(f1_scores)
average_iou = np.mean(ious)
average_specificity = np.mean(specificities)
# 计算标准差
std_accuracy = np.std(accuracies)
std_recall = np.std(recalls)
std_precision = np.std(precisions)
std_f1_score = np.std(f1_scores)
std_iou = np.std(ious)
std_specificity = np.std(specificities)
# 打印结果
print(f"Accuracy: {average_accuracy:.4f} +- {std_accuracy:.4f}")
print(f"Recall: {average_recall:.4f} +- {std_recall:.4f}")
print(f"Precision: {average_precision:.4f} +- {std_precision:.4f}")
print(f"F1 Score: {average_f1_score:.4f} +- {std_f1_score:.4f}")
print(f"IoU: {average_iou:.4f} +- {std_iou:.4f}")
print(f"Specificity: {average_specificity:.4f} +- {std_specificity:.4f}")
