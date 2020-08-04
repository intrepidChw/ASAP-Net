import class_mapping
import matplotlib.pyplot as plt
import matplotlib


if __name__ == "__main__":
    label_list = []
    color_list = []

    for i in range(1, 16):
        label_idx = class_mapping.index_to_label[i]
        index = class_mapping.label_to_index[label_idx]
        label_str = class_mapping.index_to_class[index]
        label_color = class_mapping.index_to_color[index]

        for j in range(len(label_color)):
            label_color[j] = label_color[j] / 255

        tmp = label_color[2]
        label_color[2] = label_color[0]
        label_color[0] = tmp

        label_list.append(label_str)
        color_list.append(label_color)

    x = []
    for i in range(1, 16):
        x.append(i + 0.5)
    rects = plt.bar(x=x, height=1, width=0.5, alpha=0.6, color=color_list)
    # for rect in rects1:
    #     height = rect.get_height()
    plt.xticks([index + 0.5 for index in x], label_list)
    # plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha="center", va="bottom")

    plt.show()


