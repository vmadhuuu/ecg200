import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset
from collections import Counter
from torchvision import transforms

def create_area_chart_mc(ts, chart_path, color_mode, label_mode):
    plt.figure()
    if color_mode == 'color':
        plt.fill_between(range(len(ts)), ts, color='blue')
    else:
        plt.fill_between(range(len(ts)), ts, color='black')

    if label_mode == 'with_label':
        plt.title('Area Chart')
    else:
        plt.axis('off')


    plt.savefig(chart_path)
    plt.close()

def create_bar_chart_mc(ts, chart_path, bar_mode, color_mode, label_mode):
    plt.figure()
    if color_mode == 'color':
        color = 'blue'
    else:
        color = 'black'

    if bar_mode == 'fill':
        plt.bar(range(len(ts)), ts, color=color, width=1.0)
    elif bar_mode == 'border':
        plt.bar(range(len(ts)), ts, color='none', edgecolor=color, width=1.0)

    if label_mode == 'with_label':
        plt.title('Bar Chart')
    else:
        plt.axis('off')

    plt.savefig(chart_path)
    plt.close()

def create_line_chart_mc(ts, chart_path, color_mode, label_mode):
    plt.figure()
    color = 'blue' if color_mode == 'color' else 'black'
    plt.plot(ts, color=color)

    if label_mode == 'with_label':
        plt.title('Line Chart')
    else:
        plt.axis('off')

    plt.savefig(chart_path)
    plt.close()

def create_scatter_chart_mc(ts, chart_path, scatter_mode, color_mode, label_mode):
    plt.figure()
    color = 'blue' if color_mode == 'color' else 'black'
    if scatter_mode == 'plain':
        plt.scatter(range(len(ts)), ts, color=color)
    elif scatter_mode == 'join':
        plt.plot(ts, color='skyblue')
        plt.scatter(range(len(ts)), ts, color=color)

    if label_mode == 'with_label':
        plt.title('Scatter Chart')
    else:
        plt.axis('off')

    plt.savefig(chart_path)
    plt.close()


class TimeSeriesImageDatasetMC(Dataset):
    def __init__(self, time_series_data, labels, split, transform=None, chart_type='area', label_mode='with_label', scatter_mode='join', bar_mode='fill', color_mode='color'):         
        self.time_series_data = time_series_data
        self.labels = labels
        self.split = split
        self.transform = transform
        self.chart_type = chart_type
        self.color_mode = color_mode
        self.label_mode = label_mode
        self.scatter_mode = scatter_mode
        self.bar_mode = bar_mode
        
        self.area_chart_dir_mc = f'data/area_charts_{color_mode}_{label_mode}/{split}'
        self.line_chart_dir_mc = f'data/line_charts_{color_mode}_{label_mode}/{split}'
        self.scatter_chart_dir_mc = f'data/scatter_charts_{scatter_mode}_{color_mode}_{label_mode}/{split}' if scatter_mode else f'data/scatter_charts_{color_mode}_{label_mode}/{split}'
        self.bar_chart_dir_mc = f'data/bar_charts_{bar_mode}_{color_mode}_{label_mode}/{split}' if bar_mode else f'data/bar_charts_{color_mode}_{label_mode}/{split}'

        # Create directories if they don't exist
        os.makedirs(self.area_chart_dir_mc, exist_ok=True)
        os.makedirs(self.line_chart_dir_mc, exist_ok=True)
        os.makedirs(self.scatter_chart_dir_mc, exist_ok=True)
        os.makedirs(self.bar_chart_dir_mc, exist_ok=True)

        self._generate_charts()

    def _generate_charts(self):
        for i, ts in enumerate(self.time_series_data):
            if self.chart_type == 'area':
                chart_dir = self.area_chart_dir_mc
                chart_name = f'area_chart_{self.color_mode}_{self.label_mode}_{i}.png'
                chart_path = os.path.join(chart_dir, chart_name)
                if not os.path.exists(chart_path):
                    create_area_chart_mc(ts, chart_path, self.color_mode, self.label_mode)
    
            elif self.chart_type == 'bar':
                chart_dir = self.bar_chart_dir_mc
                chart_name = f'bar_chart_{self.bar_mode}_{self.color_mode}_{self.label_mode}_{i}.png'
                chart_path = os.path.join(chart_dir, chart_name)
                if not os.path.exists(chart_path):
                    create_bar_chart_mc(ts, chart_path, self.bar_mode, self.color_mode, self.label_mode)
    
            elif self.chart_type == 'line':
                chart_dir = self.line_chart_dir_mc
                chart_name = f'line_chart_{self.color_mode}_{self.label_mode}_{i}.png'
                chart_path = os.path.join(chart_dir, chart_name)
                if not os.path.exists(chart_path):
                    create_line_chart_mc(ts, chart_path, self.color_mode, self.label_mode)
    
            elif self.chart_type == 'scatter':
                chart_dir = self.scatter_chart_dir_mc
                chart_name = f'scatter_chart_{self.scatter_mode}_{self.color_mode}_{self.label_mode}_{i}.png'
                chart_path = os.path.join(chart_dir, chart_name)
                if not os.path.exists(chart_path):
                    create_scatter_chart_mc(ts, chart_path, self.scatter_mode, self.color_mode, self.label_mode)
    
            else:
                raise ValueError(f"Unsupported chart type: {self.chart_type}")


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        bar_type_option = self.bar_mode  
        scatter_type_option = self.scatter_mode 
        color_option = self.color_mode
        label_option = self.label_mode  
        if self.chart_type == 'area':
            img_name = f'area_chart_{color_option}_{label_option}_{idx}.png'
            img_path = os.path.join(self.area_chart_dir_mc, img_name)
    
        elif self.chart_type == 'bar':
            img_name = f'bar_chart_{bar_type_option}_{color_option}_{label_option}_{idx}.png'
            img_path = os.path.join(self.bar_chart_dir_mc, img_name)
    
        elif self.chart_type == 'line':
            img_name = f'line_chart_{color_option}_{label_option}_{idx}.png'
            img_path = os.path.join(self.line_chart_dir_mc, img_name)
    
        elif self.chart_type == 'scatter':
            img_name = f'scatter_chart_{scatter_type_option}_{color_option}_{label_option}_{idx}.png'
            img_path = os.path.join(self.scatter_chart_dir_mc, img_name)
    
        else:
            raise ValueError(f"Unsupported chart type: {self.chart_type}")
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File not found: {img_path}")

    
        img = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        
        if self.transform:
            img = self.transform(img)

        return img, label


def create_area_chart_mc(ts, chart_path, color_mode, label_mode):
    plt.figure()
    if color_mode == 'color':
        plt.fill_between(range(len(ts)), ts, color='blue')
    else:
        plt.fill_between(range(len(ts)), ts, color='black')

    if label_mode == 'with_label':
        plt.title('Area Chart')
    else:
        plt.axis('off')


    plt.savefig(chart_path)
    plt.close()

def create_bar_chart_mc(ts, chart_path, bar_mode, color_mode, label_mode):
    plt.figure()
    if color_mode == 'color':
        color = 'blue'
    else:
        color = 'black'

    if bar_mode == 'fill':
        plt.bar(range(len(ts)), ts, color=color, width=1.0)
    elif bar_mode == 'border':
        plt.bar(range(len(ts)), ts, color='none', edgecolor=color, width=1.0)

    if label_mode == 'with_label':
        plt.title('Bar Chart')
    else:
        plt.axis('off')

    plt.savefig(chart_path)
    plt.close()

def create_line_chart_mc(ts, chart_path, color_mode, label_mode):
    plt.figure()
    color = 'blue' if color_mode == 'color' else 'black'
    plt.plot(ts, color=color)

    if label_mode == 'with_label':
        plt.title('Line Chart')
    else:
        plt.axis('off')

    plt.savefig(chart_path)
    plt.close()

def create_scatter_chart_mc(ts, chart_path, scatter_mode, color_mode, label_mode):
    plt.figure()
    color = 'blue' if color_mode == 'color' else 'black'
    if scatter_mode == 'plain':
        plt.scatter(range(len(ts)), ts, color=color)
    elif scatter_mode == 'join':
        plt.plot(ts, color='skyblue')
        plt.scatter(range(len(ts)), ts, color=color)

    if label_mode == 'with_label':
        plt.title('Scatter Chart')
    else:
        plt.axis('off')

    plt.savefig(chart_path)
    plt.close()


def display_chart_mc(image_path):
    img = Image.open(image_path)
    img.show()

def accuracy_fn_mc(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


# ######################################################### COLOR AND WITH LABELLING AREA CHART - 1 ############################################################
# def create_area_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.fill_between(range(len(time_series)), time_series, color="skyblue", alpha=0.4)
#     plt.plot(time_series, color="Slateblue", alpha=0.6)
#     plt.title('Area Chart')
#     plt.xlabel('Time')
#     plt.ylabel('Value')
#     plt.savefig(save_path)
#     plt.close()

# # ######################################################### COLOR AND WITH LABELLING LINE CHART - 2 ###########################################################
# def create_line_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.plot(time_series, color="Slateblue", alpha=0.6)
#     plt.title('Line Chart')
#     plt.xlabel('Time')
#     plt.ylabel('Value')
#     plt.savefig(save_path)
#     plt.close()

######################################################### COLOR AND WITH LABELLING SCATTER PLAIN - 3 ##########################################################

# def create_scatter_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.scatter(range(len(time_series)), time_series, color="blue")
#     plt.title('Scatter Chart')
#     plt.xlabel('Time')
#     plt.ylabel('Value')
#     plt.savefig(save_path) 
#     plt.close()

######################################################### COLOR AND WITH LABELLING SCATTER JOINED - 4 ##########################################################

# def create_scatter_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.plot(range(len(time_series)), time_series, color="skyblue")
#     plt.scatter(range(len(time_series)), time_series, color="blue", s = 8)
#     plt.title('Scatter Chart')
#     plt.xlabel('Time')
#     plt.ylabel('Value')
#     plt.savefig(save_path) 
#     plt.close()

########################################################## COLOR AND WITH LABELLING AND WITH BARCHART FILL - 5 ##########################################################

# def create_bar_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.bar(range(len(time_series)), time_series, color="skyblue", edgecolor="black", width=1.0)
#     plt.title('Bar Chart')
#     plt.xlabel('Time')
#     plt.ylabel('Value')
#     plt.savefig(save_path)
#     plt.close()

######################################################### COLOR AND WITH LABELLING BARCHART WITH BORDER - 6 ##########################################################

# def create_bar_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.bar(range(len(time_series)), time_series, color='none', edgecolor="blue", width=1.0)
#     plt.title('Bar Chart')
#     plt.xlabel('Time')
#     plt.ylabel('Value')
#     plt.savefig(save_path)
#     plt.close()


# # ####################################################### COLOR AND WITHOUT LABELLING AREA CHART - 7  ###########################################################
# def create_area_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.fill_between(range(len(time_series)), time_series, color="skyblue", alpha=0.4)
#     plt.plot(time_series, color="Slateblue", alpha=0.6)
#     plt.axis("off")
#     plt.savefig(save_path)
#     plt.close()

# # ######################################################### COLOR AND WITHOUT LABELLING LINE CHART - 8 ##########################################################
# def create_line_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.plot(time_series, color="Slateblue", alpha=0.6)
#     plt.axis("off")
#     plt.savefig(save_path)
#     plt.close()

# ######################################################### COLOR AND WITHOUT LABELLING SCATTER PLAIN - 9 ##########################################################

# # def create_scatter_chart_mc(time_series, save_path):
# #     plt.figure()
# #     plt.scatter(range(len(time_series)), time_series, color="blue")
# #     plt.axis("off")
# #     plt.savefig(save_path) 
# #     plt.close()

# ######################################################### COLOR AND WITHOUT LABELLING SCATTER JOINED - 10 #########################################################

# def create_scatter_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.plot(range(len(time_series)), time_series, color="skyblue")
#     plt.scatter(range(len(time_series)), time_series, color="blue")
#     plt.axis("off")
#     plt.savefig(save_path) 
#     plt.close()

# ######################################################### COLOR AND WITHOUT LABELLING AND WITH BARCHART FILL - 11 ########################################################

# # def create_bar_chart_mc(time_series, save_path):
# #     plt.figure()
# #     plt.bar(range(len(time_series)), time_series, color="skyblue", edgecolor="blue", width=1.0)
# #     plt.axis("off")
# #     plt.savefig(save_path)
# #     plt.close()
    
# ######################################################### COLOR AND WITHOUT LABELLING BARCHART WITH BORDER - 12 ###########################################################

# def create_bar_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.bar(range(len(time_series)), time_series, color='none', edgecolor="blue", width=1.0)
#     plt.axis("off")
#     plt.savefig(save_path)
#     plt.close()


######################################################### MONOCHROME AND WITH LABELLING AND WITH AREA, LINE, PLAIN SCATTER, BAR FILL - 13, 14, 15, 17   ###########################################################
# def create_area_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.fill_between(range(len(time_series)), time_series, color="black", alpha=0.4)
#     plt.plot(time_series, color="black", alpha=0.6)
#     plt.title('Line Chart')
#     plt.xlabel('Time')
#     plt.ylabel('Value')
#     plt.savefig(save_path) 
#     plt.close()

# def create_line_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.plot(time_series, color="black", alpha=0.6)
#     plt.title('Line Chart')
#     plt.xlabel('Time')
#     plt.ylabel('Value') 
#     plt.savefig(save_path) 
#     plt.close()

# def create_scatter_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.scatter(range(len(time_series)), time_series, color="black")
#     plt.title('Scatterplot')
#     plt.xlabel('Time')
#     plt.ylabel('Value')
#     plt.savefig(save_path) 
#     plt.close()

# def create_bar_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.bar(range(len(time_series)), time_series, color="black", edgecolor="black", width=1.0)
#     plt.title('Bar Chart')
#     plt.xlabel('Time')
#     plt.ylabel('Value')  
#     plt.savefig(save_path) 
#     plt.close()

######################################################### MONOCHROME AND WITH LABELLING BARCHART ONLY BORDER - 18   ##########################################################

# def create_bar_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.bar(range(len(time_series)), time_series, color='none', edgecolor="black", width=1.0)
#     plt.title('Bar Chart')
#     plt.xlabel('Time')
#     plt.ylabel('Value')  
#     plt.savefig(save_path) 
#     plt.close()

# ######################################################### MONOCHROME AND WITH LABELLING SCATTERPLOT JOINED - 16   ############################################################

# def create_scatter_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.scatter(range(len(time_series)), time_series, color="black", s = 8)
#     plt.plot(range(len(time_series)), time_series, color="black")
#     plt.title('Scatterplot')
#     plt.xlabel('Time')
#     plt.ylabel('Value')
#     plt.savefig(save_path) 
#     plt.close()

######################################################### MONOCHROME AND WITHOUT LABELLING AREA -  19 ############################################################
# def create_area_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.fill_between(range(len(time_series)), time_series, color="black", alpha=0.4)
#     plt.plot(time_series, color="black", alpha=0.6)
#     plt.axis("off")
#     plt.savefig(save_path) 
#     plt.close()

######################################################## MONOCHROME AND WITHOUT LABELLING LINE -  20 ###########################################################

# def create_line_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.plot(time_series, color="black", alpha=0.6)
#     plt.axis("off")
#     plt.savefig(save_path) 
#     plt.close()

######################################################### MONOCHROME AND WITHOUT LABELLING SCATTER PLAIN -  21 #######################################################

# def create_scatter_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.scatter(range(len(time_series)), time_series, color="black")
#     plt.axis("off")
#     plt.savefig(save_path) 
#     plt.close()

######################################################### MONOCHROME AND WITHOUT LABELLING SCATTER JOIN -  22 #########################################################

# def create_scatter_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.scatter(range(len(time_series)), time_series, color="black", s = 8)
#     plt.plot(range(len(time_series)), time_series, color="black")
#     plt.axis("off")
#     plt.savefig(save_path) 
#     plt.close()
    
######################################################## MONOCHROME AND WITHOUT LABELLING BAR FILL -  23 #######################################################
# def create_bar_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.bar(range(len(time_series)), time_series, color="black", edgecolor="black", width=1.0)
#     plt.axis("off")
#     plt.savefig(save_path) 
#     plt.close()

######################################################## MONOCHROME AND WITHOUT LABELLING BAR BORDER -  24 #####################################################
# def create_bar_chart_mc(time_series, save_path):
#     plt.figure()
#     plt.bar(range(len(time_series)), time_series, color="none", edgecolor="black", width=1.0)
#     plt.axis("off")
#     plt.savefig(save_path) 
#     plt.close()


