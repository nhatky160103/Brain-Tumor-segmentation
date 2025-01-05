"""
    CS5001 Fall 2022
    Final Project: Brain Tumor Segmentation
    - Generating learning curves
    Hua Wang
"""
from unet.test import *
import pandas as pd


def get_dicts(log_file):
    """
    get the loss_accuracy_list and train_valid_dict
    :param log_file: a path to read the training log file
    :return: loss_accuracy_dict, train_valid_dict, dictionaries
    """

    # read lines from log_file.txt
    with open(log_file, 'r') as f:
        lines = f.readlines()

    # convert string into list
    loss_accuracy_list = []
    for line in lines:
        if 'loss' in line:
            loss_accuracy_list.append(line)

    # convert list into dict
    loss_accuracy_dict = {}
    train_valid_dict = {}
    for loss_accuracy in loss_accuracy_list:
        # extract epoch-iteration, loss and accuracy from log file
        epoch_iteration = loss_accuracy.split('-')[1] + '-' +\
                          loss_accuracy.split('-')[3]
        loss = loss_accuracy.split('loss')[1].split('accuracy')[0].strip()
        accuracy = loss_accuracy.split('accuracy')[1].strip()

        # extract validation accuracy (dice score) and training accuracy
        if 'valid' in accuracy:
            temp_str = accuracy.split("valid")
            accuracy = temp_str[0].strip()
            dice = temp_str[2].strip().split('dice score:')[1].strip()
            train_valid_dict[epoch_iteration] = [accuracy, dice]
        loss_accuracy_dict[epoch_iteration] = [loss, accuracy]

    return loss_accuracy_dict, train_valid_dict


def convert_dict_to_df(loss_accuracy_dict, columns):
    """
    Convert the loss_accuracy_dict into a dataframe
    with customized column names
    :param loss_accuracy_dict: dictionary
    :param columns: column names for the dataframe
    :return: df, dataframe
    """

    df = pd.DataFrame(columns=columns)
    for key, value in loss_accuracy_dict.items():
        # separate train epoch and iterations
        epoch_iteration = key.split(":")[0].split("-")
        epoch = int(epoch_iteration[0])
        iteration = int(epoch_iteration[1])
        train_instance = (epoch - 1) * 88 + iteration

        df_ = pd.DataFrame(
            [[key, train_instance, float(value[0]), float(value[1])]],
            columns=columns
        )  # temporary dataframe for the current row

        df = pd.concat([df, df_])  # concat dataframes

    return df


def calculate_mean_per_10_instances(df, iteration_col, band_col, band_num):
    """
    Calculate the mean for every other 'band_num' training
    instances over the training process
    :param df: dataframe, the accuracy and the index of training instances
    :param iteration_col: the column name of the training instance index
    :param band_col: the column name of a column to store new index
    :param band_num: int
    :return: df, dataframe
    """
    df = df.copy()
    df[band_col] = df[iteration_col] // band_num
    df = df.groupby([band_col]).mean()
    df = df.reset_index()
    return df


def draw_accuracy_curve(df, x_col, y_col, title='', x_label='', y_label=''):
    """
    Draw an accuracy curve
    :param df: dataframe that stores accuracies for every training instance
    :param x_col: training instance column name, string
    :param y_col: accuracy column name, string
    :param title: figure title, string
    :param x_label: x label, string
    :param y_label: y label, string
    :return: a figure
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df[x_col], df[y_col], label='accuracy')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    curve = plt.gcf()
    plt.show()
    return curve


def draw_train_valid_curves(train_valid_df):
    """
    Draw accuracies of the training set and the validation set
    :param train_valid_df: dataframe,
                        a dataframe with accuracies on training/validation sets
    :return: a figure
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_valid_df['train_instance'],
             train_valid_df['train'], label='train')
    plt.plot(train_valid_df['train_instance'],
             train_valid_df['valid'], label='valid')
    plt.title('Dice score (i.e. accuracy) of train set and valid set')
    plt.xlabel('Training instance')
    plt.ylabel('Dice score (accuracy)')
    plt.legend()
    curves = plt.gcf()
    plt.show()
    return curves


def plot_curves_one_model(chosen_model):
    """
    plot figure of a learning curve and a figure of a train/valid
    accuracy curve for the chosen model
    :param chosen_model: model name
    :return: none
    """

    # create loss_accuracy_df and draw learning curve
    loss_accuracy_dict, train_valid_dict = \
        get_dicts(f"../data/{chosen_model}/log_file.txt")
    loss_accuracy_df = convert_dict_to_df(
        loss_accuracy_dict,
        ['epoch-iteration', 'train_instance', 'loss', 'accuracy']
    )
    loss_accuracy_df_mean = calculate_mean_per_10_instances(
        loss_accuracy_df,
        'train_instance',
        'per_10_instance',
        10
    )
    accuracy_curve = draw_accuracy_curve(
        loss_accuracy_df_mean, 'per_10_instance',
        'accuracy', title='Accuracy curve',
        x_label='per 10 training instances',
        y_label='dice score (mean of 10 training instances)'
    )
    accuracy_curve.savefig(
        f"../data/data_analysis/{chosen_model}_accuracy_curve.png"
    )

    # create train_valid_df and draw curves
    train_valid_df = convert_dict_to_df(
        train_valid_dict,
        ['epoch-iteration', 'train_instance', 'train', 'valid']
    )
    print(train_valid_df.head(10))
    train_valid_curves = draw_train_valid_curves(train_valid_df)
    train_valid_curves.savefig(
        f"../data/data_analysis/{chosen_model}_train_valid_dice.png"
    )


def plot_curves_multi_model(model_list, label_list, file_name):
    """
    plot learning curves for a list of model chosen on one figure
    :param model_list: a list of model names
    :param label_list: a list of labels for models
    :param file_name: the output file name
    :return: none
    """
    plt.figure(figsize=(10, 5))  # initialize canvas
    for i in range(len(model_list)):
        path = model_list[i]
        label = label_list[i]

        # create dataframe to store mean loss_accuracy
        # for each 10 training instances
        loss_accuracy_dict, train_valid_dict = get_dicts(
            f"../data/{path}/log_file.txt"
        )
        loss_accuracy_df = convert_dict_to_df(
            loss_accuracy_dict,
            ['epoch-iteration', 'train_instance', 'loss', 'accuracy']
        )
        loss_accuracy_df_mean = calculate_mean_per_10_instances(
            loss_accuracy_df,
            'train_instance',
            'per_10_instance',
            10
        )

        # plot
        plt.plot(loss_accuracy_df_mean['per_10_instance'],
                 loss_accuracy_df_mean['accuracy'],
                 label=label)

    plt.title("Learning Curve")
    plt.xlabel("per 10 training instances")
    plt.ylabel("Dice score (mean of 10 training instances)")
    plt.legend()
    curve = plt.gcf()
    plt.show()
    curve.savefig(f"../data/data_analysis/{file_name}.png")


if __name__ == '__main__':
    plot_curves_one_model("pretrained2")
    plot_curves_multi_model(
        ["pretrained1", "pretrained3", "pretrained4", "pretrained5"],
        ["l_rate=0.0001", "l_rate=0.001", "l_rate=0.01", "l_rate=0.1"],
        "accuracy_curves_multiple_loss"
    )
