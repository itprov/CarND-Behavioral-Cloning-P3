import csv
import cv2
import h5py
import numpy as np

data_dir = './data'
driving_log_file = data_dir + '/driving_log.csv'
image_dir = data_dir + '/IMG'
datasets_file = data_dir + '/datasets.h5'
max_samples = 20000

def get_log_lines():
    driving_log_lines = []
    with open(driving_log_file) as driving_log:
        reader = csv.reader(driving_log)
        for line in reader:
            driving_log_lines.append(line)
    return driving_log_lines

def get_sample(log_line, position):
    source_path = log_line[position]

    filename = source_path.split('/')[-1]
    current_path = image_dir + '/' + filename

    image = cv2.imread(current_path)
    steering_angle = float(log_line[3])
    return image, steering_angle,

def process_image(images, image, steering_angles, steering_angle, save_images=False):
    images.append(image)
    # Flip image along Y axis
    image_flipped = cv2.flip(image, flipCode=1)
    # Save if asked
    if save_images:
        cv2.imwrite('images/normal_sample.jpg', image)
        cv2.imwrite('images/flipped_sample.jpg', image_flipped)
    images.append(image_flipped)
    steering_angles.append(steering_angle)
    # Flipped image => Flip steering angle
    steering_angle_flipped = -steering_angle
    steering_angles.append(steering_angle_flipped)

def save_data(images, steering_angles):
    X = np.array(images)
    Y = np.array(steering_angles)

    datasets = {}
    datasets.update({'X': X, 'Y': Y})

    # Save dataset for convenience
    h5_data = h5py.File(datasets_file, 'w')
    for dataset_name, dataset in datasets.items():
        print('Saving:', dataset_name, dataset.shape)
        h5_dataset = h5_data.create_dataset(dataset_name, dataset.shape, dtype='f')
        if dataset_name.startswith('X'):
            h5_dataset[:,:] = dataset
        else:
            h5_dataset[:] = dataset

    return X, Y

def get_data(use_all_cams = False):
    driving_log_lines = get_log_lines()
    images = []
    steering_angles = []
    sample_num = 0
    for log_line in driving_log_lines:
        sample_num += 1
        if (sample_num <= max_samples):
            center_image, center_angle = get_sample(log_line, 0)
            # Save one of the flipped images for writeup
            save_images = sample_num == 1000
            process_image(images, center_image, steering_angles, center_angle, save_images)
            if use_all_cams:
                correction = 0.1
                left_image, left_angle = get_sample(log_line, 1)
                left_angle += correction
                process_image(images, left_image, steering_angles, left_angle)
                right_image, right_angle = get_sample(log_line, 2)
                right_angle -= correction
                process_image(images, right_image, steering_angles, right_angle)
        else:
            break

    print('#images: ', len(images))
    print('#lines: ', sample_num)
    X, Y = save_data(images, steering_angles)
    return X, Y
