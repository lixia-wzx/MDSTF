import numpy as np
import glob
import os
import pickle
from tqdm import tqdm

data_root = r'/root/data_apolloscape'
history_frames = 6
future_frames = 6
total_frames = history_frames + future_frames
feature_id = [2, 9, 6, 7, 3, 4]
max_object_nums = 115


# 求两点距离
def get_distance(x1y1, x2y2):
    return np.sqrt(np.sum((x1y1 - x2y2) ** 2))


def get_heading_difference(heading_1, heading_2):
    if heading_1 * heading_2 < 0:
        if heading_1 < 0:
            if 2 * np.pi - heading_2 + heading_1 > np.pi:
                return heading_2 - heading_1
            else:
                return 2 * np.pi - heading_2 + heading_1
        else:
            if 2 * np.pi + heading_2 - heading_1 > np.pi:
                return heading_1 - heading_2
            else:
                return 2 * np.pi + heading_2 - heading_1

    else:
        return np.abs(heading_1 - heading_2)


def GenerateData(file_path_list, data_root, is_train=True):
    all_data = []

    for file_path_idx in tqdm(file_path_list):
        with open(file_path_idx, 'r') as reader:
            content = np.array([x.strip().split(' ') for x in reader.readlines()]).astype(float)
        scene_frames = content[:, 0].astype(np.int64)
        unique_frames = sorted(np.unique(scene_frames).tolist())
        if is_train:
            start_frame_ids = unique_frames[:-total_frames + 1]
        else:
            start_frame_ids = unique_frames[::history_frames]
        data_list = []

        for start_index in start_frame_ids:
            if is_train:
                sample_frames = np.arange(start_index, start_index + total_frames)
                last_history_sample_frames = np.arange(start_index + history_frames - 1, start_index + history_frames)
            else:
                sample_frames = np.arange(start_index, start_index + history_frames)
                last_history_sample_frames = np.arange(start_index + history_frames - 1, start_index + history_frames)

            sample_mask = np.any(scene_frames.reshape(-1, 1) == sample_frames.reshape(1, -1), axis=1)
            sample_object_ids = np.unique(content[sample_mask, 1].astype(np.int32))

            history_sample_mask = np.any(scene_frames.reshape(-1, 1) == last_history_sample_frames.reshape(1, -1),
                                         axis=1)
            xy_coordinate = content[history_sample_mask, 3:5].astype(float)
            mean_xy = np.mean(xy_coordinate, axis=0)

            if is_train:

                distance_adj = np.zeros((history_frames, max_object_nums, max_object_nums), dtype=np.float32)
                category_adj = np.zeros((history_frames, max_object_nums, max_object_nums), dtype=np.float32)
                heading_adj = np.zeros((history_frames, max_object_nums, max_object_nums), dtype=np.float32)
                # similarity_adj = np.zeros((max_object_nums,max_object_nums), dtype=np.float32)
                sample_object_input = np.zeros((total_frames, max_object_nums, len(feature_id) + 2), dtype=np.float32)
                sample_object_mask = np.zeros((total_frames, max_object_nums), dtype=bool)
            else:

                distance_adj = np.zeros((history_frames, max_object_nums, max_object_nums), dtype=np.float32)
                heading_adj = np.zeros((history_frames, max_object_nums, max_object_nums), dtype=np.float32)
                category_adj = np.zeros((history_frames, max_object_nums, max_object_nums), dtype=np.float32)
                # similarity_adj = np.zeros((max_object_nums,max_object_nums), dtype=np.float32)
                sample_object_input = np.zeros((history_frames, max_object_nums, len(feature_id) + 2), dtype=np.float32)
                sample_object_mask = np.zeros((history_frames, max_object_nums), dtype=bool)
                sample_object_origin = np.zeros((1, max_object_nums, 3), dtype=np.int32)

            # for every frame
            for frame_idx, frame in enumerate(sample_frames):

                exist_object_idx = []
                for object_idx, object_id in enumerate(sample_object_ids):
                    # frame and object
                    matched_obj = content[np.logical_and(content[:, 0] == frame, content[:, 1] == object_id)]
                    if 0 == len(matched_obj):
                        continue
                    obj_feature = matched_obj[0, feature_id]
                    obj_feature[-2:] = obj_feature[-2:] - mean_xy
                    sample_object_input[frame_idx, object_idx, 2:] = obj_feature

                    # 在时间域内，某个障碍物在某段时间内存在
                    sample_object_mask[frame_idx, object_idx] = True

                    exist_object_idx.append(object_idx)

                    if not is_train and frame_idx == 5:
                        sample_object_origin[0, object_idx, :3] = matched_obj[0, :3]

                if frame_idx <= 5:
                    for obj_id_i in exist_object_idx:
                        xy_1 = sample_object_input[frame_idx, obj_id_i, -2:]
                        heading_1 = sample_object_input[frame_idx, obj_id_i, 3]
                        category_1 = sample_object_input[frame_idx, obj_id_i, 2]
                        for obj_id_j in exist_object_idx:
                            xy_2 = sample_object_input[frame_idx, obj_id_j, -2:]
                            heading_2 = sample_object_input[frame_idx, obj_id_j, 3]
                            category_2 = sample_object_input[frame_idx, obj_id_j, 2]
                            # cos_simi = get_cos_simi(heading_1, heading_2)
                            distance_12 = get_distance(xy_1, xy_2)
                            heading_diff = get_heading_difference(heading_1, heading_2)

                            if category_1 == 3:
                                distance_adj[frame_idx, obj_id_i, obj_id_j] = 1 if distance_12 <= 10 else 0
                                if heading_diff <= np.pi / 6:
                                    heading_adj[frame_idx, obj_id_i, obj_id_j] = 1
                            elif category_1 == 4:
                                distance_adj[frame_idx, obj_id_i, obj_id_j] = 1 if distance_12 <= 15 else 0
                                if heading_diff <= np.pi / 6:
                                    heading_adj[frame_idx, obj_id_i, obj_id_j] = 1
                            else:
                                distance_adj[frame_idx, obj_id_i, obj_id_j] = 1 if distance_12 <= 20 else 0
                                if heading_diff <= np.pi / 6:
                                    heading_adj[frame_idx, obj_id_i, obj_id_j] = 1

            # add speed x ,y in dim 4,5
            new_mask = (sample_object_input[1:, :, -2:] != 0) * (sample_object_input[:-1, :, -2:] != 0).astype(float)
            sample_object_input[1:, :, :2] = (sample_object_input[1:, :, -2:] - sample_object_input[:-1, :,
                                                                                -2:]).astype(float) * new_mask
            sample_object_input[0, :, :2] = 0.

            sample_object_mask = np.expand_dims(sample_object_mask, axis=-1)
            # refine the future masks
            # data['masks'].sum(axis=0) == history_frames表示如果过去帧都在
            # 表示在过去帧都存在的情况下对未来的掩码
            if is_train:
                data = dict(
                    features=sample_object_input, masks=sample_object_mask, distance_adj=distance_adj,
                    heading_adj=heading_adj, mean=mean_xy)
                data['masks'] = data['masks'] & data['masks'][history_frames - 1]

            else:
                data = dict(
                    features=sample_object_input, masks=sample_object_mask, origin=sample_object_origin,
                    distance_adj=distance_adj, heading_adj=heading_adj, mean=mean_xy)
                data['masks'] = data['masks'] & data['masks'][history_frames - 1]

            data_list.append(data)

        all_data.extend(data_list)

    all_data = np.array(all_data)  # Train 5010 Test 415
    print(np.shape(all_data))

    # save training_data and trainjing_adjacency into a file.
    if is_train:
        save_path = os.path.join(data_root, 'train_data.pkl')
    else:
        save_path = os.path.join(data_root, 'test_data.pkl')
    with open(save_path, 'wb') as writer:
        pickle.dump([all_data], writer)


if __name__ == '__main__':
    train_file_path_list = sorted(
        glob.glob(os.path.join(data_root, 'prediction_train/*.txt')))
    test_file_path_list = sorted(
        glob.glob(os.path.join(data_root, 'prediction_test/*.txt')))

    print('Generating Training Data.')
    GenerateData(train_file_path_list, data_root, is_train=True)

    print('Generating Testing Data.')
    GenerateData(test_file_path_list, data_root, is_train=False)
