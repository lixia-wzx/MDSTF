import numpy as np
import torch


def inverse_transform(prediction, last_position):
    # (64,6,115,2),(64,115,2)
    for step in range(prediction.shape[1]):
        prediction[:, step, :, :] = prediction[:, step, :, :] + last_position
        last_position = prediction[:, step, :, :]
    return prediction


def eva(predicted, ground_truth, future_masks):
    # (64,6,114,2),(64,6,114,8),(64,6,114,1)
    predicted = predicted.transpose(2, 3).transpose(1, 2)  # (64,2,6,114)
    ground_truth = ground_truth.transpose(2, 3).transpose(1, 2)  # (64,8,6,114)
    future_masks = future_masks.transpose(2, 3).transpose(1, 2)  # (64,1,6,114)

    category_mask = ground_truth[:, 2:3, :, :]  # (N, C, T, V)=(64, 1, 6, 114)

    ### overall dist
    overall_sum_time, overall_num = compute_RMSE(predicted, ground_truth[:, -2:, :, :], future_masks)

    ### car dist
    car_mask = (((category_mask == 1) + (category_mask == 2)) > 0).float().to(device)
    car_mask = future_masks * car_mask
    car_sum_time, car_num = compute_RMSE(predicted, ground_truth[:, -2:, :, :], car_mask)

    ### human dist
    human_mask = (category_mask == 3).float().to(device)
    human_mask = future_masks * human_mask
    human_sum_time, human_num = compute_RMSE(predicted, ground_truth[:, -2:, :, :], human_mask)

    ### bike dist
    bike_mask = (category_mask == 4).float().to(device)
    bike_mask = future_masks * bike_mask
    bike_sum_time, bike_num = compute_RMSE(predicted, ground_truth[:, -2:, :, :], bike_mask)

    return overall_num, overall_sum_time, car_num, car_sum_time, human_num, human_sum_time, bike_num, bike_sum_time


def compute_RMSE(predicted, ground_truth, masks, error_order=2):
    predicted = predicted * masks  # (N, C, T, V)=(N, 2, 6, 114)
    ground_truth = ground_truth * masks  # (N, C, T, V)=(N, 2, 6, 114)

    x2y2 = torch.sum(torch.abs(predicted - ground_truth) ** error_order,
                     dim=1)  # x^2+y^2, (N, C, T, V)->(N, T, V)=(64, 6, 114)
    total_sum_time = x2y2.sum(dim=-1)  # (N, T, V) -> (N, T)=(64, 6)
    total_mask = masks.sum(dim=1).sum(dim=-1)  # (N, C, T, V) -> (N, T)=(N, 6)

    return total_sum_time.detach().cpu().numpy(), total_mask.detach().cpu().numpy()


def display_result(pra_results, pra_pref='Train_epoch'):
    # all_overall_sum_list,all_overall_num_list:(num_batch*batch_size,6)
    all_overall_sum_list, all_overall_num_list = pra_results
    overall_sum_time = np.sum(all_overall_sum_list ** 0.5, axis=0)
    overall_num_time = np.sum(all_overall_num_list, axis=0)
    overall_loss_time = (overall_sum_time / overall_num_time)
    return overall_loss_time


def show_result(result_car, result_human, result_bike, stage="val"):
    result = 0.20 * result_car + 0.58 * result_human + 0.22 * result_bike
    WSADE = np.sum(result) / 6
    ADE_v = np.sum(result_car) / 6
    ADE_p = np.sum(result_human) / 6
    ADE_b = np.sum(result_bike) / 6

    WSFDE = result[-1]
    FDE_v = result_car[-1]
    FDE_p = result_human[-1]
    FDE_b = result_bike[-1]

    if stage == "val":
        log = 'val ADEv: {:.4f}, val ADEp: {:.4f}, val ADEb: {:.4f}, val WSADE: {:.4f}, val FDEv: {:.4f}, val FDEp: {:.4f}, val FDEb: {:.4f},val WSFDE: {:.4f}'
        print(log.format(ADE_v, ADE_p, ADE_b, WSADE, FDE_v, FDE_p, FDE_b, WSFDE), flush=True)

    return WSADE, WSFDE


def inverse_transform_1(data, last_position, mean_xy):
    # (64,6,115,2),(64,115,2),(64,2)
    mean_xy = mean_xy.unsqueeze(dim=1)
    last_position = last_position + mean_xy
    for step in range(data.shape[1]):
        data[:, step, :, :] = data[:, step, :, :] + last_position
        last_position = data[:, step, :, :]
    return data


def save_result1(prediction, origin):
    # prediction:(1,6,115,2), origin:(1,1,115,3)
    with open("/root/data_apolloscape/prediction_result/prediction_result/prediction_result.txt",
              'a') as writer:
        for step in range(prediction.shape[1]):
            idx = torch.where(origin[0, 0, :, 0] != 0)[0]
            step_info = prediction[0, step][idx]
            front = origin[0, 0][idx]
            front[:, 0] = front[:, 0] + step + 1
            all_info = torch.cat([front, step_info], dim=1)
            for i in range(all_info.shape[0]):
                a = str(int(all_info[i, 0])) + " " + str(int(all_info[i, 1])) + " " + str(
                    int(all_info[i, 2])) + " " + str(float(all_info[i, 3])) + " " + str(float(all_info[i, 4])) + "\n"
                writer.write(a)
