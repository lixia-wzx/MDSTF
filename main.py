from torch.utils.data import DataLoader
import time
from feeder import Feeder
from model import Model
from utils import *
from evalution import evalution_result

SEED = 5
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

train_datasets = r"/root/data_apolloscape/train_data_119.pkl"
test_datasets = r"/root/data_apolloscape/test_data_119.pkl"
model_save_path = r"/root/data_apolloscape/model_save/"
max_object_num = 115
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
lr = 0.001
batch_size = 32
epochs = 200
dropout = 0.2
in_channels = 6
out_channels = 2
hidden_size = 64
heads = 8
layers = 5
history_frames = 6
future_frames = 6
kernel_size = [2, 3]
paddings = [[1, 2, 2], [2, 2, 2]]
dilations = [[1, 2, 2], [1, 1, 1]]

train_time = []
best_wsade = []
best_epoch = []
best_wsfde = []


def get_data_loader():
    trainLoader = Feeder(r"/kaggle/input/", train_datasets, 0.8, 'train')
    train_datasets_loader = DataLoader(dataset=trainLoader, batch_size=batch_size, shuffle=True, num_workers=2)
    valLoader = Feeder(r"/kaggle/input/", train_datasets, 0.8, 'val')
    val_datasets_loader = DataLoader(dataset=valLoader, batch_size=batch_size, shuffle=True, num_workers=2)
    testLoader = Feeder(r"/kaggle/input/", test_datasets, 0.8, 'test')
    test_datasets_loader = DataLoader(dataset=testLoader, batch_size=1, shuffle=False, num_workers=2)
    return train_datasets_loader, val_datasets_loader, test_datasets_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_data_loader()
    model = Model(in_channels, out_channels, heads, hidden_size, layers, history_frames, max_object_num, paddings,
                  dilations, kernel_size, dropout)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                                                           threshold=0.01, threshold_mode="abs", verbose=False)
    for epoch in range(1, epochs + 1):
        print("Train start")
        print("Epoch:", epoch)
        for param_group in optimizer.param_groups:
            print(f'Learning Rate: {param_group["lr"]}')

        model.train()
        epoch_start_time = time.perf_counter()
        for i, batch_data in enumerate(train_loader):
            features, masks, distance_adj, heading_adj, mean = batch_data
            masks = masks.to(device)
            distance_adj = distance_adj.to(device)
            heading_adj = heading_adj.to(device)
            # category_adj = category_adj.to(device)
            features_x = features[:, :history_frames, :, :6].to(device)  # (64,6,114,6)
            features_y = features[:, history_frames:, :, :].to(device)  # (64,6,114,8)
            future_masks = masks[:, history_frames:, :, :]  # (64,6,114,1)

            prediction = model(features_x, distance_adj, heading_adj)  # (64,6,114,2)

            loss = torch.sum(torch.abs(prediction * future_masks - features_y[:, :, :, :2] * future_masks)) / torch.sum(
                future_masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 40 == 0:
                log = 'Train Iter: {:03d}, Train Loss: {:.4f}'
                print(log.format(i, loss.item()), flush=True)

        # val
        all_overall_sum_list = []
        all_overall_num_list = []
        all_car_sum_list = []
        all_car_num_list = []
        all_human_sum_list = []
        all_human_num_list = []
        all_bike_sum_list = []
        all_bike_num_list = []
        model.eval()
        for i, batch_data in enumerate(val_loader):
            features, masks, distance_adj, heading_adj, mean = batch_data
            masks = masks.to(device)
            distance_adj = distance_adj.to(device)
            heading_adj = heading_adj.to(device)
            # category_adj = category_adj.to(device)
            features_x = features[:, :history_frames, :, :6].to(device)  # (64,6,114,6)
            features_y = features[:, history_frames:, :, :].to(device)  # (64,6,114,8)

            future_masks = masks[:, history_frames:, :, :]  # (64,6,114,1)
            last_position = features[:, history_frames - 1, :, -2:].to(device)

            prediction = model(features_x, distance_adj, heading_adj)  # (64,6,114,2)
            prediction = inverse_transform(prediction, last_position)

            a, b, c, d, e, f, g, h = eva(prediction, features_y, future_masks)
            all_overall_num_list.extend(a)
            all_overall_sum_list.extend(b)
            all_car_num_list.extend(c)
            all_car_sum_list.extend(d)
            all_human_num_list.extend(e)
            all_human_sum_list.extend(f)
            all_bike_num_list.extend(g)
            all_bike_sum_list.extend(h)

        result_car = display_result([np.array(all_car_sum_list), np.array(all_car_num_list)], pra_pref='car')
        result_human = display_result([np.array(all_human_sum_list), np.array(all_human_num_list)], pra_pref='human')
        result_bike = display_result([np.array(all_bike_sum_list), np.array(all_bike_num_list)], pra_pref='bike')
        WSADE, WSFDE = show_result(result_car, result_human, result_bike)
        torch.save(model.state_dict(), model_save_path + "epoch_" + str(epoch) + "_" + str(round(WSADE, 4)) + ".pth")
        best_wsade.append(WSADE)
        best_wsfde.append(WSFDE)
        best_epoch.append(epoch)
        epoch_end_time = time.perf_counter()
        print("epoch spend time: %.4f" % (epoch_end_time - epoch_start_time))
        print("-----------------------------------------------------------")
        scheduler.step(WSADE)

    bestid = np.argmin(best_wsade)  # index
    model.load_state_dict(
        torch.load(
            model_save_path + "epoch_" + str(best_epoch[bestid]) + "_" + str(round(best_wsade[bestid], 4)) + ".pth"))

    for k, batch_data in enumerate(test_loader):
        features, masks, origin, distance_adj, heading_adj, category_adj, mean_xy = batch_data
        features = features.to(device)
        mean_xy = mean_xy.to(device)  # (64,6,114,1)
        origin = origin.to(device)
        heading_adj = heading_adj.to(device)  # (64,6,115,115)
        category_adj = category_adj.to(device)  # (64,6,115,115)
        distance_adj = distance_adj.to(device)  # (64,6,115,115)
        last_position = features[:, history_frames - 1, :, -2:]
        with torch.no_grad():
            prediction = model(features[:, :, :, :6], distance_adj, heading_adj, category_adj,
                               teacher_location=None)  # (64,6,114,2)
        prediction = inverse_transform_1(prediction, last_position, mean_xy)
        save_result1(prediction, origin)
    print("evalution...")
    evalution_result()
