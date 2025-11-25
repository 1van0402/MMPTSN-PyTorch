import numpy as np
import pandas as pd
import random

class TSLoader(object):

    def __init__(self, path, split, split_n, seed):
        data = np.array(pd.read_csv(path)).astype("float32")

        self.data = data
        self.seed = seed
        self.split = split
        self.split_n = split_n

    def get_phenology_data(self, data):
        return data[:, -7:-2].astype("float32")

    def normalize(self, X, mean=None, std=None):
        if mean is None:
            mean = np.mean(X, axis=(0, 1))
        if std is None:
            std = np.std(X, axis=(0, 1))
        return (X - mean) / (std + 1e-8), mean, std

    def add_noise(self, X, scale=0.01):
        noise = np.random.normal(0, scale, X.shape)
        return X + noise

    def smoothing(self, X, window_size=3):
        return np.convolve(X, np.ones(window_size) / window_size, mode='valid')

    def random_scaling(self, X, scale_range=(0.8, 1.2)):
        scale_factor = random.uniform(scale_range[0], scale_range[1])
        return X * scale_factor

    def random_time_flip(self, X):
        if random.random() > 0.5:
            return X[::-1]
        return X

    def noise_phenology(self, x_phe, std=0.03):
        noise = np.random.normal(0, std, size=x_phe.shape)
        return x_phe + noise

    def augment(self, X_spectral, X_spatial, X_sar, X_phe, Y):
        augmented_spectral = []
        augmented_spatial = []
        augmented_sar = []
        augmented_phe = []
        augmented_labels = []

        for x_spec, x_vhd, x_sar, x_phe, y in zip(X_spectral, X_spatial, X_sar, X_phe, Y):
            augmented_spectral.append(x_spec)
            augmented_spatial.append(x_vhd)
            augmented_sar.append(x_sar)
            augmented_phe.append(x_phe)
            augmented_labels.append(y)

            augmented_spectral.append(self.add_noise(x_spec))
            augmented_spatial.append(self.add_noise(x_vhd, scale=0.005))
            augmented_sar.append(self.add_noise(x_sar, scale=0.005))
            augmented_phe.append(self.noise_phenology(x_phe))
            augmented_labels.append(y)

            shift = np.random.randint(-1, 2)
            augmented_spectral.append(np.roll(x_spec, shift, axis=0))
            augmented_spatial.append(np.roll(x_vhd, shift, axis=0))
            augmented_sar.append(np.roll(x_sar, shift, axis=0))
            augmented_phe.append(x_phe)
            augmented_labels.append(y)

            augmented_spectral.append(self.random_scaling(x_spec, scale_range=(0.8, 1.2)))
            augmented_spatial.append(self.random_scaling(x_vhd, scale_range=(0.8, 1.2)))
            augmented_sar.append(self.random_scaling(x_sar, scale_range=(0.8, 1.2)))
            augmented_phe.append(x_phe)
            augmented_labels.append(y)

            augmented_spectral.append(self.random_time_flip(x_spec))
            augmented_spatial.append(self.random_time_flip(x_vhd))
            augmented_sar.append(self.random_time_flip(x_sar))
            augmented_phe.append(x_phe)
            augmented_labels.append(y)

        return np.array(augmented_spectral), np.array(augmented_spatial), np.array(augmented_sar), np.array(augmented_phe), np.array(augmented_labels)

    def get_data(self):
        label0 = self.data[self.data[:, 0] == 0]
        label1 = self.data[self.data[:, 0] == 1]
        label2 = self.data[self.data[:, 0] == 2]

        number0 = label0.shape[0]
        number1 = label1.shape[0]
        number2 = label2.shape[0]

        np.random.seed(self.seed)
        label0_dis = np.random.permutation(label0)
        label1_dis = np.random.permutation(label1)
        label2_dis = np.random.permutation(label2)

        train0 = label0_dis[:int(number0 * self.split)]
        val0 = label0_dis[int(number0 * self.split):int(number0 * self.split_n)]
        test0 = label0_dis[int(number0 * self.split_n):]

        train1 = label1_dis[:int(number1 * self.split)]
        val1 = label1_dis[int(number1 * self.split):int(number1 * self.split_n)]
        test1 = label1_dis[int(number1 * self.split_n):]

        train2 = label2_dis[:int(number2 * self.split)]
        val2 = label2_dis[int(number2 * self.split):int(number2 * self.split_n)]
        test2 = label2_dis[int(number2 * self.split_n):]

        train = np.vstack((train0, train1, train2))
        val = np.vstack((val0, val1, val2))
        test = np.vstack((test0, test1, test2))

        def extract_features(data):
            spectral_features = []
            vhd_features = []
            sar_features = []

            for i in range(12):
                start_idx = 12 + i * 17
                end_idx_spectral = start_idx + 12

                month_spectral = data[:, start_idx:end_idx_spectral]
                spectral_features.append(month_spectral)

                end_idx_sar = end_idx_spectral + 2
                month_sar = data[:, end_idx_spectral:end_idx_sar]
                sar_features.append(month_sar)

                end_idx_vhd = end_idx_sar + 3
                month_vhd = data[:, end_idx_sar:end_idx_vhd]
                vhd_features.append(month_vhd)

            spectral_features = np.stack(spectral_features, axis=1).astype("float32")
            vhd_features = np.stack(vhd_features, axis=1).astype("float32")
            sar_features = np.stack(sar_features, axis=1).astype("float32")
            phenology_features = self.get_phenology_data(data)

            return spectral_features, vhd_features, sar_features,  phenology_features

        X_train_spectral, X_train_spatial, X_train_sar, X_train_phe = extract_features(train)
        X_val_spectral, X_val_spatial, X_val_sar, X_val_phe = extract_features(val)
        X_test_spectral, X_test_spatial, X_test_sar, X_test_phe = extract_features(test)

        Y_train, Y_val, Y_test = train[:, 0], val[:, 0], test[:, 0]

        X_train_augmented_spectral, X_train_augmented_spatial, X_train_augmented_sar, X_train_augmented_phe, Y_train_augmented = self.augment(
            X_train_spectral, X_train_spatial, X_train_sar, X_train_phe, Y_train)

        X_train_spectral = X_train_spectral.astype("float32")
        X_train_augmented_spectral = X_train_augmented_spectral.astype("float32")
        X_val_spectral = X_val_spectral.astype("float32")
        X_test_spectral = X_test_spectral.astype("float32")

        X_train_spatial = X_train_spatial.astype("float32")
        X_train_augmented_spatial = X_train_augmented_spatial.astype("float32")
        X_val_spatial = X_val_spatial.astype("float32")
        X_test_spatial = X_test_spatial.astype("float32")

        X_train_sar = X_train_sar.astype("float32")
        X_train_augmented_sar = X_train_augmented_sar.astype("float32")
        X_val_sar = X_val_sar.astype("float32")
        X_test_sar = X_test_sar.astype("float32")

        X_train_phe = X_train_phe.astype("float32")
        X_train_augmented_phe = X_train_augmented_phe.astype("float32")
        X_val_phe = X_val_phe.astype("float32")
        X_test_phe = X_test_phe.astype("float32")

        return (X_train_spectral, X_train_spatial, X_train_sar, X_train_phe), \
               (X_train_augmented_spectral, X_train_augmented_spatial, X_train_augmented_sar, X_train_augmented_phe), \
               Y_train, Y_train_augmented, \
               (X_val_spectral, X_val_spatial, X_val_sar, X_val_phe), Y_val, \
               (X_test_spectral, X_test_spatial, X_test_sar, X_test_phe), Y_test

    def return_data(self):
        (X_train_spectral, X_train_spatial, X_train_sar, X_train_phe), \
        (X_train_augmented_spectral, X_train_augmented_spatial, X_train_augmented_sar, X_train_augmented_phe), \
        Y_train, Y_train_augmented, \
        (X_val_spectral, X_val_spatial, X_val_sar, X_val_phe), Y_val, \
        (X_test_spectral, X_test_spatial, X_test_sar, X_test_phe), Y_test = self.get_data()

        train_dataset = [(X_train_spectral[i], X_train_spatial[i], X_train_sar[i], X_train_phe[i], int(Y_train[i]))
                         for i in range(X_train_spectral.shape[0])]

        train_augmented_dataset = [
            (X_train_augmented_spectral[i], X_train_augmented_spatial[i], X_train_augmented_sar[i], X_train_augmented_phe[i], int(Y_train_augmented[i]))
            for i in range(X_train_augmented_spectral.shape[0])]

        val_dataset = [(X_val_spectral[i], X_val_spatial[i], X_val_sar[i],  X_val_phe[i], int(Y_val[i]))
                       for i in range(X_val_spectral.shape[0])]

        test_dataset = [(X_test_spectral[i], X_test_spatial[i], X_test_sar[i], X_test_phe[i], int(Y_test[i]))
                        for i in range(X_test_spectral.shape[0])]

        return train_dataset, train_augmented_dataset, val_dataset, test_dataset


