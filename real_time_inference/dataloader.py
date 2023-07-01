from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import torch    

NOISE_BOUND = 0.1
IMG_DIM = 28
MNIST_PATH = r"cnn-deployment/real_time_inference/mnist"
print(f"MNIST_PATH: {MNIST_PATH}")
class DataLoader():
    """
    object for loading MNIST dataset
    """
    def __init__(self):
        self.normal_train = None
        self.normal_data = None
        self.anomaly_data = None
        
    def addNoise(self, image, noise_bound=0.3):
        noisy_image = image.copy()

        for (j, _) in enumerate(noisy_image):
            if np.random.uniform() <= noise_bound:
                noise = np.random.randint(255)
                noisy_image[j] = noise

        # fig = plt.figure()
        # fig.add_subplot(1, 2, 1)
        # plt.imshow(np.array(image).reshape(IMG_DIM, IMG_DIM))
        # fig.add_subplot(1, 2, 2)
        # plt.imshow(np.array(noisy_image).reshape(IMG_DIM, IMG_DIM))
        # noise_bound = str(noise_bound).replace(".","")
        # plt.savefig(f"Orignal_vs_corrupt_MNIST_noise_lvl_{noise_bound}_{np.random.randint(4)}.")
        # plt.close()

        return noisy_image

    def prepareMNIST(self, mnist_path, num_train_data, num_normal_data, num_anomaly_data, corrupt_train=None, noise_bound=0.3):
        mndata = MNIST(mnist_path)
        train_images, _ = mndata.load_training()

        normal_train = train_images[:num_train_data]
        normal_data = train_images[num_train_data:num_train_data + num_normal_data] 
        anomaly_data = train_images[num_train_data + num_normal_data:num_train_data + num_normal_data + num_anomaly_data] 

        normal_train = np.array(normal_train)
        # Corrupt training data
        if corrupt_train!=None:
            corrupt_data = np.array([self.addNoise((image), noise_bound=noise_bound) for image in normal_train])
            num_corrupt_data = int(len(normal_train) * corrupt_train)
            normal_train = np.concatenate([corrupt_data[:num_corrupt_data], normal_train[num_corrupt_data:]], axis=0)
        
        normal_data = np.array(normal_data)
        anomaly_data = np.array([self.addNoise((image), noise_bound=noise_bound) for image in anomaly_data])

        # Reshape and noramlize data
        normal_train = normal_train.reshape(num_train_data, 1, IMG_DIM, IMG_DIM) / 255
        normal_data = normal_data.reshape(num_normal_data, 1, IMG_DIM, IMG_DIM) / 255
        anomaly_data = anomaly_data.reshape(anomaly_data.shape[0], 1, IMG_DIM, IMG_DIM) / 255

        normal_train = torch.from_numpy(normal_train).type(torch.FloatTensor)
        normal_data = torch.from_numpy(normal_data).type(torch.FloatTensor)
        anomaly_data = torch.from_numpy(anomaly_data).type(torch.FloatTensor)

        self.normal_train = [
            normal_train, 
            torch.ones(num_train_data, dtype=torch.int64)]
        self.normal_data = [
            normal_data, 
            torch.ones(num_normal_data, dtype=torch.int64)]
        self.anomaly_data = [
            anomaly_data, 
            torch.zeros(num_anomaly_data, dtype=torch.int64)]

    def getDataLoaderMNIST(self, batch_size):
        train_images, train_labels = self.normal_train
        normal_images, normal_labels = self.normal_data
        anomaly_images, anomaly_labels = self.anomaly_data
        val_images, val_labels = torch.cat([normal_images, anomaly_images], axis=0), torch.cat([normal_labels, anomaly_labels], axis=0) 

        train = torch.utils.data.TensorDataset(train_images, train_labels)
        val = torch.utils.data.TensorDataset(val_images, val_labels)
        normal = torch.utils.data.TensorDataset(normal_images, normal_labels)
        anomaly = torch.utils.data.TensorDataset(anomaly_images, anomaly_labels)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)
        normal_loader = torch.utils.data.DataLoader(normal, batch_size=batch_size, shuffle=True)
        anomaly_loader = torch.utils.data.DataLoader(anomaly, batch_size=batch_size, shuffle=True)
        
        return train_loader, val_loader, normal_loader, anomaly_loader

if __name__ == '__main__':
    dl = DataLoader()
    dl.prepareMNIST(
        mnist_path=MNIST_PATH,
        num_train_data=100,
        num_normal_data=10,
        num_anomaly_data=10,
        corrupt_train=0.5
    )
    train_loader, val_loader, normal_loader, anomaly_loader = dl.getDataLoaderMNIST(10)
    train_images = dl.normal_train[0]
    normal_images = dl.normal_data[0]
    anomaly_images = dl.anomaly_data[0]

    for i, (images, label) in enumerate(val_loader):
        for image in images:
        # img_idx = np.random.randint(len(normal_images))
            plt.imshow(image.reshape(IMG_DIM, IMG_DIM))
            plt.show()