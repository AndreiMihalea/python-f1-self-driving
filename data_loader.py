import numpy as np
import cv2


class DataLoader:
    def __init__(self, path, img_shape=(128, 52), batch_size=512, label_len=16):
        self.path = path
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.label_len = label_len
        self.train = []
        self.current_index = 0

        with open(path + 'driving_log.csv') as words_file:
            lines = words_file.readlines()

            for line in lines:
                line = line.split(',')
                center = line[0]
                acceleration, brake, steer = line[1], line[2], line[3]
                img_path = center
                self.train.append((img_path, acceleration, brake, steer))
            self.n = len(self.train)

    def get_images(self, batch):
        X = []
        y = []
        for e in batch:
            img_path = e[0]
            acceleration = e[1]
            brake = e[2]
            steer = e[3]
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, self.img_shape)
                X.append(img)
                y.append(degree)
        return np.array(X), np.array(y)

    def generate_batch(self):
        while True:
            start = self.current_index
            # See if we are at the end of the training set
            stop = min(self.current_index + self.batch_size, self.n)
            to_take = self.batch_size - (stop - start)
            batch_train = [self.train[i] for i in range(start, stop)]
            self.current_index += self.batch_size
            # Shuffle the set and take values from the beginning to complete the batch
            if to_take > 0:
                np.random.shuffle(self.train)
                self.current_index = 0
                batch_train += [self.train[i] for i in range(self.current_index, self.current_index + to_take)]
                self.current_index += to_take
            print(self.get_images(batch_train)[0].shape)
            #yield get_images(batch_train)

l = DataLoader('data/')
l.generate_batch()