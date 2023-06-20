class DroneDataset(Dataset):
    def __init__(self, image_dir, mask_dir, n_class=14, transform = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # self.mean = mean # for normalization
        # self.std = std # for normalization
        self.n_class = n_class # number of classes
        self.images = os.listdir(image_dir) # list all the files in that folder

    def __len__(self):
        return len(self.images)

    def onehot(self, img, nb):
        oh = np.zeros((img.shape[0], img.shape[1], nb))
        for i in range(nb):
            oh[:,:,i] = (img[:,:,0] == i)
        return oh

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))

        t = T.Compose([T.ToTensor(), T.Normalize(0, 1)]) # normalize the data

        image = np.array(Image.open(img_path).convert("RGB"))
        # mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # L: for PIL grey, and the numbers should be from 0 to 13
        mask = np.array(Image.open(mask_path))


        # mask[mask == 255.0] = 1.0 # use sigmoid on R

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        # if self.transform is None:
        #     image = image
        mask = self.onehot(mask, self.n_class)
        image = t(image) # normalization of images

        return image, mask # in the form of numpy image
