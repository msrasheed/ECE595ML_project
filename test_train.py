import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torch
from skimage import io
import numpy as np
import glob 
import tqdm
import unet_model

class Resize:
	def __init__(self, shape):
		self.shape = shape

	def __call__(self, sample):
		if sample.shape[0] != self.shape[0]:
			sample = sample.transpose(1,0,2)
		return sample

class Normalize:
	def __call__(self, sample):
		return sample / 255

class ToTensor:
	def __call__(self, sample):
		sample = sample.transpose((2,0,1))
		return torch.as_tensor(sample, dtype=torch.float)

class MyDataset(Dataset):
	def __init__(self, path, transform=ToTensor()):
		self.path = path
		self.files = glob.glob(self.path + '/*')
		self.transform = transform

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		imgfile = self.files[idx]
		image = io.imread(imgfile)
		image = self.transform(image)
		# noisey1 = torch.normal(image, 50/255.)
		noisey1 = torch.normal(image, 50/255)
		# noisey2 = torch.normal(image, 50/255.)
		noisey2 = torch.normal(image, 50/255)
		return image, noisey1, noisey2

def imagesToPsnr(img1, img2):
	B = img1.shape[0]
	mse = F.mse_loss(img1, img2, reduction='none').reshape(B, -1)
	mse = torch.mean(mse,1).detach().numpy() + 1e-16
	psnr = 10 * np.log10(1/mse)
	return psnr


def train_loop(model, data, optimizer, device):
	for clean_image, noisey_image1, noisey_image2 in data:
		noisey_image1 = noisey_image1.to(device=device)
		noisey_image2 = noisey_image2.to(device=device)
		optimizer.zero_grad()

		reconstructed_image = model(noisey_image1)

		loss = F.mse_loss(reconstructed_image, noisey_image2)

		loss.backward()
		optimizer.step()
	psnr = imagesToPsnr(clean_image, reconstructed_image.to(device='cpu'))
	psnr = np.mean(psnr)
	return loss.to('cpu').detach().numpy(), psnr

def main():
	num_epochs = 30
	dev = torch.device('cuda')

	model = unet_model.UNet(n_channels=3, n_classes=3, bilinear=True).to(device=dev)
	dataset = MyDataset('kodak', 
                            transform=transforms.Compose([Resize((512,768)),Normalize(),ToTensor()]))
	dataloader = DataLoader(dataset, batch_size=4,
	                        shuffle=True, num_workers=0,
				pin_memory=True)
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

	loss = np.zeros(num_epochs)
	psnr = np.zeros(num_epochs)
	for epoch in tqdm.tqdm(range(num_epochs)):
		loss[epoch], psnr[epoch] = train_loop(model, dataloader, optimizer, dev)
		print(loss[epoch], psnr[epoch])
		scheduler.step()

	torch.save(model.state_dict(), "model_weights.pt")
	data = np.vstack((np.arange(num_epochs), loss, psnr)).T
	np.savetxt("train_data.dat", data)

if __name__ == "__main__":
	main()
