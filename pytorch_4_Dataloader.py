import torchvision
#将数据一批一批送给GPU
train_set = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root='./CIFAR10',train=False, transform=torchvision.transforms.ToTensor(),download=True)


img,target = test_set[0]
print(img.shape)
print(target)
