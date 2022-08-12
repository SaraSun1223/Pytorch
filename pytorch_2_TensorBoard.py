import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

image_path = 'dataset/catvsdog/dog/dog.23.jpg' # 图像目录
img_PIL = Image.open(image_path)  # 打开图片文件（PILimage）
img_array = np.array(img_PIL)    # 转成numpy格式

print(type(img_array))
print(img_array.shape)  # (403, 499, 3)

writer.add_image('dog', img_array, 1, dataformats='HWC')

x = range(100)
for i in x:
    writer.add_scalar('y=2x', i * 2, i)

writer.close()
