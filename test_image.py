import torch
from dataset import TestDataset
from utils import colorEncode
from scipy.io import loadmat
from types import SimpleNamespace

# test on a given image
def segmentation_test(test_image_name, options):
    dataset_test = TestDataset([{'fpath_img': test_image_name}], options, max_sample=-1)

    batch_data = dataset_test[0]
    segSize = (batch_data['img_ori'].shape[0], batch_data['img_ori'].shape[1])
    img_resized_list = batch_data['img_data']

    scores = torch.zeros(1, options.num_class, segSize[0], segSize[1])
    if torch.cuda.is_available():
        scores = scores.cuda()

    for img in img_resized_list:
        feed_dict = batch_data.copy()
        feed_dict['img_data'] = img
        del feed_dict['img_ori']
        del feed_dict['info']
    if torch.cuda.is_available():
        feed_dict = {k: o.cuda() for k, o in feed_dict.items()}

    # forward pass
    pred_tmp = segmentation_module(feed_dict, segSize=segSize)
    scores = scores + pred_tmp / len(options.imgSizes)

    _, pred = torch.max(scores, dim=1)
    return pred.squeeze(0).cpu().numpy()