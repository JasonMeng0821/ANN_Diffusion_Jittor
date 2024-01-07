import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import jittor as jt
import jittor.transform as TF
from PIL import Image
from scipy import linalg
# from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from jittor_inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('--save-stats', action='store_true',
                    help=('Generate an npz archive from a directory of samples. '
                          'The first path is used as input and the second as output.'))
parser.add_argument('--path', type=str, nargs=2,
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class ImagePathDataset(jt.dataset.Dataset):
    def __init__(self, files, transforms=None):
        super().__init__()
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img

def get_ref_activations(model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    dataset = np.load('images/VIRTUAL_imagenet256_labeled.npz')['arr_0']
    dataset = dataset.transpose(0, 3, 1, 2)
    # split into batches
    dataset = np.split(dataset, len(dataset) // batch_size)
    pred_arr = np.empty((10000, dims))
    spatial_arr = np.empty((10000, 2023))

    start_idx = 0

    # pred_arr
    for batch in tqdm(dataset):
        batch = jt.array(batch)

        with jt.no_grad():
            pred, spatial = model(batch)
            pred = pred[0]
            spatial = spatial[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        # if pred.size(2) != 1 or pred.size(3) != 1:
        #     pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        spatial = spatial[:, :7, :, :].reshape(spatial.shape[0], -1).cpu().numpy()
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        #spatial = spatial.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        spatial_arr[start_idx:start_idx + spatial.shape[0]] = spatial

        start_idx = start_idx + pred.shape[0]

    return pred_arr, spatial_arr

'''
FID
'''

def get_activations(files, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    dataset = ImagePathDataset(files, transforms=TF.ToTensor()).set_attrs(batch_size=batch_size, 
                                                                            shuffle=False,
                                                                            drop_last=False,
                                                                            num_workers=num_workers)
    dataloader = jt.dataset.DataLoader(dataset)

    # split into batches
    pred_arr = np.empty((len(files), dims))
    spatial_arr = np.empty((len(files), 2023))

    start_idx = 0

    # pred_arr
    for batch in tqdm(dataloader):
        batch = jt.array(batch)

        with jt.no_grad():
            pred, spatial = model(batch)
            pred = pred[0]
            spatial = spatial[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        # if pred.size(2) != 1 or pred.size(3) != 1:
        #     pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        spatial = spatial[:, :7, :, :].reshape(spatial.shape[0], -1).cpu().numpy()
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        #spatial = spatial.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        spatial_arr[start_idx:start_idx + spatial.shape[0]] = spatial

        start_idx = start_idx + pred.shape[0]

    return pred_arr, spatial_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act, act_s = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    mu_s = np.mean(act_s, axis=0)
    sigma_s = np.cov(act_s, rowvar=False)
    return mu, sigma, mu_s, sigma_s


def compute_statistics_of_path(path, model, batch_size, dims, device,
                               num_workers=1):
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
            sm, ss = f['mu_s'][:], f['sigma_s'][:]
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
        m, s, sm, ss = calculate_activation_statistics(files, model, batch_size,
                                               dims, device, num_workers)

    return m, s, sm, ss


def calculate_fid_given_paths(paths, batch_size, device, dims, num_workers=1):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])

    m1, s1, sm1, ss1 = compute_statistics_of_path(paths[0], model, batch_size,
                                        dims, device, num_workers)
    m2, s2, sm2, ss2 = compute_statistics_of_path(paths[1], model, batch_size,
                                        dims, device, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    sfid_value = calculate_frechet_distance(sm1, ss1, sm2, ss2)

    return fid_value, sfid_value


'''
Inception Score
'''
def compute_inception_score(activations, batch_size, split_size=5000):
    softmax_out = []
    for i in range(0, len(activations), batch_size):
        acts = activations[i : i + batch_size]
        softmax_output = jt.nn.softmax(jt.array(acts), dim=1)
        softmax_out.append(softmax_output.numpy())
    preds = np.concatenate(softmax_out, axis=0)
    # https://github.com/openai/improved-gan/blob/4f5d1ec5c16a7eceb206f42bfc652693601e1d5c/inception_score/model.py#L46
    scores = []
    for i in range(0, len(preds), split_size):
        part = preds[i : i + split_size]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return float(np.mean(scores))

'''
Precision and Recall
'''
def manifold_radii(activations):
    radii = []
    for i in range(activations.shape[0]):
        distances = np.linalg.norm(activations - activations[i], axis=1)
        radii.append(np.max(distances))
    return np.array(radii) # should be of shape (activations.shape[0], )

def calculate_precision_and_recall(sample_activations, ref_activations, k=3):

    radii_1 = manifold_radii(ref_activations)
    radii_2 = manifold_radii(sample_activations)

    precision = 0
    recall = 0
    for i in range(sample_activations.shape[0]):
        distances = np.linalg.norm(ref_activations - sample_activations[i], axis=1)
        distances = np.sort(distances)
        precision += np.sum(distances[:k] < radii_1[i]) / k
        recall += np.sum(distances[:k] < radii_2[i]) / k
    precision /= sample_activations.shape[0]
    recall /= sample_activations.shape[0]

    return precision, recall


def main():
    args = parser.parse_args()
    device = jt.cuda(0)
    # if args.device is None:
    #     device = jt.cuda()
    # else:
    #     device = jt.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    # if args.save_stats:
    #     save_fid_stats(args.path, args.batch_size, device, args.dims, num_workers)
    #     return
        
    path = pathlib.Path(args.path[0])
    files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
        
    # sample activations
    activations = get_activations(files, InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]]), args.batch_size, args.dims, device, num_workers)
    activations = activations[0]

    # ref activations
    ref_activations = get_ref_activations(InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]]), args.batch_size, args.dims, device, num_workers)
    ref_activations = ref_activations[0]
    # save activations
    #np.savez('activations.npz', activations)
    #np.savez('ref_activations.npz', ref_activations)

    precision, recall = calculate_precision_and_recall(activations, ref_activations, args.batch_size)
    print('Precision: ', precision)
    print('Recall: ', recall)

    fid_value, sfid_value = calculate_fid_given_paths(args.path,
                                          args.batch_size,
                                          device,
                                          args.dims,
                                          num_workers)
    print('FID: ', fid_value)
    print('SFID: ', sfid_value)

    is_score = compute_inception_score(activations, args.batch_size)
    print('IS: ', is_score)



if __name__ == '__main__':
    main()