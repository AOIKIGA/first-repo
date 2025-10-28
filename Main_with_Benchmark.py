import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from Codes import *
import logging
import random
from datetime import datetime
from torch.utils import data
from model import ECC_Transformer
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import numpy as np
import os
from thop import profile

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Conv_Dataset(data.Dataset):
    def __init__(self, code, sigma, len, zero_cw=True):
        self.code = code
        self.sigma = sigma
        self.len = len

        self.code.blk_len += self.code.l * 4  # redundant block length, pick the middle
        G, H = get_generator_and_parity(self.code)
        print(G.shape, H.shape)
        self.generator_matrix = torch.from_numpy(G).long()
        self.parity_matrix = torch.from_numpy(H).transpose(0, 1).long()
        self.code.blk_len -= self.code.l * 4  # recover block length

        self.zero_word = torch.zeros(
            (self.code.blk_len + 6 * self.code.l - 2) * self.code.k).long() if zero_cw else None
        self.zero_cw = torch.zeros((self.code.blk_len + 6 * self.code.l - 2) * self.code.n).long() if zero_cw else None
        self.origin_m_l = (self.code.blk_len + 6 * self.code.l - 2) * self.code.k
        self.origin_n_l = (self.code.blk_len + 6 * self.code.l - 2) * self.code.n

    def __getitem__(self, index):
        if self.zero_cw is None:
            m = torch.randint(0, 2, (1, self.origin_m_l)).squeeze()
            x = torch.matmul(m, self.generator_matrix) % 2
        else:
            m = self.zero_word
            x = self.zero_cw
        z = torch.randn(self.origin_n_l) * random.choice(self.sigma)
        y = bin_to_sign(x) + z
        magnitude = torch.abs(y)
        syndrome = torch.matmul(sign_to_bin(torch.sign(y)).long(), self.parity_matrix) % 2
        return m.float()[(self.code.l * 3 - 1) * self.code.k: (self.code.l * 3 - 1 + self.code.blk_len) * self.code.k], \
            x.float()[self.code.l * 2 * self.code.n: (self.code.l * 4 - 2 + self.code.blk_len) * self.code.n], \
            z.float()[self.code.l * 2 * self.code.n: (self.code.l * 4 - 2 + self.code.blk_len) * self.code.n], \
            y.float()[self.code.l * 2 * self.code.n: (self.code.l * 4 - 2 + self.code.blk_len) * self.code.n], \
            magnitude.float()[self.code.l * 2 * self.code.n: (self.code.l * 4 - 2 + self.code.blk_len) * self.code.n], \
            syndrome.float()[
            self.code.l * 2 * (self.code.n - self.code.k): (self.code.l * 4 - 2 + self.code.blk_len) * (
                        self.code.n - self.code.k)]

    def __len__(self):
        return self.len


def train(model, device, train_loader, optimizer, epoch, LR, code=None):
    model.train()
    cum_loss = cum_ber = cum_fer = cum_samples = 0
    t = time.time()
    for batch_idx, (m, x, z, y, magnitude, syndrome) in enumerate(train_loader):
        z_mul = (y * bin_to_sign(x))[:, (code.l - 1) * code.n: (code.l - 1 + code.blk_len) * code.n]
        z_pred = model(magnitude.to(device), syndrome.to(device), device)
        y = y[:, (code.l - 1) * code.n: (code.l - 1 + code.blk_len) * code.n]
        loss, x_pred = model.loss(-z_pred, z_mul.to(device), y.to(device))
        model.zero_grad()
        loss.backward()
        optimizer.step()
        ###
        ber = BER(x_pred, x.to(device)[:, (code.l - 1) * code.n: (code.l - 1 + code.blk_len) * code.n])
        fer = FER(x_pred, x.to(device)[:, (code.l - 1) * code.n: (code.l - 1 + code.blk_len) * code.n])

        cum_loss += loss.item() * x.shape[0]
        cum_ber += ber * x.shape[0]
        cum_fer += fer * x.shape[0]
        cum_samples += x.shape[0]
        if (batch_idx + 1) % 500 == 0 or batch_idx == len(train_loader) - 1:
            logging.info(
                f'Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.2e} BER={cum_ber / cum_samples:.2e} FER={cum_fer / cum_samples:.2e}')
    logging.info(f'Epoch {epoch} Train Time {time.time() - t}s\n')
    return cum_loss / cum_samples, cum_ber / cum_samples, cum_fer / cum_samples


def test(model, device, test_loader_list, EbNo_range_test, min_FER=100, code=None, measure_time=False):
    model.eval()
    test_loss_list, test_loss_ber_list, test_loss_fer_list, cum_samples_all = [], [], [], []
    inference_times = []  # time recorder list
    total_inference_time = 0
    num_inferences = 0
    t = time.time()
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_loss = test_ber = test_fer = cum_count = 0.
            while True:
                (m, x, z, y, magnitude, syndrome) = next(iter(test_loader))
                z_mul = (y * bin_to_sign(x))[:, (code.l - 1) * code.n: (code.l - 1 + code.blk_len) * code.n]

                # 测量推理时间
                if measure_time:
                    infer_start = time.time()

                z_pred = model(magnitude.to(device), syndrome.to(device), device)

                if measure_time:
                    infer_end = time.time()
                    infer_time = infer_end - infer_start
                    inference_times.append(infer_time)
                    total_inference_time += infer_time
                    num_inferences += 1

                y = y[:, (code.l - 1) * code.n: (code.l - 1 + code.blk_len) * code.n]
                loss, x_pred = model.loss(-z_pred, z_mul.to(device), y.to(device))

                test_loss += loss.item() * x.shape[0]

                test_ber += BER(x_pred, x.to(device)[:, (code.l - 1) * code.n: (code.l - 1 + code.blk_len) * code.n]) * \
                            x.shape[0]
                test_fer += FER(x_pred, x.to(device)[:, (code.l - 1) * code.n: (code.l - 1 + code.blk_len) * code.n]) * \
                            x.shape[0]
                cum_count += x.shape[0]
                if (min_FER > 0 and test_fer > min_FER and cum_count > 1e5) or cum_count >= 1e9:
                    if cum_count >= 1e9:
                        print(f'Number of samples threshold reached for EbN0:{EbNo_range_test[ii]}')
                    else:
                        print(f'FER count threshold reached for EbN0:{EbNo_range_test[ii]}')
                    break
            cum_samples_all.append(cum_count)
            test_loss_list.append(test_loss / cum_count)
            test_loss_ber_list.append(test_ber / cum_count)
            test_loss_fer_list.append(test_fer / cum_count)
            print(f'Test EbN0={EbNo_range_test[ii]}, BER={test_loss_ber_list[-1]:.2e}')
        ###
        logging.info('\nTest Loss ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_list, EbNo_range_test))]))
        logging.info('Test FER ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_fer_list, EbNo_range_test))]))
        logging.info('Test BER ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_ber_list, EbNo_range_test))]))
        logging.info('Test -ln(BER) ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, -np.log(elem)) for (elem, ebno)
             in
             (zip(test_loss_ber_list, EbNo_range_test))]))

        # benchmarking
        if measure_time and num_inferences > 0:
            avg_infer_time = total_inference_time / num_inferences
            std_infer_time = np.std(inference_times) if num_inferences > 1 else 0
            logging.info(f'Inference Time Statistics - Average: {avg_infer_time:.6f}s, Std: {std_infer_time:.6f}s, '
                         f'Total: {total_inference_time:.2f}s, Count: {num_inferences}')
            logging.info(f'Inference Throughput: {num_inferences / total_inference_time:.2f} samples/second')

    logging.info(f'# of testing samples: {cum_samples_all}\n Test Time {time.time() - t} s\n')
    return test_loss_list, test_loss_ber_list, test_loss_fer_list


def load_best_model(args, code, device):
    model_path = os.path.join(args.path, 'best_model')
    logging.info(f"Loading best model from: {model_path}")


    with torch.serialization.safe_globals([ECC_Transformer]):
        if torch.cuda.is_available() and device.type == 'cuda':
            model = torch.load(model_path, weights_only=False)                                    # !!!: Don't change this False
        else:
            model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)  # !!!: Don't change this False


    model = model.to(device)
    logging.info("Best model loaded successfully")
    return model


def main(args):
    code = args.code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EbNo_range_train = range(2, 8)
    EbNo_range_test = range(-1, 9)
    std_train = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_train]
    std_test = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_test]

    # testloader only
    test_dataloader_list = [DataLoader(Conv_Dataset(code, [std_test[ii]], len=int(args.test_batch_size), zero_cw=False),
                                       batch_size=int(args.test_batch_size),
                                       shuffle=False, num_workers=args.workers)
                            for ii in range(len(std_test))]

    # load best model
    model = load_best_model(args, code, device).to(device)
    magn = Conv_Dataset(code,[EbN0_to_std(3,0.5)], len=int(args.test_batch_size), zero_cw=False).__getitem__(2)[4]
    synd = Conv_Dataset(code,[EbN0_to_std(3,0.5)], len=int(args.test_batch_size), zero_cw=False).__getitem__(2)[5]
    flops, params = profile(model, inputs=(magn.to(device),synd.to(device),))
    print(flops)
    print(params)
    # run test
    #test(model, device, test_dataloader_list, EbNo_range_test, code=code, measure_time=True)


def original_main(args):
    code = args.code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EbNo_range_train = range(2, 8)
    EbNo_range_test = range(-1, 10)
    std_train = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_train]
    std_test = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_test]

    train_dataloader = DataLoader(Conv_Dataset(code, std_train, len=args.batch_size * 1000, zero_cw=False),
                                  batch_size=int(args.batch_size),
                                  shuffle=True, num_workers=args.workers)
    test_dataloader_list = [DataLoader(Conv_Dataset(code, [std_test[ii]], len=int(args.test_batch_size), zero_cw=False),
                                       batch_size=int(args.test_batch_size),
                                       shuffle=False, num_workers=args.workers)
                            for ii in range(len(std_test))]

    model = ECC_Transformer(args, dropout=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    logging.info(model)
    logging.info(f'# of Parameters: {np.sum([np.prod(p.shape) for p in model.parameters()])}')

    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        loss, ber, fer = train(model, device, train_dataloader, optimizer,
                               epoch, LR=scheduler.get_last_lr()[0], code=code)
        scheduler.step()
        if loss < best_loss:
            best_loss = loss
            torch.save(model, os.path.join(args.path, 'best_model'))
        if epoch % 300 == 0 or epoch in [1, args.epochs]:
            test(model, device, test_dataloader_list, EbNo_range_test, code=code, measure_time=True)


if __name__ == '__main__':
    torch.set_printoptions(linewidth=10000)
    parser = ArgumentParser(description='Pytorch Convolutional Code Decoding Transformer - Test with Best Model')

    #! new features: benchmarking arguments
    parser.add_argument('--test_only', action='store_true', help='Only run testing with best model')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model directory (if different from logging)')


    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpus', type=str, default='-1', help='gpus ids')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)

    # Code args
    parser.add_argument('--code_type', type=str, default='Conv')
    parser.add_argument('--code_n', type=int, default=2)
    parser.add_argument('--code_k', type=int, default=1)
    parser.add_argument('--code_l', type=int, default=3)
    parser.add_argument('--code_blk_len', type=int, default=100)

    # model args
    parser.add_argument('--N_dec', type=int, default=6)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--h', type=int, default=8)

    args = parser.parse_args()
    set_seed(args.seed)


    class Code():
        pass


    code = Code()

    code.n = args.code_n
    code.k = args.code_k
    code.l = args.code_l
    code.blk_len = args.code_blk_len
    code.type = args.code_type

    torch.set_printoptions(threshold=float('inf'))

    G, H = get_generator_and_parity(code)
    print(type(G), isinstance(G, np.ndarray))

    code.generator = torch.from_numpy(G).transpose(0, 1).long()
    print(type(code.generator), isinstance(code.generator, torch.Tensor))
    code.parity = torch.from_numpy(H).long()
    args.code = code

    if args.model_path:
        args.path = args.model_path
    else:
        model_dir = os.path.join('Logging_CCDT', args.code_type +
                                 '_n_' + str(args.code_n) +
                                 '_k_' + str(args.code_k) +
                                 '_l_' + str(args.code_l) +
                                 '_blk_len_' + str(args.code_blk_len) +
                                 '__' + datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
        os.makedirs(model_dir, exist_ok=True)
        args.path = model_dir

    handlers = [logging.FileHandler(os.path.join(args.path, 'testing_logging.txt'))]
    handlers += [logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)
    logging.info(f"Path to model_logs: {args.path}")
    logging.info(args)

    if args.test_only:
        main(args)                      # benchmarking with best model only
    else:
        original_main(args)             # original Training and Testing
