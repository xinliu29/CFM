import yaml
import shutil
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim

from eval.eval_full import evaluate_full


class Trainer:
    def __init__(self, model, train_loader, eval_loader=None, args=None):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.args = args

        self.base_dir = args.base_dir
        self.save_path = args.save_path
        self.resume_path = args.resume_path

        self.num_epochs = args.epochs
        self.init_lr = args.lr
        self.min_lr = args.min_lr

        if args.optim == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.init_lr)
        elif args.optim == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.init_lr)
        else:
            raise ValueError(f"Unknown optimizer: {args.optim}")

        if self.resume_path != 'None':
            log_dir = self.resume_path.split('/')[-2]
            resume_log = torch.load(self.save_path / self.resume_path, map_location='cpu')
            self.epoch = resume_log['epoch'] + 1
            self.iteration = resume_log['iteration'] if 'iteration' in resume_log.keys() else len(
                self.train_loader) * self.epoch
            self.min_loss = resume_log['min_loss']
        else:
            import datetime
            now = datetime.datetime.now()
            log_dir = f"{now.strftime('%Y_%m_%d_%H_%M_%S')}_{args.network}_{args.feature}"
            self.epoch = 0
            self.iteration = 0
            self.min_loss = 1e10

        self.tag = log_dir
        self.save_dir = self.save_path / self.tag
        self.save_dir.mkdir(exist_ok=True)

        self.log_file = open(self.save_dir / "log.txt", "a+")

        if args.local_rank == 0:
            self.log_file.write(f'[network:{args.network}]_[feature:{args.feature}]\n'
                                f'[use_prune:{args.use_prune}]_[n_min_tokens:{args.n_min_tokens}]\n'
                                f'[threshold:{args.threshold}]_[layer_prune:{args.layer_prune}]\n'
                                # f'[filter_prune:{args.filter_prune}]_[use_2nd_nn:{args.use_2nd_nn}]\n'
                                )
            print(f'save_dir: {self.save_dir}')
            print(f'Start to train the model from epoch: {self.epoch}')
            with open(self.save_dir / 'args.yaml', 'w') as outfile:
                yaml.dump(args, outfile, default_flow_style=False)

    def process_epoch(self):
        self.model.train()

        epoch_losses = []
        epoch_matching_loss = []
        epoch_filter_loss = []
        epoch_matching_scores = []

        n_invalid_its = 0
        for bidx, pred in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            for k in pred:
                if k not in ['image0', 'image1', 'depth0', 'depth1'] and not (k.find('file_name') >= 0):
                    if type(pred[k]) == torch.Tensor:
                        pred[k] = pred[k].float().cuda()
                    else:
                        pred[k] = torch.stack(pred[k]).float().cuda()

            data = self.model(pred)

            loss = data['loss']
            matching_loss = data['matching_loss'] if 'matching_loss' in data.keys() else torch.zeros(1)
            filter_loss = data['filter_loss'] if 'filter_loss' in data.keys() else torch.zeros(1)
            matching_score = data['matching_scores0'][-1] if 'matching_scores0' in data.keys() else torch.zeros(1)

            if torch.numel(loss) > 1:
                loss = torch.mean(loss)
                matching_loss = torch.mean(matching_loss)
                filter_loss = torch.mean(filter_loss)

            if torch.isinf(loss) or torch.isnan(loss):
                print('Loss is INF/NAN', loss)
                self.optimizer.zero_grad()
                n_invalid_its += 1
                if n_invalid_its >= 10:
                    print('Exit because of INF/NAN in loss')
                    return None
                continue

            epoch_losses.append(loss.item())
            epoch_matching_loss.append(matching_loss.item())
            epoch_filter_loss.append(filter_loss.item())
            epoch_matching_scores.append(torch.max(matching_score).item())

            self.optimizer.zero_grad()
            loss.backward()
            nan_detected = any(
                param.grad is not None and torch.isnan(param.grad).any() for param in self.model.parameters())
            if nan_detected:
                print("NaN values found in gradients after backward pass. Skipping this iteration.")
                self.optimizer.zero_grad()
                continue
            self.optimizer.step()

            self.iteration += 1

            lr = min(self.args.lr * self.args.decay_rate ** (self.iteration - self.args.decay_iter), self.args.lr)
            if lr < self.min_lr:
                lr = self.min_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        if self.args.local_rank == 0:
            print_text = f'Epoch [{self.epoch}/{self.num_epochs}], ' \
                         f'AVG Loss [m:{np.mean(epoch_matching_loss):.2f}/f:{np.mean(epoch_filter_loss):.2f}' \
                         f'/t:{np.mean(epoch_losses):.2f}], ' \
                         f'matching_score[{np.mean(epoch_matching_scores):.2f}]\n'
            self.log_file.write(print_text + '\n')
            self.log_file.flush()
            print(print_text)

        return np.mean(epoch_losses)

    def eval_matching(self):
        self.model.eval()
        with torch.no_grad():
            for dataset in ['yfcc', 'scannet']:
                eval_out = evaluate_full(model=self.model,
                                         feat_type=self.args.feature,
                                         dataset=dataset,
                                         base_dir=self.base_dir)
                text = f"Eval Epoch [{self.epoch}] for {dataset}"
                for k in eval_out.keys():
                    text = text + f" {k} [{eval_out[k]:.2f}]"
                self.log_file.write(text + "\n\n")
                self.log_file.flush()

    def train(self):
        hist_values = []
        min_value = self.min_loss
        self.model.module.use_prune = True if self.args.use_prune and self.epoch > 25 else False
        if self.args.local_rank == 0:
            print("use prune:", self.model.module.use_prune)

        # if self.args.local_rank == 0:
        #     self.eval_matching()
        #     exit(1)

        while self.epoch < self.num_epochs:
            self.train_loader.sampler.set_epoch(epoch=self.epoch)

            train_loss = self.process_epoch()

            if self.args.local_rank == 0:
                if self.epoch % 25 == 0:
                    self.eval_matching()

                if self.args.use_prune and self.epoch == 25:
                    self.model.module.use_prune = True

                hist_values.append(train_loss)
                checkpoint_path = self.save_dir / f'{self.args.network}.{self.epoch:02d}.pth'
                checkpoint = {
                    'epoch': self.epoch,
                    'iteration': self.iteration,
                    'model': self.model.state_dict() if len(self.args.gpu) == 1 else self.model.module.state_dict(),
                    'min_loss': min_value,
                }
                torch.save(checkpoint, checkpoint_path)

                if hist_values[-1] < min_value:
                    min_value = hist_values[-1]
                    best_checkpoint_path = self.save_dir / f'{self.tag}.best.pth'
                    shutil.copy(checkpoint_path, best_checkpoint_path)

            self.epoch += 1
            self.train_loader.dataset.build_dataset(seed=self.epoch)

        if self.args.local_rank == 0:
            self.log_file.close()
