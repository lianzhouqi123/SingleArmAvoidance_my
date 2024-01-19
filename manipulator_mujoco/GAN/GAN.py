import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def batch_feed_array(array, batch_size):
    data_size = array.shape[0]
    # assert data_size >= batch_size

    if data_size <= batch_size:  # 有的少于要的，全取
        while True:
            yield array  # 生成器
    else:
        start = 0
        while True:  # 有的多余要的，轮着取
            if start + batch_size < data_size:  # 剩下的不用循环
                yield array[start:start + batch_size, :]
            else:  # 剩下的要循环，取完再从头
                yield torch.cat(
                    [array[start:data_size, :], array[0: start + batch_size - data_size, :]],
                    dim=0
                )
            start = (start + batch_size) % data_size


class Generator(nn.Module):
    def __init__(self, n_noise, n_hiddens, n_outputs):
        super(Generator, self).__init__()
        self.lin1 = nn.Linear(n_noise, n_hiddens)
        self.act1 = nn.LeakyReLU(0.01)
        self.lin2 = nn.Linear(n_hiddens, n_hiddens)
        self.act2 = nn.LeakyReLU(0.01)
        self.lin3 = nn.Linear(n_hiddens, n_hiddens)
        self.act3 = nn.LeakyReLU(0.01)
        self.lin4 = nn.Linear(n_hiddens, n_outputs)

    def forward(self, x):
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = self.lin3(x)
        x = self.act3(x)
        x = self.lin4(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, n_inputs, n_hiddens, n_outputs):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(n_inputs, n_hiddens)
        self.act1 = nn.LeakyReLU(0.01)
        self.lin2 = nn.Linear(n_hiddens, n_hiddens)
        self.act2 = nn.LeakyReLU(0.01)
        self.lin3 = nn.Linear(n_hiddens, n_hiddens)
        self.act3 = nn.LeakyReLU(0.01)
        self.lin4 = nn.Linear(n_hiddens, n_outputs)

    def forward(self, x):
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = self.lin3(x)
        x = self.act3(x)
        x = self.lin4(x)

        return x


class GAN:
    def __init__(self, noise_size, gen_n_hiddens, gen_n_outputs, discr_n_hiddens, discr_n_outputs, gen_lr, discr_lr,
                 device):
        self.noise_size = noise_size
        # self.batch_size = batch_size
        self.discr_n_outputs = discr_n_outputs
        self.device = device

        # 定义网络
        self.gen = Generator(self.noise_size, gen_n_hiddens, gen_n_outputs).to(self.device)
        self.discr = Discriminator(gen_n_outputs, discr_n_hiddens, self.discr_n_outputs).to(self.device)

        # 优化器
        self.gen_optimizer = torch.optim.RMSprop(self.gen.parameters(), lr=gen_lr)
        self.discr_optimizer = torch.optim.RMSprop(self.discr.parameters(), lr=discr_lr)

    def sample_random_noise(self, size):
        return torch.randn(size, self.noise_size).to(self.device)

    def sample_generator(self, size):
        # generator_sample = torch.tensor([])
        # generator_noise = torch.tensor([])
        # batch_size = self.batch_size
        # for i in range(0, size, batch_size):
        #     sample_size = min(batch_size, size - i)  # 防止不整除
        #     noise = self.sample_random_noise(sample_size)  # 生成噪声（gen的输入）
        #     generator_noise.append(noise)  # 存噪声
        #     generator_sample.append(self.gen(noise))  # 生成goals并存
        noise = self.sample_random_noise(size)  # 生成噪声（gen的输入）
        generator_noise = torch.clone(noise)  # 存噪声
        generator_sample = torch.clone(self.gen(noise))  # 生成goals并存

        return generator_sample, generator_noise

    def train(self, goals_input, labels_input, batch_size, outer_iters=1):
        """
        :param goals_input:
        :param labels_input:
        :param batch_size: 每次训练的batch_size
        :param outer_iters: 总循环数
        :return:
        """
        input_size = goals_input.shape[0]
        if batch_size > input_size:
            batch_size = input_size
        for ii in range(outer_iters):
            # 随机取样进行训练
            sample_i = random.sample(range(input_size), batch_size)
            goals_sample = goals_input[sample_i, :].to(self.device)
            labels_sample = labels_input[sample_i, :].to(self.device)

            # 生成器生成数据，由优化函数，长度需与batch_size相等
            generated_goals, random_noise = self.sample_generator(batch_size)  # 生成器随机生成的值
            generated_labels = torch.zeros((batch_size, self.discr_n_outputs)).to(self.device)

            # 将真实数据与生成数据混在一起
            train_X = torch.vstack([goals_sample, generated_goals.to(self.device)])  # 沿 axis = 0 堆叠，输入的goal采样+generator生成
            train_Y = torch.vstack([labels_sample, generated_labels])  # 输入的label采样+0

            # 更新discriminator
            D_train_X = self.discr(train_X)  # [Dg, D(Gz)]
            # ( 2 * yg - 1 - Dg) ** 2 + D(Gz) ** 2  经推导和论文中一样 Eg [yg(Dg-1)**2 + (1-yg)*(Dg+1)**2] + Ez [D(Gz)+1)**2]
            discriminator_loss = F.mse_loss(2 * train_Y - 1 - D_train_X, torch.zeros_like(D_train_X))

            self.discr_optimizer.zero_grad()  # 清空过往梯度
            discriminator_loss.backward()  # 反向传播，计算当前梯度
            self.discr_optimizer.step()  # 根据梯度更新网络参数

            # 更新generator
            generator_output = self.discr(self.gen(random_noise))  # Gz
            generator_loss = F.mse_loss(generator_output, torch.ones_like(generator_output))  # ( D(Gz) - 1) ** 2

            self.gen_optimizer.zero_grad()  # 清空过往梯度
            generator_loss.backward()  # 反向传播，计算当前梯度
            self.gen_optimizer.step()  # 根据梯度更新网络参数

        return discriminator_loss, generator_loss

    # 用discriminator预测输入
    def discriminator_predict(self, X):
        output = torch.tensor([], dtype=torch.float32).cuda()
        # for i in range(0, X.shape[0], self.batch_size):
        #     sample_size = min(self.batch_size, X.shape[0] - i)
        #     torch.cat([output, self.discr(X[i:i + sample_size]).detach()], dim=0)
        output = torch.cat([output, self.discr(X).detach()], dim=0)
        return output
