import numpy as np
import torch
import torch.nn as nn


def gen_loss(DGz):
    return DGz ** 2


def discr_loss_g(Dg, yg):
    return yg * (Dg - 1) ** 2 + (1 - yg) * (Dg + 1) ** 2


def discr_loss_z(DGz):
    return (DGz + 1) ** 2


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
    def __init__(self, noise_size, gen_hiddens, gen_outputs, discr_hiddens, discr_outputs, gen_lr, discr_lr,
                 batch_size):
        self.noise_size = noise_size
        self.batch_size = batch_size

        # 定义网络
        self.gen = Generator(self.noise_size, gen_hiddens, gen_outputs)
        self.discr = Discriminator(gen_outputs, discr_hiddens, discr_outputs)

        # 定义损失函数
        self.loss_gen = gen_loss
        self.loss_discr_g = discr_loss_g
        self.loss_discr_z = discr_loss_z

        # 优化器
        self.gen_optimizer = torch.optim.Adam(self.gen.parameters(), lr=gen_lr)
        self.discr_optimizer = torch.optim.Adam(self.discr.parameters(), lr=discr_lr)

    def sample_random_noise(self, size):
        return np.random.randn(size, self.noise_size)

    def sample_generator(self, size):
        generator_sample = []
        generator_noise = []
        batch_size = self.batch_size
        for i in range(0, size, batch_size):
            sample_size = min(batch_size, size - i)  # 防止不整除
            noise = self.sample_random_noise(sample_size)  # 生成噪声（gen的输入）
            generator_noise.append(noise)  # 存噪声
            generator_sample.append(self.gen(noise))  # 生成goals并存

        return generator_noise, generator_sample

    def train(self):
        # TODO
        pass

    def train_discriminator(self, goals, labels):
        """
        :param goals: goal that we know labels of
        :param labels: labels of those goals
        The batch size is given by the configs of the class!
        discriminator_batch_noise_stddev > 0: check that std on each component is at least this. (if com: 2)
        """
        assert goals.shape[0] == labels.shape[0], "goals.shape[0] != goals.shape[0]"

        loss = self.loss_discr_g(goals, labels) + self.loss_discr_z(self.generator_output)
        self.discr_optimizer.zero_grad()  # 清空过往梯度
        loss.backward()  # 反向传播，计算当前梯度
        self.discr_optimizer.step()  # 根据梯度更新网络参数

        return loss
