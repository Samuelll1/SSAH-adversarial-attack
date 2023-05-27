import torch
import torch.nn as nn
from utils.auxiliary_utils import *
import torch.nn.functional as F
from attack.DWT import *
from torch.autograd import Variable
import torch.optim as optim


class SSAH(nn.Module):
    """
    SSAH 类，继承自 nn.Module
    """

    def __init__(self,
                 model: nn.Module,
                 num_iteration: int = 150,
                 learning_rate: float = 0.001,
                 device: torch.device = torch.device('cuda'),
                 Targeted: bool = False,
                 dataset: str = 'cifar10',
                 m: float = 0.2,
                 alpha: float = 1,
                 lambda_lf: float = 0.1,
                 wave: str = 'haar',) -> None:
        """
        构造函数
        :param model: 模型
        :param num_iteration: 迭代次数
        :param learning_rate: 学习率
        :param device: 设备
        :param Targeted: 目标类型
        :param dataset: 数据集名称
        :param m: m 值
        :param alpha: alpha 值
        :param lambda_lf: lambda_lf 值
        :param wave: 小波名称
        """
        super(SSAH, self).__init__()
        self.model = model
        self.device = device
        self.lr = learning_rate
        self.target = Targeted
        self.num_iteration = num_iteration
        self.dataset = dataset
        self.m = m
        self.alpha = alpha
        self.lambda_lf = lambda_lf

        # 初始化模型和变量
        self.encoder_fea = nn.Sequential(*list(self.model.children())[:-1]).to(self.device)
        self.encoder_fea= nn.DataParallel(self.encoder_fea)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)).to(self.device)
        self.model = nn.DataParallel(self.model)

        # 初始化归一化函数和小波变换函数
        self.normalize_fn = normalize_fn(self.dataset)
        self.DWT = DWT_2D_tiny(wavename= wave)
        self.IDWT = IDWT_2D_tiny(wavename= wave)

    def fea_extract(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        提取特征函数
        :param inputs: 输入张量
        :return: 特征张量
        """
        fea = self.encoder_fea(inputs)
        b, c, h, w = fea.shape
        fea = self.avg_pool(fea).view(b, c)
        return fea

    def cal_sim(self, adv, inputs):
      """
      计算相似度函数
      :param adv: 广告张量
      :param inputs: 输入张量
      :return: 正相似度和负相似度张量
      """
      adv = F.normalize(adv, dim=1)
      inputs = F.normalize(inputs, dim=1)

      r, c = inputs.shape
      sim_matrix = torch.matmul(adv, inputs.T)
      mask = torch.eye(r, dtype=torch.bool).to(self.device)
      pos_sim = sim_matrix[mask].view(r, -1)
      neg_sim = sim_matrix.view(r, -1)
      return pos_sim, neg_sim

    def select_setp1(self, pos_sim, neg_sim):
      """
      选择步骤 1 函数，选择最不相似的样本并返回正负相似度和索引。
      :param pos_sim: 正相似度张量
      :param neg_sim: 负相似度张量
      :return: 正负相似度和索引张量。
      """
      neg_sim, indices = torch.sort(neg_sim, descending=True)
      pos_neg_sim = torch.cat([pos_sim, neg_sim[:, -1].view(pos_sim.shape[0], -1)], dim=1)
      return pos_neg_sim, indices

    def select_step2(self, pos_sim, neg_sim, indices):
      """
      选择步骤 2 函数，根据索引选择最不相似的样本并返回正负相似度。
      :param pos_sim: 正相似度张量。
      :param neg_sim: 负相似度张量。
      :param indices: 索引张量。
      :return: 正负相似度张量。
      """
      hard_sample = indices[:, -1]
      ones = torch.sparse.torch.eye(neg_sim.shape[1]).to(self.device)
      hard_one_hot = ones.index_select(0, hard_sample).bool()
      hard_sim = neg_sim[hard_one_hot].view(neg_sim.shape[0], -1)
      pos_neg_sim = torch.cat([pos_sim, hard_sim], dim=1)
      return pos_neg_sim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
       """
       前向传播函数，实现优化算法。
       :param inputs: 输入张量。
       :return: 广告张量。
       """
       with torch.no_grad():
           inputs_fea = self.fea_extract(self.normalize_fn(inputs))

       # 计算低频分量。
       inputs_ll = self.DWT(inputs)
       inputs_ll = self.IDWT(inputs_ll)

        # 变量修改器和优化器。
        eps = 3e-7
        modifier = torch.arctanh(inputs * (2 - eps * 2) - 1 + eps)
        modifier = Variable(modifier, requires_grad=True)
        modifier = modifier.to(self.device)
        optimizer = optim.Adam([modifier], lr=self.lr)

        # 定义损失函数。
        lowFre_loss = nn.SmoothL1Loss(reduction='sum')

        # 迭代优化。
        for step in range(self.num_iteration):
            optimizer.zero_grad()
            self.encoder_fea.zero_grad()

            adv = 0.5 * (torch.tanh(modifier) + 1)
            adv_fea = self.fea_extract(self.normalize_fn(adv))

            adv_ll = self.DWT(adv)
            adv_ll = self.IDWT(adv_ll)

            pos_sim, neg_sim = self.cal_sim(adv_fea, inputs_fea)
            # 在第一次迭代中选择最不相似的样本。
            if step == 0:
                pos_neg_sim, indices = self.select_setp1(pos_sim, neg_sim)

            # 根据索引记录最不相似的样本并计算相似度。
            else:
                pos_neg_sim = self.select_step2(pos_sim, neg_sim, indices)

            sim_pos = pos_neg_sim[:, 0]
            sim_neg = pos_neg_sim[:, -1]

            w_p = torch.clamp_min(sim_pos.detach() - self.m, min=0)
            w_n = torch.clamp_min(1 + self.m - sim_neg.detach(), min=0)

            adv_cost = torch.sum(torch.clamp(w_p * sim_pos - w_n * sim_neg, min=0))
            lowFre_cost = lowFre_loss(adv_ll, inputs_ll)
            total_cost = self.alpha * adv_cost + self.lambda_lf * lowFre_cost

            optimizer.zero_grad()
            total_cost.backward()
            optimizer.step()

        adv = 0.5 * (torch.tanh(modifier.detach()) + 1)
        return adv
