import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F


# Constants
N_PAIR = 'n-pair'
ANGULAR = 'angular'
N_PAIR_ANGULAR = 'n-pair-angular'
MAIN_LOSS_CHOICES = (N_PAIR, ANGULAR, N_PAIR_ANGULAR)

CROSS_ENTROPY = 'cross-entropy'


def cross_entropy(logits, target, size_average=True):
    if size_average:
        return torch.mean(torch.sum(- target * F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(- target * F.log_softmax(logits, -1), -1))

class NpairLoss(nn.Module):
    """the multi-class n-pair loss"""
    def __init__(self, l2_reg=0.02):
        super(NpairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, anchor, positive, target):
        batch_size = anchor.size(0)
        target = target.view(target.size(0), 1)

        target = (target == torch.transpose(target, 0, 1)).float()
        target = target / torch.sum(target, dim=1, keepdim=True).float()

        logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))
        loss_ce = cross_entropy(logit, target)
        l2_loss = torch.sum(anchor**2) / batch_size + torch.sum(positive**2) / batch_size

        loss = loss_ce + self.l2_reg*l2_loss*0.25
        return loss


def get_angular_loss(input_labels, anchor_features, pos_features, degree=45):
    
    alpha = np.deg2rad(degree)
    sq_tan_alpha = np.float(np.tan(alpha) ** 2)
    

    # 2(1+(tan(alpha))^2 * xaTxp)
    #batch_size = 10
    batch_size = input_labels.shape[0]
    xaTxp = torch.matmul(anchor_features, pos_features.t())
    sim_matrix_1 = 2.0 * (1.0 + sq_tan_alpha) * xaTxp.float() * torch.eye(batch_size).cuda()
    
    # 4((tan(alpha))^2(xa + xp)Txn
    xaPxpTxn = torch.matmul((anchor_features + pos_features), pos_features.t())
    sim_matrix_2 = (4.0 * sq_tan_alpha * xaPxpTxn).float() * (torch.ones_like(xaPxpTxn).float() - torch.eye(batch_size).float().cuda())
    similarity_matrix = sim_matrix_1 + sim_matrix_2

    labels = input_labels.unsqueeze(1)
    N = labels.shape[0]
    labels_remapped = labels.expand(N, N).eq(labels.expand(N, N).t()).float()

    temp = torch.sum(labels_remapped, dim=1, keepdim=True)
    labels_remapped /= temp
    
    loss = torch.sum(- labels_remapped * F.log_softmax(similarity_matrix, -1), -1)
    xent_loss = loss.mean()

    '''
    p = F.softmax(similarity_matrix)
    h = -labels_remapped.cuda() * torch.log(p)
    xent_loss = torch.sum(h, dim=1)
    xent_loss = torch.mean(xent_loss) 
    '''

    return xent_loss


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output



class BlendedLoss(object):
    def __init__(self, main_loss_type, cross_entropy_flag):
        super(BlendedLoss, self).__init__()
        self.main_loss_type = main_loss_type
        assert main_loss_type in MAIN_LOSS_CHOICES, "invalid main loss: %s" % main_loss_type

        self.metrics = []
        if self.main_loss_type == N_PAIR:
            self.main_loss_fn = NPairLoss()
        elif self.main_loss_type == ANGULAR:
            self.main_loss_fn = AngularLoss()
        elif self.main_loss_type == N_PAIR_ANGULAR:
            self.main_loss_fn = NPairAngularLoss()
        else:
            raise ValueError

        self.cross_entropy_flag = cross_entropy_flag
        self.lambda_blending = 0
        if cross_entropy_flag:
            self.cross_entropy_loss_fn = nn.CrossEntropyLoss()
            self.lambda_blending = 0.3

    def calculate_loss(self, target, output_embedding, output_cross_entropy=None):
        if target is not None:
            target = (target,)

        loss_dict = {}
        blended_loss = 0
        if self.cross_entropy_flag:
            assert output_cross_entropy is not None, "Outputs for cross entropy loss is needed"

            loss_inputs = self._gen_loss_inputs(target, output_cross_entropy)
            cross_entropy_loss = self.cross_entropy_loss_fn(*loss_inputs)
            blended_loss += self.lambda_blending * cross_entropy_loss
            loss_dict[CROSS_ENTROPY + '-loss'] = [cross_entropy_loss.item()]

        loss_inputs = self._gen_loss_inputs(target, output_embedding)
        main_loss_outputs = self.main_loss_fn(*loss_inputs)
        main_loss = main_loss_outputs[0] if type(main_loss_outputs) in (tuple, list) else main_loss_outputs
        blended_loss += (1-self.lambda_blending) * main_loss
        loss_dict[self.main_loss_type+'-loss'] = [main_loss.item()]

        for metric in self.metrics:
            metric(output_embedding, target, main_loss_outputs)

        return blended_loss, loss_dict

    @staticmethod
    def _gen_loss_inputs(target, embedding):
        if type(embedding) not in (tuple, list):
            embedding = (embedding,)
        loss_inputs = embedding
        if target is not None:
            if type(target) not in (tuple, list):
                target = (target,)
            loss_inputs += target
        return loss_inputs


class NPairLoss(nn.Module):
    """
    N-Pair loss
    Sohn, Kihyuk. "Improved Deep Metric Learning with Multi-class N-pair Loss Objective," Advances in Neural Information
    Processing Systems. 2016.
    http://papers.nips.cc/paper/6199-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective
    """

    def __init__(self, l2_reg=0.02):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, embeddings, target):
        n_pairs, n_negatives = self.get_n_pairs(target)

        if embeddings.is_cuda:
            n_pairs = n_pairs.cuda()
            n_negatives = n_negatives.cuda()

        anchors = embeddings[n_pairs[:, 0]]    # (n, embedding_size)
        positives = embeddings[n_pairs[:, 1]]  # (n, embedding_size)
        negatives = embeddings[n_negatives]    # (n, n-1, embedding_size)

        losses = self.n_pair_loss(anchors, positives, negatives) \
            + self.l2_reg * self.l2_loss(anchors, positives)

        return losses

    @staticmethod
    def get_n_pairs(labels):
        """
        Get index of n-pairs and n-negatives
        :param labels: label vector of mini-batch
        :return: A tuple of n_pairs (n, 2)
                        and n_negatives (n, n-1)
        """
        labels = labels.cpu().data.numpy()
        n_pairs = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            anchor, positive = np.random.choice(label_indices, 2, replace=False)
            n_pairs.append([anchor, positive])

        n_pairs = np.array(n_pairs)

        n_negatives = []
        for i in range(len(n_pairs)):
            negative = np.concatenate([n_pairs[:i, 1], n_pairs[i+1:, 1]])
            n_negatives.append(negative)

        n_negatives = np.array(n_negatives)

        return torch.LongTensor(n_pairs), torch.LongTensor(n_negatives)

    @staticmethod
    def n_pair_loss(anchors, positives, negatives):
        """
        Calculates N-Pair loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :return: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

        x = torch.matmul(anchors, (negatives - positives).transpose(1, 2))  # (n, 1, n-1)
        x = torch.sum(torch.exp(x), 2)  # (n, 1)
        loss = torch.mean(torch.log(1+x))
        return loss

    @staticmethod
    def l2_loss(anchors, positives):
        """
        Calculates L2 norm regularization loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :return: A scalar
        """
        return torch.sum(anchors ** 2 + positives ** 2) / anchors.shape[0]


class AngularLoss(NPairLoss):
    """
    Angular loss
    Wang, Jian. "Deep Metric Learning with Angular Loss," CVPR, 2017
    https://arxiv.org/pdf/1708.01682.pdf
    """

    def __init__(self, l2_reg=0.02, angle_bound=1., lambda_ang=2):
        super(AngularLoss, self).__init__()
        self.l2_reg = l2_reg
        self.angle_bound = angle_bound
        self.lambda_ang = lambda_ang
        self.softplus = nn.Softplus()

    def forward(self, embeddings, target):

        embeddings = l2_norm(embeddings)


        n_pairs, n_negatives = self.get_n_pairs(target)

        if embeddings.is_cuda:
            n_pairs = n_pairs.cuda()
            n_negatives = n_negatives.cuda()

        anchors = embeddings[n_pairs[:, 0]]  # (n, embedding_size)
        positives = embeddings[n_pairs[:, 1]]  # (n, embedding_size)
        negatives = embeddings[n_negatives]  # (n, n-1, embedding_size)

        losses = self.angular_loss(anchors, positives, negatives, self.angle_bound) \
                 + self.l2_reg * self.l2_loss(anchors, positives)

        return losses

    @staticmethod
    def angular_loss(anchors, positives, negatives, angle_bound=1.):
        """
        Calculates angular loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :param angle_bound: tan^2 angle
        :return: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

        x = 4. * angle_bound * torch.matmul((anchors + positives), negatives.transpose(1, 2)) \
            - 2. * (1. + angle_bound) * torch.matmul(anchors, positives.transpose(1, 2))  # (n, 1, n-1)

        # Preventing overflow
        with torch.no_grad():
            t = torch.max(x, dim=2)[0]

        x = torch.exp(x - t.unsqueeze(dim=1))
        x = torch.log(torch.exp(-t) + torch.sum(x, 2))
        loss = torch.mean(t + x)

        return loss


class NPairAngularLoss(AngularLoss):
    """
    Angular loss
    Wang, Jian. "Deep Metric Learning with Angular Loss," CVPR, 2017
    https://arxiv.org/pdf/1708.01682.pdf
    """

    def __init__(self, l2_reg=0.02, angle_bound=1., lambda_ang=2):
        super(NPairAngularLoss, self).__init__()
        self.l2_reg = l2_reg
        self.angle_bound = angle_bound
        self.lambda_ang = lambda_ang

    def forward(self, embeddings, target):
        n_pairs, n_negatives = self.get_n_pairs(target)

        if embeddings.is_cuda:
            n_pairs = n_pairs.cuda()
            n_negatives = n_negatives.cuda()

        anchors = embeddings[n_pairs[:, 0]]    # (n, embedding_size)
        positives = embeddings[n_pairs[:, 1]]  # (n, embedding_size)
        negatives = embeddings[n_negatives]    # (n, n-1, embedding_size)

        losses = self.n_pair_angular_loss(anchors, positives, negatives, self.angle_bound) \
            + self.l2_reg * self.l2_loss(anchors, positives)

        return losses

    def n_pair_angular_loss(self, anchors, positives, negatives, angle_bound=1.):
        """
        Calculates N-Pair angular loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :param angle_bound: tan^2 angle
        :return: A scalar, n-pair_loss + lambda * angular_loss
        """
        n_pair = self.n_pair_loss(anchors, positives, negatives)
        angular = self.angular_loss(anchors, positives, negatives, angle_bound)

        return (n_pair + self.lambda_ang * angular) / (1+self.lambda_ang)


class TripletLoss(nn.Module):
    def __init__(self, margin=0, num_instances=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)
        self.K = num_instances

    def forward(self, inputs, targets):
        n = inputs.size(0)
        P = n/self.K

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []

        for i in range(n):
            dist_ap.append(dist[i][mask[i] == 1].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        
        dist_an = torch.stack(dist_an)
        dist_ap = torch.stack(dist_ap)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = float((dist_an.data > dist_ap.data).sum()) * 1. / float(y.size(0))
        return loss, prec