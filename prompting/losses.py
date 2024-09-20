import torch.nn.functional as F
import torch
def transpose(x):
    return x.t() if x.dim() == 2 else x.permute(0, 2, 1)

def contrastive_loss(visual_features, class_prototypes, labels=None, t=0.07):
    # print(visual_features.shape, class_prototypes.shape, transpose(class_prototypes).shape)
    logits = t.exp() * visual_features @ transpose(class_prototypes)

    if labels is not None:
        return F.cross_entropy(logits, labels), logits
    else:
        return None, logits


def moe_contrastive_loss(visual_features, class_prototypes, labels=None, t=0.07, instance_loss=False, groups=None, indices=None):
    # print(visual_features.shape, class_prototypes.shape, transpose(class_prototypes).shape)
    if not instance_loss:
        # logits = t.exp() * visual_features @ transpose(class_prototypes)
        # logits = t.exp() * torch.einsum('bcd, ned->bnce', visual_features, class_prototypes)
        # logits = t.exp() * torch.einsum('cd, ned->nce', visual_features, class_prototypes)
        # print(visual_features.shape, class_prototypes.shape)
        # logits = t.exp() * torch.einsum('ncd, ned->nce', visual_features, class_prototypes)
        # print(logits.shape, labels.shape)
        # labels = labels.unsqueeze(0).repeat(len(visual_features), 1,1)
        # 假设 visual_features 和 class_prototypes 已经定义好
        # visual_features: [b, c, d]
        # class_prototypes: [n, e, d]
        # 其中 n = b * g, 这里分成 b 组，每组 g 个
        # print(visual_features.shape, class_prototypes.shape)

        # Example dimensions
        """
        b = visual_features.size(0)
        c = visual_features.size(1)
        d = visual_features.size(2)
        n = class_prototypes.size(0)
        e = class_prototypes.size(1)
        g = n // b  # Assuming n is a multiple of b

        # Reshape class_prototypes to [b, g, e, d]
        class_prototypes_reshaped = class_prototypes.view(b, g, e, d)

        # Compute the logits using einsum
        logits = t.exp() * torch.einsum('bcd, bged->bgce', visual_features, class_prototypes_reshaped)

        # Sum over the g dimension to get the result in shape [b, c, e]
        logits = logits.sum(dim=1)

        # If needed, reshape back to [n, c, e] by expanding the batch dimension
        logits = logits.view(b, c, e)
        """

        #######################################
        # 预定义的组切割点

        # 按照分组点进行切割
        grouped_prototypes = class_prototypes
        # grouped_prototypes = torch.split(class_prototypes, groups)

        # 获取每个组的大小
        # group_sizes = [len(group) for group in grouped_prototypes]

        # 获取必要的维度
        b = visual_features.size(0)
        c = visual_features.size(1)
        d = visual_features.size(2)
        e = class_prototypes[0].size(0)
        # e = class_prototypes.size(1)

        # 验证分组后的总和是否与原始的类原型数一致
        # assert sum(group_sizes) == class_prototypes.size(0), "分组后的总大小与原数据不一致"

        # 重塑每个组以匹配视觉特征的批次大小
        # reshaped_prototypes = [group.view(b, -1, e, d) for group in grouped_prototypes]

        # 计算每个组的logits并求和
        # logits_list = []
        # for i, group in enumerate(grouped_prototypes):
        #     # print(group.shape)
        #     if i in indices:
        #         logits = torch.einsum('cd, med->cme', visual_features[i], group)
        #         logits = logits.sum(dim=1)  # 在组内维度求和
        #         logits_list.append(logits)
        #     else:
        #         logits_list.append(visual_features.new_zeros((c, e)))
        final_logits = torch.einsum('gcd, ged->gce', visual_features, grouped_prototypes)

        # 合并所有组的logits
        # final_logits = torch.stack(logits_list)  # 在通道维度上拼接

        # 如果需要，调整形状为 [n, c, e]
        logits = final_logits.view(-1, c, e)

    else:
        # logits = t.exp() * torch.einsum('bd, cd->bc', visual_features, class_prototypes)
        logits = t.exp() * torch.einsum('bd, bcd->bc', visual_features, class_prototypes)
    # print(logits.shape, labels.shape)
    if labels is not None:
        labels = labels.reshape(-1)
        if not instance_loss:
            # group_anchor_Feature
            # for group in grouped_prototypes:
            #     group = group.mean(dim=0)

                # [g, c, d] [b] [b, g, d] [b d] [b, g]
            # return F.cross_entropy(logits.reshape(len(labels), -1), labels), logits, torch.stack([group.mean(dim=0) for group in grouped_prototypes])
            return F.cross_entropy(logits.reshape(len(labels), -1), labels), logits, None
            # return F.cross_entropy(logits.reshape(len(labels), -1), labels), logits, torch.stack([group.mean(dim=(0,1)) for group in grouped_prototypes])
        else:
            return F.cross_entropy(logits, labels), logits
    else:
        return None, logits