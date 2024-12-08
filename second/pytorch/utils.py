import torch


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num ([type]): pillar有效点数 [P]
        max_num ([type]): pillar采样最大点数 scalar

    Returns:
        [type]: 掩码 [P max_num]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)  # 在指定位置插入一个大小为 1 的新维度：[P 1]
    max_num_shape = [1] * len(actual_num.shape)     # [1, 1]
    max_num_shape[axis + 1] = -1                    # [1, -1]
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)  # [1 max_num]   [0, 1, ..., max_num - 1]
    paddings_indicator = actual_num.int() > max_num
    return paddings_indicator  # [P max_num]
