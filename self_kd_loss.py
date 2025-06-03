import torch.nn.functional as F # Need to import functional
import torch

# Function to calculate distillation loss (KL Divergence between soft distributions)
def distillation_loss(student_logits, teacher_logits, temp, reduction='batchmean'):
    """
    Calculate KL Divergence loss for knowledge distillation.
    Args:
        student_logits: Logits from the student model.
        teacher_logits: Logits from the teacher model.
        temp: Temperature to soften the softmax.
        reduction: Method to reduce the loss ('batchmean', 'sum', 'mean', 'none').
    Returns:
        Scaled distillation loss.
    """
    # Soften teacher and student logits using temperature T
    # Use .detach() for teacher_logits to prevent gradient calculation for the teacher
    soft_targets = F.softmax(teacher_logits / temp, dim=-1)
    soft_prob = F.log_softmax(student_logits / temp, dim=-1) # kl_div requires log-probabilities as input

    # Calculate KL divergence.
    # reduction='batchmean' will average over the batch and other dimensions (usually number of classes)
    # log_target=False because soft_targets are probabilities, not log-probabilities
    kd_loss = F.kl_div(soft_prob, soft_targets.detach(), reduction=reduction, log_target=False)

    # Scale loss by T*T. This compensates for the gradient reduction due to division by T in softmax.
    # Refer to the original paper by Hinton et al.
    scaled_kd_loss = kd_loss * (temp * temp)
    return scaled_kd_loss

# Function to update EMA weights for the teacher model
@torch.no_grad() # Ensure no gradient calculation when updating the teacher
def update_ema_teacher(student_model, teacher_model, decay):
    """
    Update teacher model weights using Exponential Moving Average.
    """
    student_params = dict(student_model.named_parameters())
    teacher_params = dict(teacher_model.named_parameters())

    # Ensure keys match (safety check)
    assert student_params.keys() == teacher_params.keys()

    for name, param_student in student_params.items():
        param_teacher = teacher_params[name]
        # In-place update
        param_teacher.data.mul_(decay).add_(param_student.data, alpha=1 - decay)