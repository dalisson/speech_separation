import torch

def upit_loss_2speakers(y_hat, y):
    '''
    the si_snr loss computed for two speakers
    y_hat: tensor shaped like (2, T)
    y: tensor shape like (2, T)

    where T = utterance duration
    '''
    y_power = torch.pow(y, 2).sum(-1, keepdim=False)

    scale_factor = y_hat@y.t()/y_power

    s_target = (torch.pow(y.unsqueeze(1) * scale_factor.t().unsqueeze(-1), 2).sum(-1)).t()

    #e_noise
    residual = y_hat.unsqueeze(1) - y
    residual_norms = torch.pow(residual, 2).sum(-1, keepdim=False)

    temp = (10*(torch.log10(s_target) - torch.log10(residual_norms)))
    loss_one = temp[0, 0] + temp[1, 1]
    loss_two = temp[0, 1] + temp[1, 0]
    maximum_loss = loss_one if loss_one > loss_two else loss_two

    return maximum_loss
    