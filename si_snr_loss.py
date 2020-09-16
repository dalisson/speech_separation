import torch

def upit_loss_2speakers(y_hat, y):
    '''
    the si_snr loss computed for two speakers
    y_hat: tensor shaped like (2, T)
    y: tensor shape like (2, T)
    where T = utterance duration
    '''
    si_til_numerator = (y@y_hat.t()).t()[..., None] * y
    si_til_denominator = (y * y).sum(-1).sqrt().unsqueeze(-1)
    si_til = si_til_numerator/ si_til_denominator
    
    #y[0]@y_hat[0]/mod(y_0)
    #y[0]@y_hat[1]/mod(y_0)
    #y[1]@y_hat[0]/mod(y_1)
    #y[1]@y_hat[1]/mod(y_1)
    si_til = torch.transpose(si_til, 0, 1)
    
    # y_hat[0] - y[0]@y_hat[0]/mod(y_0)
    # y_hat[1] - y[0]@y_hat[1]/mod(y_0)
    # y_hat[0] - y[1]@y_hat[0]/mod(y_1)
    # y_hat[1] - y[1]@y_hat[1]/mod(y_1)
    
    e_til = y_hat - si_til
    
    #[[y_hat[0] - y[0]@y_hat[0]/mod(y_0), y_hat[1] - y[0]@y_hat[1]/mod(y_0)]
    # y_hat[0] - y[1]@y_hat[0]/mod(y_1), y_hat[1] - y[1]@y_hat[1]/mod(y_1)]
    
    e_til_mod = (e_til * e_til).sum(-1).sqrt()
    
    #[[y[0]@y_hat[0]/mod(y_0), y[0]@y_hat[1]/mod(y_0)]
    # y[1]@y_hat[0]/mod(y_1),  y[1]@y_hat[1]/mod(y_1)]
    si_til_mod = (si_til * si_til).sum(-1).sqrt()

    loss = 10*(torch.log10(si_til_mod) - torch.log10(e_til_mod))
    
    loss_one = loss[0][0] + loss[1][1]
    loss_two = loss[1][0] + loss[0][1]

    return -max(loss_one, loss_two)
    
def si_snr(y_hat, y):
    loss = 0
    for prediction in y_hat:
        loss += upit_loss_2speaker(prediction.squeeze(0), y)
    
    return loss
