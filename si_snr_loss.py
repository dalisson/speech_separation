import torch

def upit_loss(y_hat, y):
    '''
    the si_snr loss computed for two speakers
    y_hat: tensor shaped like (2, T)
    y: tensor shape like (2, T)
    where T = utterance duration
    '''
    si_til_numerator = (y@y_hat.t()).t()[..., None] * y
    si_til_denominator = (y * y).sum(-1).unsqueeze(-1)
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
    
    e_til_mod = (e_til * e_til).sum(-1)
    
    #[[y[0]@y_hat[0]/mod(y_0), y[0]@y_hat[1]/mod(y_0)]
    # y[1]@y_hat[0]/mod(y_1),  y[1]@y_hat[1]/mod(y_1)]
    si_til_mod = (si_til * si_til).sum(-1)

    loss = 10*(torch.log10(si_til_mod) - torch.log10(e_til_mod))
    
    n_speakers = loss.shape[0]
    
    indexes = torch.tensor(list(range(n_speakers)) +list(range(n_speakers))).unfold(0, n_speakers, 1)[:-1]
    
    return torch.gather(loss, 0, indexes).sum(-1).max()

    
def si_snr(y_hat, y):
    loss = 0
    for prediction in y_hat:
        loss += -upit_loss(prediction.squeeze(0), y)
    
    return loss
