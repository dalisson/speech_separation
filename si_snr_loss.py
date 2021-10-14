import torch

def upit_loss(y_hat, y):
    '''
    the si_snr loss computed for number of speakers
    y_hat: tensor shaped like (N, T)
    y: tensor shape like (N, T)
    where T = utterance duration
          N = number of speakers
    '''

    #compute the mean 
    y_mean = torch.mean(y, dim=-1).t()[...,None]
    y_hat_mean = torch.mean(y_hat, dim=-1).t()[...,None]

    #set y and y_hat to zero mean
    y_zero_mean = y - y_mean
    y_hat_zero_mean = y_hat - y_hat_mean


    si_til_numerator = (y_zero_mean@y_hat_zero_mean.t()).t()[..., None] * y
    si_til_denominator = (y_zero_mean * y_zero_mean).sum(-1).unsqueeze(-1)
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
    
    e_til = y_hat_zero_mean - si_til
    
    #[[y_hat[0] - y[0]@y_hat[0]/mod(y_0), y_hat[1] - y[0]@y_hat[1]/mod(y_0)]
    # y_hat[0] - y[1]@y_hat[0]/mod(y_1), y_hat[1] - y[1]@y_hat[1]/mod(y_1)]
    
    e_til_mod = (e_til * e_til).sum(-1)
    
    #[[y[0]@y_hat[0]/mod(y_0), y[0]@y_hat[1]/mod(y_0)]
    # y[1]@y_hat[0]/mod(y_1),  y[1]@y_hat[1]/mod(y_1)]
    si_til_mod = (si_til * si_til).sum(-1)
    n_speakers = y.shape[0]
    loss = 10*(torch.log10(si_til_mod) - torch.log10(e_til_mod))
    
   
    
    indexes = torch.tensor(list(range(n_speakers)) +list(range(n_speakers))).unfold(0, n_speakers, 1)[:-1]
    
    return torch.gather(loss, 0, indexes).mean(-1).max()

    
def si_snr(y_hat, y):
    loss = 0
    counter = 0
    for prediction in y_hat:
        loss -= upit_loss(prediction.squeeze(0), y)
        counter += 1
    return loss/counter
