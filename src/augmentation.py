import numpy as np
import torchvision.transforms as transforms

class Scaling:
    """
    scaling the sequence
    """
    def __init__(self, sc_factor: float=0.3):
        """
        Args:
            sc_factor (float): scaling factor
        """
        self.sc_factor = sc_factor

    def __call__(self, seq):
        """
        seq.shape: (1, 1, len_seq)
        seq.type: torch.Tensor
        """
        # factor = np.random.normal(loc=1., scale=self.sc_factor)
        factor = np.random.uniform(low=1-self.sc_factor, high=1+self.sc_factor)
        mean_zero_seq = seq - seq.mean()
        scaled_mean_zero_seq = mean_zero_seq * factor
        return scaled_mean_zero_seq.float()
    
def transform(sigma):
    return transforms.Compose([
        Scaling(sigma)
    ])