import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from tqdm import trange


def save_dc_fp_data(layer, cali_data: torch.Tensor,
                    batch_size: int = 32, keep_gpu: bool = True,
                    input_prob: bool = False, lamb=50, bn_lr=1e-3):
    """Activation after correction"""
    device = 'cuda'
    get_inp_out = GetDcFpLayerInpOut(layer, device=device, input_prob=input_prob, lamb=lamb, bn_lr=bn_lr)
    cached_batches = []

    print("Start correcting {} batches of data!".format(int(cali_data.size(0) / batch_size)))
    for i in trange(int(cali_data.size(0) / batch_size)):
            cur_out, out_fp = get_inp_out(cali_data[i * batch_size:(i + 1) * batch_size])
            cached_batches.append((cur_out.cpu(), out_fp.cpu()))
    cached_outs = torch.cat([x[0] for x in cached_batches])

    return cached_outs




class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """

    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


class input_hook(object):
    """
	Forward_hook used to get the output of the intermediate layer. 
	"""
    def __init__(self, stop_forward=False):
        super(input_hook, self).__init__()
        self.inputs = None

    def hook(self, module, input, output):
        self.inputs = input

    def clear(self):
        self.inputs = None

def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()

class GetDcFpLayerInpOut:
    def __init__(self, layer,
                 device: torch.device, input_prob: bool = False, lamb=50, bn_lr=1e-3):
        self.layer = layer
        self.device = device
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=False)
        self.input_prob = input_prob
        self.bn_stats = []
        self.eps = 1e-6
        self.lamb=lamb
        self.bn_lr=bn_lr
        for n, m in self.layer.named_modules():
            if isinstance(m, nn.BatchNorm2d):
            # get the statistics in the BatchNorm layers
                self.bn_stats.append(
                    (m.running_mean.detach().clone().flatten().cuda(),
                    torch.sqrt(m.running_var +
                                self.eps).detach().clone().flatten().cuda()))
    
    def own_loss(self, A, B):
        return (A - B).norm()**2 / B.size(0)
    
    def relative_loss(self, A, B):
        return (A-B).abs().mean()/A.abs().mean()

    def __call__(self, model_input):
        handle = self.layer.register_forward_hook(self.data_saver)
        hooks = []
        hook_handles = []
        for name, module in self.layer.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                hook = input_hook()
                hooks.append(hook)
                hook_handles.append(module.register_forward_hook(hook.hook))
        assert len(hooks) == len(self.bn_stats)

        input_sym = self.data_saver.input_store[0].detach()
            
        handle.remove()
        para_input = input_sym.data.clone()
        para_input = para_input.to(self.device)
        para_input.requires_grad = True
        optimizer = optim.Adam([para_input], lr=self.bn_lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        min_lr=1e-5,
                                                        verbose=False,
                                                        patience=100)
        # iters=500
        iters = 500
        for iter in range(iters):
            self.layer.zero_grad()
            optimizer.zero_grad()
            for hook in hooks:
                hook.clear()
            _ = self.layer(para_input)
            mean_loss = 0
            std_loss = 0
            for num, (bn_stat, hook) in enumerate(zip(self.bn_stats, hooks)):
                tmp_input = hook.inputs[0]
                bn_mean, bn_std = bn_stat[0], bn_stat[1]
                tmp_mean = torch.mean(tmp_input.view(tmp_input.size(0),
                                                    tmp_input.size(1), -1),
                                    dim=2)
                tmp_std = torch.sqrt(
                    torch.var(tmp_input.view(tmp_input.size(0),
                                            tmp_input.size(1), -1),
                            dim=2) + self.eps)
                mean_loss += self.own_loss(bn_mean, tmp_mean)
                std_loss += self.own_loss(bn_std, tmp_std)
            constraint_loss = lp_loss(para_input, input_sym) / self.lamb
            total_loss = mean_loss + std_loss + constraint_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss.item())
            # if (iter+1) % 500 == 0:
            #     print('Total loss:\t{:.3f} (mse:{:.3f}, mean:{:.3f}, std:{:.3f})\tcount={}'.format(
            #     float(total_loss), float(constraint_loss), float(mean_loss), float(std_loss), iter))
                
        with torch.no_grad():
            out_fp = self.layer(para_input)

        return out_fp.detach()