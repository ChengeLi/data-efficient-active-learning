import torch
import torch.nn.functional as F
import hyptorch.pmath as pmath
import pdb

def contrastive_loss(x0, x1, target=None, tau=0.2, hyp_c=1/15, cuda_ind=0):
    # x0 and x1 - positive pair
    # tau - cross-entropy temperature
    # hyp_c - hyperbolic curvature, "0" enables sphere mode

    if hyp_c == 0:
        dist_f = lambda x, y: x @ y.t()
    else:
        dist_f = lambda x, y: -pmath.dist_matrix(x, y, c=hyp_c)
    bsize = x0.shape[0]
    # target = torch.arange(bsize).cuda()
    # eye_mask = torch.eye(bsize).cuda() * 1e9
    # device = [torch.device(f"cuda:{i}") for i in range(4)]
    if target==None:
        target = torch.arange(bsize).to(torch.device(f"cuda:{cuda_ind}"))
    else:
        target = torch.tensor([tt[0].data.cpu().item() for tt in target]).to(torch.device(f"cuda:{cuda_ind}"))
    eye_mask = torch.eye(bsize).to(torch.device(f"cuda:{cuda_ind}")) * 1e9
    logits00 = dist_f(x0, x0) / torch.tensor(tau).to(torch.device(f"cuda:{cuda_ind}")) - eye_mask
    logits01 = dist_f(x0, x1) / torch.tensor(tau).to(torch.device(f"cuda:{cuda_ind}"))
    logits = torch.cat([logits01, logits00], dim=1)
    logits -= logits.max(1, keepdim=True)[0].detach()
    loss = F.cross_entropy(logits, target)
    # stats = {
    #     "logits/min": logits01.min().item(),
    #     "logits/mean": logits01.mean().item(),
    #     "logits/max": logits01.max().item(),
    #     "logits/acc": (logits01.argmax(-1) == target).float().mean().item(),
    # }
    # return loss, stats
    return loss, None

