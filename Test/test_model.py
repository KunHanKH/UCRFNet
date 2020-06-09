import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Unet.model import Unet
from CRF.crfrnn import CrfRnn


def main():
    n_class = 10
    n_channel = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = Unet(n_channel, n_class).to(device)
    crfrnn = CrfRnn(n_class, 5)

    if n_class > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer_unet = optim.RMSprop(unet.parameters(), lr=0.01, weight_decay=1e-8, momentum=0.9)
    optimizer_crfrnn = optim.RMSprop(crfrnn.parameters(), lr=0.01, weight_decay=1e-8, momentum=0.9)

    x = torch.rand(1, 3, 500, 500).to(device)
    gt = torch.ones(1, 500, 500).long().to(device)

    optimizer_unet.zero_grad()
    optimizer_crfrnn.zero_grad()
    logits = unet(x).to('cpu')
    x = x.to('cpu')
    res = crfrnn(x, logits).to(device)
    print(res.shape)
    loss = criterion(res, gt)
    loss.backward()
    optimizer_unet.step()
    optimizer_crfrnn.step()


if __name__ == '__main__':
    main()