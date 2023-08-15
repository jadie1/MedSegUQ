import importlib
import torch
import torch.nn as nn
import numpy as np
from scipy.special import softmax

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.members = []
        device = args.device
        for model_path in args.members:
            print(model_path)
            model_module = importlib.import_module('.%s' % args.member_model_name, 'models')
            model = model_module.Model(args)
            model.to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            self.members.append(model)
        self.num_members = len(self.members)
    def forward(self, x: torch.Tensor, member_id=None) -> torch.Tensor:
        if member_id == None:
            outputs = [model(x) for model in self.members]
            pred = torch.stack(outputs).mean(0)
        else:
            pred = self.members[member_id](x)
        return pred