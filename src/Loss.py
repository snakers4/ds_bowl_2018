import torch
import torch.nn as nn
import torch.nn.functional as F

class TDiceLoss(nn.Module):
    def __init__(self, dice_weight=1):
        super().__init__()
        self.nll_loss = nn.BCELoss()
        self.dice_weight = dice_weight

    def forward(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        if self.dice_weight:
            eps = 1e-15
            dice_target = (targets == 1).float()
            dice_output = outputs
            intersection = (dice_output * dice_target).sum()
            union = dice_output.sum() + dice_target.sum() + eps
            loss += 1 - torch.log(2 * intersection / union)

        return loss

class WDiceLoss(nn.Module):
    def __init__(self,
                 bce_weight=1,
                 dice_weight=1):
        super().__init__()

        self.nll_loss = nn.BCELoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight        

    def forward(self, outputs, targets):
        loss = self.nll_loss(outputs, targets) * self.bce_weight
        
        eps = 1e-15
        dice_target = (targets == 1).float()
        dice_output = outputs
        intersection = (dice_output * dice_target).sum()
        union = dice_output.sum() + dice_target.sum() + eps
        loss += ( 1 - torch.log(2 * intersection / union) ) * self.dice_weight

        return loss

class TSDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.nll_loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        
        eps = 1e-10
        dice_target = (targets == 1).float()
        dice_output = F.sigmoid(outputs)
        intersection = (dice_output * dice_target).sum()
        union = dice_output.sum() + dice_target.sum() + eps
        loss += (1 - torch.log(2 * intersection / union))

        return loss    
    
class AVDiceLoss(nn.Module):
    def __init__(self,
                 bce_weight=1,
                 dice_weight=1,
                 is_vectors=False):
        super().__init__()

        self.nll_loss = nn.BCEWithLogitsLoss()
        # also try acos distance?
        self.v_loss = torch.nn.MSELoss()
        
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.is_vectors = is_vectors

    def forward(self, outputs, targets):
        if self.is_vectors == False:
            loss = self.nll_loss(outputs, targets) * self.bce_weight

            eps = 1e-10
            dice_target = (targets == 1).float()
            dice_output = F.sigmoid(outputs)
            intersection = (dice_output * dice_target).sum()
            union = dice_output.sum() + dice_target.sum() + eps
            loss += (1 - torch.log(2 * intersection / union)) * self.dice_weight
            # loss += torch.clamp(1 - torch.log(2 * intersection / union),0,100)  * self.dice_weight
            return loss
        else:
            # only last 2 components
            outputs_vectors = outputs[:,-2:,:,:]
            targets_vectors = targets[:,-2:,:,:]
            # all the other components
            outputs = outputs[:,:-2,:,:]
            targets = targets[:,:-2,:,:]            
            
            vector_loss = self.v_loss(outputs_vectors,targets_vectors)
            loss = self.nll_loss(outputs, targets) * self.bce_weight
            
            eps = 1e-10
            dice_target = (targets == 1).float()
            dice_output = F.sigmoid(outputs)
            intersection = (dice_output * dice_target).sum()
            union = dice_output.sum() + dice_target.sum() + eps
            loss += (1 - torch.log(2 * intersection / union)) * self.dice_weight
            loss += vector_loss

            return loss            

def dice_loss(preds, trues, weight=None, is_average=True):
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    if weight is not None:
        w = torch.autograd.Variable(weight).view(num, -1)
        preds = preds * w
        trues = trues * w
    intersection = (preds * trues).sum(1)
    scores = 2. * (intersection + 1) / (preds.sum(1) + trues.sum(1) + 1)

    if is_average:
        score = scores.sum()/num
        return torch.clamp(score, 0., 1.)
    else:
        return scores

def dice_clamp(preds,
               trues,
               is_average=True):
    preds = torch.round(preds)
    return dice_loss(preds, trues, is_average=is_average)    
    
class DiceLoss(nn.Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average

    def forward(self, input, target, weight=None):
        return 1-dice_loss(F.sigmoid(input), target, weight=weight, is_average=self.size_average)    
    
class BCEDiceLoss(nn.Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.dice = DiceLoss(size_average=size_average)

    def forward(self, input, target, weight=None):
        return nn.modules.loss.BCEWithLogitsLoss(size_average=self.size_average, weight=weight)(input, target) + self.dice(input, target, weight=weight)