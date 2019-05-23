import torch
from torch.autograd import Function

'''
自定义Dice损失,(2*预测正确的结果)/(真实结果+预测结果)
'''
class DiceCoeff(Function):
    """Dice coeff for individual examples"""
    def forward(self,input,target):
        input, target = input.float(), target.float()
        smooth = 0.0001         # 平滑项
        self.save_for_backward(input, target)
        self.inter = torch.dot(input.view(-1),target.view(-1))
        self.union = torch.sum(input) + torch.sum(target)

        return (2*self.inter + smooth).float()/(self.union + smooth).float()

    def backward(self,grad_output):
        input, target = self.saved_variables
        grad_input = grad_target = None
        if self.needs_input_grad[0]:
            grad_input = 2 * grad_output * (target * self.union - self.inter).float() \
                         / self.union.pow(2)
        if self.needs_input_grad[1]:
            grad_target = None
        return grad_input, grad_target


def dice_coeff(x, y):
    """Dice coeff for batches"""
    if x.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
    for i,wrap in enumerate(zip(x,y)):
        s += DiceCoeff()(wrap[0],wrap[1])
    return s
