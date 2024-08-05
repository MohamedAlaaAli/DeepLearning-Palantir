import torch
import torch.nn.functional as F

def cw_l2_attack(model, x, target, c=1, max_iter=1000, learning_rate=0.01):
    """
    Carlini-Wagner L2 attack on a given model.
    """
    device = x.device
    
    # Create delta as a leaf tensor
    delta = torch.zeros(x.shape, requires_grad=True, device=device)
    print(delta.is_leaf)
    x_adv = (delta + x).clamp(0, 1.0)
    
    target_lbl = torch.tensor([target], device=device)
    optimizer = torch.optim.Adam([delta], lr=learning_rate)
    
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    for _ in range(max_iter):

        logits = model(x_adv)
        f_values = F.cross_entropy(logits, target_lbl)

        # Compute L2 loss
        l2_loss = torch.sum(delta ** 2)
        loss = l2_loss + c * f_values

        optimizer.zero_grad()
        loss.backward()

        # Print gradients and delta values
        print("Loss:", loss.item())
        optimizer.step()

        x_adv = (delta + x).clamp(0.0, 1.0)

    return (x + delta).detach(), delta.detach()
