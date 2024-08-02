import torch

def fgsm_attack(image, epsilon, gradients):
    
    sign_data_grad = gradients.sign()
    p = epsilon * sign_data_grad

    perturbed_image = image - p
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image, p