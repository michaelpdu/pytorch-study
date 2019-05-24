import torch

print('Test unsqueeze...')
print('>==========================')
a = torch.randn(2,3)
print('a:', a)
print('a.shape:', a.shape)

# a: tensor([[-0.4498, -1.3094,  0.2040],
#         [ 1.6251, -1.1315, -0.2166]])
# a.shape: torch.Size([2, 3])

print('>==========================')
b = torch.unsqueeze(a, 1)
print('b:', b)
print('b.shape:', b.shape)

# b: tensor([[[-0.4498, -1.3094,  0.2040]],
#         [[ 1.6251, -1.1315, -0.2166]]])
# b.shape: torch.Size([2, 1, 3])

print('>==========================')
c = a.unsqueeze(0)
print('c:', c)
print('c.shape:', c.shape)

# c: tensor([[[-0.4498, -1.3094,  0.2040],
#          [ 1.6251, -1.1315, -0.2166]]])
# c.shape: torch.Size([1, 2, 3])

print('\n\nTest squeeze...')
print('>==========================')
d = torch.randn(1,1,3)
print('d:', d)
print('d.shape:', d.shape)

# d: tensor([[[ 0.3004, -0.7830,  1.2858]]])
# d.shape: torch.Size([1, 1, 3])

print('>==========================')
e = torch.squeeze(d)
print('e:', e)
print('e.shape:', e.shape)

# e: tensor([ 0.3004, -0.7830,  1.2858])
# e.shape: torch.Size([3])

print('>==========================')
f = torch.squeeze(d, 0)
print('f:', f)
print('f.shape:', f.shape)

# f: tensor([[ 0.3004, -0.7830,  1.2858]])
# f.shape: torch.Size([1, 3])
