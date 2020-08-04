import torch

x = torch.tensor([1,2,3,4], dtype=torch.float32)
y = torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0.0, dtype= torch.float32, requires_grad=True)

#model predection
def forward(x):
    return  w*x

#loss
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()
#gradient
#MSE = 1/N * (w*x - y)**2
#dJ/dw = 1/N 2x (w*x - y)

def gradient(x, y, y_predicted):
    return torch.dot(2*x, y_predicted - y).mean()

print(f'Prediection brfore training: f(5) = {forward(5):.3f}')

#training
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = forward(x)

    #loss
    l = loss(y, y_pred)

    #gradient
    l.backward() # dl/dw

    #update weights
    with torch.no_grad():
        w -= learning_rate * w.grad
    #zero dradients
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'epoch {epoch + 1}: w = {w:.3f}, loss = {l:.3f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')