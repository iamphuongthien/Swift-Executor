import torch
import torch.nn as nn
# 1) design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#   - forward pass: coompute prediction
#   - backward pass: gradients
#   - update weights

x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

x_test = torch.tensor([50], dtype=torch.float32)

n_samples, n_features = x.shape
print(n_samples, n_features)

input_size = n_features
out_size = n_features

# model = nn.Linear(input_size, out_size)

class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        #define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, out_size)

print(f'Prediection brfore training: f(5) = {model(x_test).item():.3f}')

#training
learning_rate = 0.01
n_iters = 2500

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), learning_rate)
for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = model(x)

    #loss
    l = loss(y, y_pred)

    #gradient
    l.backward() # dl/dw

    #update weights
    optimizer.step()
    #zero dradients
    optimizer.zero_grad()

    if epoch % 100 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f}')

print(f'Prediction after training: f(5) = {model(x_test).item():.3f}')