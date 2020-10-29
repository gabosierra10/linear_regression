#Created by Gabriel Sierra
#freeCodeCamp.org YouTube Video PyTorch Course
#Practice Makes Perfect!
#Linear Regression Code with no built in functions using PyTorch


import torch
import numpy as np

#| Region |Temp.(F)|Rainfall(mm)|Humidity(%)|Plantains(ton)|Avocados(ton)|
#|San Juan|   81   |     70     |     40    |      60       |     64      |
#|Carolina|   82   |     85     |     70    |      49       |     74      |
#| Caguas |   83   |     105    |     50    |      101      |     57      |
#| Ponce  |   89   |     40     |     20    |      27       |     150     |
#| Arecibo|   85   |     80     |     59    |      80       |     30      |

#plantain_yield = w11 * temp + w12 * rainfall + w13 * humidity + b1
#avocado_yield = w21 * temp + w22 * rainfall + w23 * humidity + b2


    
def mse(t1, t2):
    #Mean Squared Error
    diff = t1-t2
    return torch.sum(diff*diff)/diff.numel()
    
def main():
    #Input data (temp, rainfall and humidity)
    inputs = np.array([[81, 70, 40],
                        [82, 85, 70],
                        [83,105,50],
                        [89, 40, 20],
                        [85, 80, 59]], dtype='float32')
    
    #Goal (plantains, avocado)
    goal = np.array([[60, 64],
                        [49, 74],
                        [101, 57],
                        [27, 150],
                        [80, 30]], dtype='float32')
                        
    #Conversion to tensors
    inputs = torch.from_numpy(inputs)
    goal = torch.from_numpy(goal)
    
    #Weights and biases
    w = torch.randn(2, 3, requires_grad=True)
    b = torch.randn(2, requires_grad=True)
    
    def model(x):
        # Inputs * Weights transposed + biases
        return x @ w.t() + b
    
    #Predictions
    pred = model(inputs)
    
    #Calculate loss
    loss = mse(pred, goal)
    
    #Calculate derivatives
    loss.backward()
    
    #Adjust weights
    with torch.no_grad():
        w -= w.grad * 0.00001
        b -= b.grad * 0.00001
        w.grad.zero_()
        b.grad.zero_()
    
    #Recalculate loss with new weights
    pred = model(inputs)
    loss = mse(pred, goal)
    
    #Train
    for i in range (1500):
        pred = model(inputs)
        loss = mse(pred, goal)
        print(loss)
        loss.backward()
        with torch.no_grad():
            w -= w.grad * 0.00001
            b -= b.grad * 0.00001
            w.grad.zero_()
            b.grad.zero_()

    #Compare values
    print(goal)
    print(pred)
    
main()
