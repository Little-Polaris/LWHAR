from Model import Model7
from Model import Model8

model = Model7.Model(60, 25,2,
                     [[1, 2], [2, 21], [3, 21], [4, 3], [5, 21], [6, 5], [7, 6],
                    [8, 7], [9, 21], [10, 9], [11, 10], [12, 11], [13, 1],
                    [14, 13], [15, 14], [16, 15], [17, 1], [18, 17], [19, 18],
                    [20, 19], [22, 23], [23, 8], [24, 25], [25, 12]],
                     3)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(total_params)