import torch
import torch.utils.data as Data


torch.manual_seed(0)
Batch_size = 5


x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(
        dataset = torch_dataset,
        batch_size = Batch_size,
        shuffle = True,
        num_workers = 2,
        )

def show_batch():
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):
            print("Epoch: ", epoch, " |Step: ", step, " |batch x: ",
                    batch_x.numpy(), " | batch y: ", batch_y.numpy())



if __name__ == "__main__":
    show_batch()
