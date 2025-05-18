import torch

data = torch.load("data/graph.pt", weights_only=False)

print("data['dev'].node_id:")
print(data["dev"].node_id)