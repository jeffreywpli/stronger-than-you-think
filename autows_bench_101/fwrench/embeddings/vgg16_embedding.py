from matplotlib.pyplot import bar
import numpy as np
import torch 
import torchvision.models as models

from .base_embedding import BaseEmbedding

class VGG16Embedding(BaseEmbedding):
    def __init__(self):
        vgg16 = models.vgg16(pretrained=True)
        self.model = torch.nn.Sequential(*list(vgg16.children())[:-1])
        self.model.eval()

    def fit(self, *data):
        pass

    def transform(self, data):
        X_np = self._unpack_data(data, flatten=False)
        if X_np.shape[1] == 1: # Repeat since MNIST is greyscale
            X_np = X_np.repeat(3, axis=1)
        elif X_np.shape[1] > 3: # Probably need to permute
            X_np = np.transpose(X_np, (0, 3, 1, 2))
        with torch.no_grad():
            # TODO batching ... 
            X = torch.from_numpy(X_np)
            X = X.type(torch.FloatTensor)
            X.cuda()
            X_emb = self.model.forward(X)
            X_emb = X_emb.cpu().numpy()
            X_emb = X_emb.reshape(X_emb.shape[0], -1)
        return self._repack_data(data, X_emb)

    def fit_transform(self, data, ngpus=1, max_epochs=5, hidden_size=128):
        self.fit(data)
        return self.transform(data)
