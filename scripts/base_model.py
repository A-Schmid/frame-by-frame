class Model:
    """Base class for all neural network modules.

    Your models should also subclass this class.

    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::

        import torch.nn as nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))

    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    """
    
    def __init__(self, name):
        self.name = name
        
    # Instance methods
    def load_labels(self, label_path):
        """Load categories."""
        with open(label_path) as f:
            return [line.rstrip() for line in f.readlines()]
    
    def load_pretrained(self, model_path):
        """Load pre-trained weights"""
        print('Loading weights...')
        pass
    
    def predict(self, X):
        """Forward pass and output predicted classes with (softmax) probabilities."""
        pass
    