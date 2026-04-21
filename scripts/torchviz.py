import os
from src.config import DEVICE
from src.dataset import get_dataloaders
from src.models.neural_net import ChurnNeuralNet
from torch import randn
from torchviz import make_dot
def main():
    #set up save directory
    save_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots"
    )
    os.makedirs(save_dir, exist_ok=True)
    _, _, _, _, input_dim = get_dataloaders()

    nn_model = ChurnNeuralNet(input_dim).to(DEVICE)

    # Evaluate the model 
    nn_model.eval()
    # I think this needs to happen in order 
    # for us to be able to
    # render the thingamajig

    dummy_input = randn(1, input_dim).to(DEVICE)
    dot = make_dot(nn_model(dummy_input), params=dict(nn_model.named_parameters()))
    dot.render("neural_network_visualization", format="png", directory=save_dir)

if __name__ == "__main__":
    main()