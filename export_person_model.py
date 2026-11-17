"""Run OFFLINE of Streamlit deployment on a workstation with internet and torch/torchvision/onnx installed."""
from pathlib import Path
import torch
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.model=lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT).eval()
    def forward(self,x): return self.model(x)['out']
if __name__=='__main__':
    p=Path('models/person_lraspp.onnx'); p.parent.mkdir(exist_ok=True)
    torch.onnx.export(Wrapper().eval(),torch.zeros(1,3,320,320),str(p),input_names=['image'],output_names=['logits'],opset_version=17,dynamo=False)
    print(p)
