import io
import torch
import torch.onnx
from models.yolo import VideoClassificationModel,VideoClassificationModel2,VideoClassificationModel3,VideoClassificationModel4,VideoClassificationModel5,VideoClassificationModel6,VideoClassificationModel7,VideoClassificationModel8,VideoClassificationModel9
from onnxsim import simplify
import onnx
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
def test():
  model = VideoClassificationModel9()
  
  pthfile = r'./runs/train-cls/exp7/weights/best.pt'
  # pretrained_model['model'].state_dict()
  loaded_model = torch.load(pthfile, map_location='cpu')
 
  model.load_state_dict(loaded_model['model'].state_dict() ,strict=False)
 
  #data type nchw
  dummy_input = torch.randn(1, 3, 640, 640)
  input_names = ["input"]
  output_names = ["output"]
  torch.onnx.export(model, dummy_input, "best_video.onnx", verbose=True, input_names=input_names, output_names=output_names, opset_version=12,  do_constant_folding=True)

  onnx_model = onnx.load("best_video.onnx")

  model_simp, check = simplify(onnx_model)

  if check:
    print("Model simplified successfully!")
  else:
    print("Simplified model could not be checked.")

  onnx.save(model_simp, "best_video_simplified7.onnx")
if __name__ == "__main__":
 test()