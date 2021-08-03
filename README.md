*Data Science Enthusiast* 
## [Project1: Face Recognition using LBPHF Recognizer](https://github.com/nikita10110/ml_projects/tree/master/face_recogniton_using_LBPH)
For this project I have build a *Face recognizer using LBPHF* to identify the face in the image. This could be useful for the person in different fields. For example if we train our model on the dataset of company employee and when we give an image and try to recognize the person if he\she company employee or not. This is underlying model for building something with capabilities.


To detect an face from the image, I have applied *Haar Cascade model*. Once the faces are detected. Further, i have applied face recognition using the *local Binary Pattern Histogram Face Recognizer*.

I am able to get the model to predict with 90% accuracy after minimal tuning. For most of the cases this would meet the requirement. 


## [Project2: Face Recognition using Facenet](https://github.com/nikita10110/ml_projects/tree/master/face_recognition_using_facenet)
For this Project I have build a *Face recogniton using Facent* to recognize the person in the image. For the model i have used the *inceptionresent network* which is pretrained on vgg-16 dataset. I have performed some hypertuning like removing the end layers, adding the layers according to the rquirements, hypertuning learning rate.


`def get_model():
  model_ft = InceptionResnetV1(pretrained='vggface2', classify=False, num_classes = len(class_names))
  layer_list = list(model_ft.children())[-5:]
  model_ft = nn.Sequential(*list(model_ft.children())[:-5])
  for param in model_ft.parameters():
      param.requires_grad = False
  model_ft.avgpool_1a = nn.AdaptiveAvgPool2d(output_size=1)
  model_ft.last_linear = nn.Sequential(
      Flatten(),
      nn.Linear(in_features=1792, out_features=512, bias=False),
      normalize())
  model_ft.logits = nn.Linear(layer_list[3].num_features, len(class_names))

  model_ft.softmax = nn.Softmax(dim=1)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)
  model_ft = model_ft.to(device)
  return model_ft`
  
  
I am able to get the model to predict with an accuracy of 90% after minimal tuning. 

