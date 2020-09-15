from absl import app, flags, logging
from absl.flags import FLAGS
import os
import json
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch

flags.DEFINE_string('model', r"C:\Users\Owner\Desktop\Classification\models\21_56.pth", 'Path to model', short_name='m')
flags.DEFINE_string('folder', None, 'Path to image folder', short_name='f')

def main(_argv):

    print(f"Reading images from: {FLAGS.folder}")
    classes = ['female', 'male']

    def get_prediction(d, model):
        prediction = {}
        for f in tqdm(os.listdir(d)):
            img = transform(Image.open(os.path.join(d,f))).to(device).view(1, 3, 224, 224)
            output = model(img)
            prediction[f] = classes[int(output.argmax())]
        return prediction

    #Loading Model
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(FLAGS.model)
    model = model.to(device)

    #Saving to json
    with open('process_results.json', 'w') as fp:
        json.dump(get_prediction(FLAGS.folder, model), fp)
    logging.info("Done")

if __name__ == '__main__':
    try:
        flags.mark_flag_as_required('folder')
        app.run(main)
    except SystemExit:
        pass