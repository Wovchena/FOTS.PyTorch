import argparse
import torch
import logging
import pathlib
import traceback
from FOTS.base.base_model import BaseModel
from FOTS.model.model import FOTSModel
from FOTS.utils.bbox import Toolbox

logging.basicConfig(level=logging.DEBUG, format='')


def main(args:argparse.Namespace):
    model_path = args.model
    input_dir = args.input_dir
    output_dir = args.output_dir
    with_image = True if output_dir else False

    model = FOTSModel().to(torch.device("cuda"))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        for image_fn in input_dir.glob('*.jpg'):
            ploy, im = Toolbox.predict(image_fn, model, with_image, output_dir, with_gpu=True)


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='Model eval')
    parser.add_argument('-m', '--model', default=None, type=pathlib.Path, required=True,
                        help='path to model')
    parser.add_argument('-o', '--output_dir', default=None, type=pathlib.Path,
                        help='output dir for drawn images')
    parser.add_argument('-i', '--input_dir', default=None, type=pathlib.Path, required=False,
                        help='dir for input images')
    args = parser.parse_args()
    main(args)
