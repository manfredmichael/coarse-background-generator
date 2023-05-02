import argparse
import os, random
import utils
import cv2
import background_generator

def parse_arguments():
    parser = argparse.ArgumentParser(description='My Program')
    parser.add_argument('--input', type=str, help='Input root directory of your dataset', required=True)
    parser.add_argument('--output', type=str, help='Output root directory of the processed dataset', required=True)
    parser.add_argument('--backgrounds', type=str, default="backgrounds", help='Root directory of the background imaFes')
    args = parser.parse_args()
    return args


def main():
    # Parse arguments
    args = parse_arguments()

    # Print arguments
    print('Argument 1:', args.input)
    print('Argument 2:', args.output)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    image_files = utils.list_images(args.input)
    background_images = utils.list_images(args.backgrounds)

    print("loading background")
    # background_images = utils.load_images(background_images[:10])

    for file_path in image_files:
        image = cv2.imread(file_path)
        bg_image = cv2.imread(random.choice(background_images))

        result = background_generator.apply_background(image, bg_image)

        new_file_path = file_path.replace(args.input + "/", args.output + "/")
        print(f"{new_file_path} created")

        folder_path = "/".join(new_file_path.split("/")[:-1])
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
            print(f"{folder_path} created")

        cv2.imwrite(new_file_path, result)


if __name__ == '__main__':
    main()

