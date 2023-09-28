"""Script to make small random ImageBind testing models."""

import os

import torch

from imagebind.models import imagebind_model


def save_model_state_dict(model, output_dir, name):
    model_state_dict = model.state_dict()
    model_save_path = os.path.join(output_dir, f"{name}.pth")
    torch.save(model_state_dict, model_save_path)


def main(args):
    # Create text encoder
    torch.manual_seed(0)
    text_encoder = imagebind_model.imagebind_test_text_encoder()
    save_model_state_dict(text_encoder, args.model_output_dir, "text_encoder")

    # Create vision encoder
    torch.manual_seed(0)
    vision_encoder = imagebind_model.imagebind_test_vision_encoder()
    save_model_state_dict(vision_encoder, args.model_output_dir, "vision_encoder")

    # Create audio encoder
    torch.manual_seed(0)
    audio_encoder = imagebind_model.imagebind_test_audio_encoder()
    save_model_state_dict(audio_encoder, args.model_output_dir, "audio_encoder")

    # Create depth encoder
    torch.manual_seed(0)
    depth_encoder = imagebind_model.imagebind_test_depth_encoder()
    save_model_state_dict(depth_encoder, args.model_output_dir, "depth_encoder")

    # Create thermal encoder
    torch.manual_seed(0)
    thermal_encoder = imagebind_model.imagebind_test_thermal_encoder()
    save_model_state_dict(thermal_encoder, args.model_output_dir, "thermal_encoder")

    # Create IMU encoder
    torch.manual_seed(0)
    imu_encoder = imagebind_model.imagebind_test_imu_encoder()
    save_model_state_dict(imu_encoder, args.model_output_dir, "imu_encoder")

    # Create full test ImageBind model
    torch.manual_seed(0)
    imagebind_test = imagebind_model.imagebind_test()
    save_model_state_dict(imagebind_test, args.model_output_dir, "imagebind_test")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_output_dir",
        type=str,
        default="/models",
        help="Output directory for the test models.",
    )

    args = parser.parse_args()

    main(args)