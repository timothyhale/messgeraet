import argparse
from dataclasses import dataclass

@dataclass
class Settings:
    threshold: int
    image_path: str
    output_step: str
    reference_object: int


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(
        description="Image pipeline configuration via CLI arguments"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=int,
        default=125,
        help="Threshold value for binarization (default: 125)"
    )
    parser.add_argument(
        "--image-path", "-i",
        type=str,
        required=True,
        help="Path to the input image file"
    )
    parser.add_argument(
        "--output-step", "-s",
        type=str,
        choices=["last", "step1", "step2", "step3"], # TODO: Replace with actual steps
        default="last",
        help="Pipeline step for which to save the output image (default: last)"
    )
    parser.add_argument(
        "--reference-object", "-r",
        type=int,
        choices=[2, 1, 50],
        required=True,
        help="Reference object for scale (2€, 1€ or 50 cent)"
    )

    args = parser.parse_args()
    return Settings(
        threshold=args.threshold,
        image_path=args.image_path,
        output_step=args.output_step,
        reference_object=args.reference_object
    )


def main():
    settings = parse_args()
    print("Loaded settings:")
    print(settings)

if __name__ == "__main__":
    main()
