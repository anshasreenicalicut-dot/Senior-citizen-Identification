
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_command(module, extra_args=None):
    command = [
        sys.executable,
        "-m",
        module,
    ]

    if extra_args:
        command.extend(extra_args)

    subprocess.run(command, cwd=ROOT, check=False)


def main():
    while True:
        print("\n========================================")
        print(" SENIOR CITIZEN IDENTIFICATION SYSTEM")
        print("========================================")
        print("1. Prepare UTKFace dataset")
        print("2. Train age and gender CNN")
        print("3. Detect using webcam")
        print("4. Detect using video file")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ").strip()

        if choice == "1":
            run_command("src.prepare_dataset")

        elif choice == "2":
            run_command("src.train_model")

        elif choice == "3":
            run_command("src.detect_video", ["--source", "0"])

        elif choice == "4":
            video_path = input(
                "Enter video path: "
            ).strip().strip('"')

            if not Path(video_path).is_absolute():
                video_path = str((ROOT / video_path).resolve())

            run_command(
                "src.detect_video",
                ["--source", video_path],
            )

        elif choice == "5":
            print("Exiting project.")
            break

        else:
            print("Invalid choice. Enter a number from 1 to 5.")


if __name__ == "__main__":
    main()
