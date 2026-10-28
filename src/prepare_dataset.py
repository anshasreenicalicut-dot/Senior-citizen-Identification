
from pathlib import Path
import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

SOURCE_FOLDERS = [
    ROOT / "crop_part1",
    ROOT / "UTKFace",
    ROOT / "utkface_aligned_cropped",
]

OUTPUT_DIR = ROOT / "dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_filename(path):
    # Standard UTKFace filename:
    # age_gender_race_datetime.jpg
    parts = path.stem.split("_")

    if len(parts) < 4:
        return None

    try:
        age = int(parts[0])
        gender_code = int(parts[1])
    except ValueError:
        return None

    if not 0 <= age <= 116:
        return None

    if gender_code not in (0, 1):
        return None

    return age, gender_code


def main():
    records = []
    seen_hashes = set()
    skipped = 0

    for folder in SOURCE_FOLDERS:
        if not folder.exists():
            print(f"Folder not found (skipping): {folder}")
            continue

        print(f"\nScanning: {folder}")

        for path in folder.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in {".jpg", ".jpeg"}:
                continue

            labels = parse_filename(path)
            if labels is None:
                skipped += 1
                continue

            try:
                file_hash = sha256_file(path)
            except OSError:
                skipped += 1
                continue

            # Remove byte-identical images across folders.
            if file_hash in seen_hashes:
                continue
            seen_hashes.add(file_hash)

            age, gender_code = labels

            records.append({
                "image_path": str(path.resolve()),
                "age": age,
                "gender": gender_code,
                "senior": int(age > 60),
            })

    df = pd.DataFrame(records)

    if len(df) < 10:
        raise RuntimeError(
            f"Only {len(df)} valid images found. "
            "Check the folder paths and UTKFace filenames."
        )

    # Split 70% train, 15% validation, 15% test.
    # Stratification preserves the senior/non-senior ratio.
    train_df, remaining_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["senior"],
    )

    val_df, test_df = train_test_split(
        remaining_df,
        test_size=0.50,
        random_state=42,
        stratify=remaining_df["senior"],
    )

    for name, split in [
        ("train.csv", train_df),
        ("val.csv", val_df),
        ("test.csv", test_df),
    ]:
        split.to_csv(OUTPUT_DIR / name, index=False)

    print("\nDataset preparation complete.")
    print(f"Unique images: {len(df)}")
    print(f"Skipped invalid filenames: {skipped}")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")
    print("\nAge-derived senior labels:")
    print(df["senior"].value_counts().rename(
        index={0: "Non-senior (age <= 60)", 1: "Senior (age > 60)"}
    ))
    print(f"\nCSV files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
