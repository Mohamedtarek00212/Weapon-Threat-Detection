from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from weapon_threat_detection.training_launcher import main


if __name__ == "__main__":
    main()
