import subprocess
import time
import os
import json
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class KaggleTrainer:
    def __init__(self, username="johnantillanca"):
        self.username = username
        self.base_dir = Path(__file__).parent.parent
        self.scripts_dir = self.base_dir / "scripts"
        self.models_dir = self.base_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

    def _run_cmd(self, cmd):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Command failed: {' '.join(cmd)}\nError: {result.stderr}")
        return result

    def train(self, subject, notebook_path):
        """Full training cycle: Push -> Wait -> Download -> Deploy"""
        subject_slug = subject.replace("_", "-").lower()
        kernel_slug = f"cpt-{subject_slug}-training"
        full_id = f"{self.username}/{kernel_slug}"
        
        notebook_path = Path(notebook_path)
        if not notebook_path.exists():
            logger.error(f"Notebook not found: {notebook_path}")
            return False

        # Create temporary directory for kernel to avoid metadata conflicts
        temp_dir = self.scripts_dir / f"tmp_{subject}"
        temp_dir.mkdir(exist_ok=True)
        
        # Copy notebook to temp dir
        shutil.copy(notebook_path, temp_dir / notebook_path.name)
        
        # Prepare metadata
        metadata = {
            "id": full_id,
            "title": f"CPT {subject.capitalize()} Training",
            "code_file": notebook_path.name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": []
        }
        
        metadata_file = temp_dir / "kernel-metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Push kernel
        logger.info(f"Pushing kernel {full_id}...")
        res = self._run_cmd(["kaggle", "kernels", "push", "-p", str(temp_dir)])
        if res.returncode != 0:
            # Handle potential metadata error or already existing slug
            if "already exists" in res.stderr:
                logger.info("Kernel exists, pushing update...")
            else:
                return False

        # Poll status
        logger.info(f"Waiting for kernel {full_id} to complete...")
        max_wait = 1200 # 20 minutes (for heavier training)
        start_time = time.time()
        
        last_status = ""
        while time.time() - start_time < max_wait:
            status_res = self._run_cmd(["kaggle", "kernels", "status", full_id])
            if status_res.returncode == 0:
                status = status_res.stdout.lower()
                if "complete" in status:
                    logger.info("Kernel complete!")
                    break
                elif "error" in status:
                    logger.error(f"Kernel failed: {status}")
                    return False
                else:
                    if status != last_status:
                        logger.info(f"Status: {status.strip()}")
                        last_status = status
            else:
                logger.warning("Failed to get kernel status, retrying...")
            
            time.sleep(30)
        else:
            logger.error("Timeout waiting for kernel completion")
            return False

        # Download output
        logger.info(f"Downloading output for {full_id}...")
        down_res = self._run_cmd(["kaggle", "kernels", "output", full_id, "-p", str(temp_dir)])
        if down_res.returncode != 0:
            logger.error("Failed to download output")
            return False

        # Move .pt file to models/
        # We look for the most recent or standard name
        pt_files = list(temp_dir.glob("*.pt"))
        if not pt_files:
            logger.error("No .pt files found in Kaggle output")
            return False
        
        for pt_file in pt_files:
            # Standardize name: subject_tabular_filter.pt
            target_name = f"{subject}_tabular_filter.pt"
            target = self.models_dir / target_name
            shutil.copy(pt_file, target)
            logger.info(f"Deployed model to {target}")

        # Cleanup
        shutil.rmtree(temp_dir)
        return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    import sys
    if len(sys.argv) == 3:
        trainer = KaggleTrainer()
        trainer.train(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python3 kaggle_trainer.py [subject] [notebook_path]")
