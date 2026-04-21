"""
Session Manager — persists state to Hugging Face Private Dataset repo.
Survives container restarts. All session state and uploaded files live here.
"""

import os
import json
import shutil
import logging
from datetime import datetime, timezone
from typing import Optional
from huggingface_hub import HfApi, hf_hub_download, upload_file as hf_upload_file

log = logging.getLogger(__name__)

HF_TOKEN   = os.environ["HF_TOKEN"]
HF_REPO_ID = os.environ.get("HF_DATASET_REPO", "")
REPO_TYPE  = "dataset"


class SessionManager:
    def __init__(self):
        self.api = HfApi(token=HF_TOKEN)
        self.repo_id = HF_REPO_ID
        if not self.repo_id:
            raise ValueError(
                "HF_DATASET_REPO env var not set. "
                "Add it in Space Settings → Repository Secrets."
            )

    def _session_key(self, owner_id: int) -> str:
        return f"sessions/session_{owner_id}.json"

    # ── Write ────────────────────────────────────────────────────────────────

    def init_session(self, owner_id: int, video_hf_path: str, timestamp: float) -> dict:
        session = {
            "owner_id": owner_id,
            "video_path": video_hf_path,
            "timestamp": timestamp,
            "products": [],
            "status": "awaiting_products",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        self.save_session(owner_id, session)
        return session

    def save_session(self, owner_id: int, session: dict):
        local_path = f"/tmp/session_{owner_id}.json"
        with open(local_path, "w") as f:
            json.dump(session, f, indent=2)
        hf_upload_file(
            path_or_fileobj=local_path,
            path_in_repo=self._session_key(owner_id),
            repo_id=self.repo_id,
            repo_type=REPO_TYPE,
            token=HF_TOKEN
        )
        log.info(f"Session saved [{owner_id}] status={session.get('status')}")

    def update_status(self, owner_id: int, status: str):
        session = self.load_session(owner_id)
        if session:
            session["status"] = status
            self.save_session(owner_id, session)

    # ── Read ─────────────────────────────────────────────────────────────────

    def load_session(self, owner_id: int) -> Optional[dict]:
        try:
            local_dir = "/tmp/hf_sessions"
            os.makedirs(local_dir, exist_ok=True)
            path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self._session_key(owner_id),
                repo_type=REPO_TYPE,
                token=HF_TOKEN,
                local_dir=local_dir,
                force_download=True   # always get freshest copy
            )
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            log.debug(f"No session for {owner_id}: {e}")
            return None

    # ── Delete ───────────────────────────────────────────────────────────────

    def clear_session(self, owner_id: int):
        try:
            self.api.delete_file(
                path_in_repo=self._session_key(owner_id),
                repo_id=self.repo_id,
                repo_type=REPO_TYPE,
                token=HF_TOKEN
            )
            log.info(f"Session cleared [{owner_id}]")
        except Exception as e:
            log.debug(f"Clear session (non-fatal): {e}")

    # ── File upload/download ─────────────────────────────────────────────────

    def upload_file(self, local_path: str, repo_path: str) -> str:
        """Upload a file to HF Dataset repo. Returns hf:// path string."""
        hf_upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=self.repo_id,
            repo_type=REPO_TYPE,
            token=HF_TOKEN
        )
        hf_path = f"hf://{self.repo_id}/{repo_path}"
        log.info(f"Uploaded → {hf_path}")
        return hf_path

    def download_hf_file(self, hf_path: str, local_dest: str) -> str:
        """Download a file from its hf:// path to local_dest."""
        # Strip hf://repo_id/ prefix to get the repo-relative path
        prefix = f"hf://{self.repo_id}/"
        if hf_path.startswith(prefix):
            repo_path = hf_path[len(prefix):]
        else:
            repo_path = hf_path  # fallback

        local_dir = "/tmp/hf_downloads"
        os.makedirs(local_dir, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=self.repo_id,
            filename=repo_path,
            repo_type=REPO_TYPE,
            token=HF_TOKEN,
            local_dir=local_dir,
            force_download=True
        )
        os.makedirs(os.path.dirname(os.path.abspath(local_dest)), exist_ok=True)
        shutil.copy(downloaded, local_dest)
        log.info(f"Downloaded {hf_path} → {local_dest}")
        return local_dest
