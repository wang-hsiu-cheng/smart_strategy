import os
import json
import tarfile
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional
from huggingface_hub import HfApi, create_repo


def get_git_commit_hash() -> str:
    """取得當前程式碼的 Git Commit SHA"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
    except Exception:
        return "unknown"

def resolve_step_lineage(pretrained_path: Optional[str], delta_timesteps: int) -> tuple[int, int]:
    """
    解析父模型步數，計算出 (base_timesteps, cumulative_timesteps)
    """
    if not pretrained_path or not os.path.exists(pretrained_path):
        return 0, delta_timesteps

    # 1. 優先從父模型的 .json 快照取得 cumulative_timesteps
    parent_json = Path(pretrained_path).with_suffix(".json")
    if parent_json.exists():
        try:
            with open(parent_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                base = data.get("cumulative_timesteps", data.get("total_timesteps", 0))
                return base, base + delta_timesteps
        except Exception:
            pass

    # 2. 若無 .json，從父模型檔名正則推斷 (例如 800K -> 800,000, 1.2M -> 1,200,000)
    stem = Path(pretrained_path).stem
    match_m = re.search(r"([\d\.]+)M", stem, re.IGNORECASE)
    match_k = re.search(r"(\d+)K", stem, re.IGNORECASE)
    base = 0
    if match_m:
        base = int(float(match_m.group(1)) * 1_000_000)
    elif match_k:
        base = int(match_k.group(1)) * 1_000

    return base, base + delta_timesteps

def tag_latest_tb_event(tb_log_dir: str, base_steps: int, cumulative_steps: int) -> Optional[str]:
    """
    將最新產生的 events.out.tfevents.* 檔案重新命名，加上明確的步數區間標籤
    """
    log_path = Path(tb_log_dir)
    if not log_path.exists():
        return None

    # 找出該目錄下未命名的原生 tfevents 檔案
    event_files = sorted(
        [f for f in log_path.glob("events.out.tfevents.*") if "step_" not in f.name],
        key=lambda x: x.stat().st_mtime
    )

    if not event_files:
        return None

    latest_event = event_files[-1]
    b_k = f"{base_steps // 1000}k" if base_steps >= 1000 else f"{base_steps}"
    c_k = f"{cumulative_steps // 1000}k" if cumulative_steps >= 1000 else f"{cumulative_steps}"
    
    # 保持以 events.out.tfevents. 為前綴，確保 TensorBoard 依然可讀取
    new_name = f"{latest_event.name}.step_{b_k}_to_{c_k}"
    new_path = latest_event.with_name(new_name)
    latest_event.rename(new_path)
    print(f"[*] TensorBoard 日誌檔已標註區間: {new_path.name}")
    return str(new_path)

def save_training_metadata(
    cfg: Dict[str, Any], 
    save_path: str,
    base_timesteps: int,
    cumulative_timesteps: int,
    tagged_event_file: Optional[str] = None
) -> Path:
    """保存明確拆解步數的中繼資料快照"""
    metadata = {
        "git_commit": get_git_commit_hash(),
        "color": cfg.get("color"),
        "version_group": cfg.get("tb_log_name"),
        # 明確區分步數語意
        "base_timesteps": base_timesteps,
        "delta_timesteps": cfg.get("total_timesteps"),
        "cumulative_timesteps": cumulative_timesteps,
        "reset_timesteps": cfg.get("reset_timesteps"),
        "pretrained_base": cfg.get("pretrained_path"),
        "enemy_model_used": cfg.get("enemy_model_path"),
        "associated_event_file": Path(tagged_event_file).name if tagged_event_file else None,
        "ppo_hyperparameters": cfg.get("ppo_params", {}),
    }

    meta_path = Path(save_path).with_suffix(".json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"[*] 完整血統快照已更新: {meta_path.name}")
    return meta_path

def sync_to_huggingface_from_metadata(
    repo_id: str,
    model_path: str,
    metadata_path: str,
    tag_name: str,
    tb_log_dir: Optional[str] = None,
    private: bool = True
) -> None:
    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)
    model_file = Path(model_path)
    meta_file = Path(metadata_path)

    print(f"\n[*] 連線至 Hugging Face Hub: {repo_id}...")
    create_repo(repo_id=repo_id, repo_type="model", token=token, exist_ok=True, private=private)

    # 1. 上傳模型檔案 (.zip)
    print(f"[*] 上傳模型權重: {model_file.name}")
    api.upload_file(
        path_or_fileobj=str(model_file),
        path_in_repo=model_file.name,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"feat: upload model {tag_name}"
    )

    # 2. 上傳對應的參數快照 (.json)
    if meta_file.exists():
        print(f"[*] 上傳參數快照: {meta_file.name}")
        api.upload_file(
            path_or_fileobj=str(meta_file),
            path_in_repo=meta_file.name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"docs: upload metadata snapshot for {tag_name}"
        )

    # 3. 封存上傳 TensorBoard logs (若有指定)
    if tb_log_dir and os.path.exists(tb_log_dir):
        tar_path = model_file.parent / f"{tag_name}_tb_logs.tar.gz"
        print(f"[*] 打包 TensorBoard 日誌: {tb_log_dir}...")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(tb_log_dir, arcname=Path(tb_log_dir).name)

        api.upload_file(
            path_or_fileobj=str(tar_path),
            path_in_repo=f"logs/{tar_path.name}",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"logs: archive logs for {tag_name}"
        )
        tar_path.unlink()

    # 4. 建立版本 Tag
    try:
        api.create_tag(repo_id=repo_id, tag=tag_name, repo_type="model")
        print(f"[+] Hugging Face Tag '{tag_name}' 建立成功！")
    except Exception as e:
        print(f"[!] Tag 建立跳過 (可能已存在): {e}")

    print(f"[+] 同步完成: https://huggingface.co/{repo_id}\n")