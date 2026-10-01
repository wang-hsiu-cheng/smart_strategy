import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

import config
from metadata_utils import sync_to_huggingface_from_metadata


def run_cmd(cmd: list):
    print(f"[*] Executing: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser(description="讀取 .env 設定發布模型至 HF 與 GitHub")

    # 預設值皆由 config (.env) 提供，無需手動輸入
    parser.add_argument(
        "--model",
        type=str,
        default=config.PUBLISH_MODEL_PATH,
        help=f"模型權重路徑 (.zip)，預設: {config.PUBLISH_MODEL_PATH}"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=config.HF_MODEL_REPO,
        help=f"Hugging Face 目標倉庫，預設: {config.HF_MODEL_REPO}"
    )
    parser.add_argument(
        "--github-repo",
        type=str,
        default=config.GITHUB_REPO,
        help=f"GitHub 專案庫名稱，預設: {config.GITHUB_REPO}"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="發布標籤版本號（未指定時由快照自動推斷，如 yellow-v800k）"
    )

    # 布林值開關：支援由 .env 設定預設值，亦可手動加上 --no-xxx 關閉
    parser.add_argument("--include-logs", dest="include_logs", action="store_true")
    parser.add_argument("--no-include-logs", dest="include_logs", action="store_false")
    parser.set_defaults(include_logs=config.PUBLISH_INCLUDE_LOGS)

    parser.add_argument("--git-push", dest="git_push", action="store_true")
    parser.add_argument("--no-git-push", dest="git_push", action="store_false")
    parser.set_defaults(git_push=config.PUBLISH_GIT_PUSH)

    args = parser.parse_args()

    model_path = Path(args.model)
    meta_path = model_path.with_suffix(".json")

    # 檢查檔案完整性
    if not model_path.exists():
        print(f"[!] 錯誤：找不到模型權重檔案: {model_path}")
        sys.exit(1)

    if not meta_path.exists():
        print(f"[!] 錯誤：找不到同名參數快照檔案: {meta_path}")
        print("[!] 請確認該模型是否由具備快照功能的 train 腳本所生成。")
        sys.exit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 自動推算版本標籤（若使用者未手動指定）
    if args.tag:
        tag_name = args.tag
    else:
        steps_k = metadata.get("total_timesteps", 0) // 1000
        color = metadata.get("color", "model")
        tag_name = f"{color}-v{steps_k}k"

    print("=" * 60)
    print(f"[*] 發布模型目標檔案 : {model_path.name}")
    print(f"[*] 結算版本標籤 (Tag): {tag_name}")
    print(f"[*] Hugging Face 倉庫: {args.repo or '(未啟用)'}")
    print(f"[*] GitHub 專案名稱  : {args.github_repo or '(未啟用)'}")
    print(f"[*] 包含 TensorBoard : {args.include_logs}")
    print(f"[*] 自動 Git Tag Push: {args.git_push}")
    print("=" * 60)

    # 1. 處理 TensorBoard 日誌
    tb_log_dir = None
    if args.include_logs:
        tb_log_name = metadata.get("tb_log_name")
        if tb_log_name:
            candidate_dir = os.path.join(config.TENSORBOARD_LOG_DIR, tb_log_name)
            if os.path.exists(candidate_dir):
                tb_log_dir = candidate_dir

    # 2. 同步至 Hugging Face Hub
    if args.repo:
        sync_to_huggingface_from_metadata(
            repo_id=args.repo,
            model_path=str(model_path),
            metadata_path=str(meta_path),
            tag_name=tag_name,
            tb_log_dir=tb_log_dir,
            github_repo=args.github_repo,
            private=True
        )
    else:
        print("[!] 未設定 HF_MODEL_REPO，跳過 Hugging Face 上傳。")

    # 3. 提交 Git 標籤並推送至 GitHub
    if args.git_push:
        print("\n[*] 正在同步 Git 倉庫與 Tag...")
        try:
            run_cmd(["git", "add", str(meta_path)])
            # 若無檔案變更則略過 commit
            commit_result = subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                check=False
            )
            if commit_result.returncode != 0:
                run_cmd(["git", "commit", "-m", f"chore(release): snapshot for {tag_name}"])

            run_cmd(["git", "tag", "-a", tag_name, "-m", f"Release RL policy checkpoint: {tag_name}"])
            run_cmd(["git", "push", "origin", "main", "--tags"])
            print(f"[+] GitHub 標籤 {tag_name} 推送完成！")
        except subprocess.CalledProcessError as e:
            print(f"[!] Git 操作失敗: {e}")


if __name__ == "__main__":
    main()