# -*- coding: utf-8 -*-
"""
approve_model.py

학습된 모델(train.py의 결과물)을 사람이 직접 검토한 뒤 명시적으로
승인하는 CLI 스크립트. RiskModel.is_reliable()은 통계적 조건을 만족해도
is_approved가 True가 되기 전까지는 절대 True를 반환하지 않으므로,
이 스크립트를 거치지 않은 모델은 자동으로 서비스에 반영되지 않는다.

의도적으로 이 스크립트는 자동화하지 않았다 — 승인은 사람이
training_report.json을 두 눈으로 확인하고, 이름을 남기고, 그 결정을
기록하는 행위여야 한다. CI/CD 파이프라인에서 자동으로 호출되도록
만들지 말 것.

사용법
------
python -m risk_model.approve_model --model-dir ./trained_model --approver "홍길동"

내부 동작
---------
1. model_dir의 meta.json과 training_report.json을 읽어 화면에 보여준다.
2. is_validated_better_than_legacy가 False면 애초에 승인 대상이 아니므로
   경고 후 종료한다 (통계적으로 legacy보다 낫다고 확인되지 않은 모델은
   사람이 승인해도 의미가 없다 — 이 조건 자체는 우회 불가).
3. 승인자 이름과 함께 y/n 확인을 받는다.
4. 승인하면 meta.json에 is_approved=True, approved_by, approved_at을
   기록하고, 승인 이력을 approval_log.jsonl에 한 줄 추가한다(감사 추적용,
   기존 승인 기록을 덮어쓰지 않고 계속 누적).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def approve_model(model_dir: str, approver: str, force: bool = False) -> bool:
    path = Path(model_dir)
    meta_path = path / "meta.json"
    report_path = path / "training_report.json"

    if not meta_path.exists():
        print(f"[오류] {meta_path} 가 없습니다. train.py로 먼저 모델을 학습하세요.")
        return False

    meta = json.loads(meta_path.read_text())

    print("=" * 60)
    print(f"모델 경로: {path}")
    print(f"학습 샘플 수: {meta.get('trained_on_n')}")
    print(f"legacy 대비 통계적 우위 확인됨: {meta.get('is_validated_better_than_legacy')}")
    print(f"현재 승인 상태: {meta.get('is_approved', False)}")
    if meta.get("is_approved"):
        print(f"  (기존 승인자: {meta.get('approved_by')}, 시각: {meta.get('approved_at')})")

    if report_path.exists():
        report = json.loads(report_path.read_text())
        print("-" * 60)
        print("training_report.json 요약:")
        for k, v in report.items():
            if k == "feature_importances":
                continue
            print(f"  {k}: {v}")
    else:
        print("[경고] training_report.json이 없습니다 — 검증 근거를 확인할 수 없습니다.")

    print("=" * 60)

    if not meta.get("is_validated_better_than_legacy", False):
        print(
            "[승인 불가] 이 모델은 legacy 대비 통계적 우위가 확인되지 않았습니다 "
            "(is_validated_better_than_legacy=False). 이 조건은 승인으로 우회할 수 없습니다. "
            "더 많은 라벨 데이터로 재학습 후 다시 시도하세요."
        )
        return False

    if not force:
        answer = input(f"\n'{approver}' 님, 위 내용을 검토했고 이 모델을 프로덕션에 반영하는 데 동의합니까? (yes/no): ")
        if answer.strip().lower() not in ("yes", "y"):
            print("승인이 취소되었습니다. is_approved는 변경되지 않습니다.")
            return False

    meta["is_approved"] = True
    meta["approved_by"] = approver
    meta["approved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    log_path = path / "approval_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "approved_by": approver,
            "approved_at": meta["approved_at"],
            "trained_on_n": meta.get("trained_on_n"),
        }, ensure_ascii=False) + "\n")

    print(f"\n승인 완료. {log_path}에 이력이 기록되었습니다.")
    print("이제 RiskModel.is_reliable()이 이 모델에 대해 True를 반환할 수 있습니다 "
          "(단, 다른 통계적 조건도 계속 만족해야 함).")
    return True


def revoke_approval(model_dir: str, revoker: str, reason: str) -> None:
    """이미 승인된 모델의 승인을 취소한다 (예: 배포 후 이상 징후 발견 시
    즉시 legacy로 되돌리기 위한 비상 절차)."""
    path = Path(model_dir)
    meta_path = path / "meta.json"
    meta = json.loads(meta_path.read_text())

    meta["is_approved"] = False
    meta["revoked_by"] = revoker
    meta["revoked_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["revoke_reason"] = reason
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    log_path = path / "approval_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "revoked_by": revoker, "revoked_at": meta["revoked_at"], "reason": reason,
        }, ensure_ascii=False) + "\n")

    print(f"승인이 취소되었습니다. 이제 이 모델은 즉시 legacy(v2)로 폴백됩니다.")


def main() -> None:
    parser = argparse.ArgumentParser(description="학습된 모델 수동 승인/취소")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--approver", help="승인자 이름 (승인 시 필수)")
    parser.add_argument("--revoke", action="store_true", help="승인 취소 모드")
    parser.add_argument("--reason", help="승인 취소 사유 (--revoke 시 필수)")
    parser.add_argument("--force", action="store_true", help="확인 프롬프트 생략 (스크립트 자동화용, 주의해서 사용)")
    args = parser.parse_args()

    if args.revoke:
        if not args.approver or not args.reason:
            print("[오류] --revoke 사용 시 --approver(취소자)와 --reason이 모두 필요합니다.")
            return
        revoke_approval(args.model_dir, args.approver, args.reason)
    else:
        if not args.approver:
            print("[오류] --approver(승인자 이름)이 필요합니다.")
            return
        approve_model(args.model_dir, args.approver, force=args.force)


if __name__ == "__main__":
    main()
