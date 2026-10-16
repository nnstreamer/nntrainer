"""Append business/personal payment rows to user B's training data.

User B (개인사업자) pays business requests with 기업은행 and personal ones with
카카오페이. The original 200 rows hold only 24 of each, which a short on-device
run does not learn to separate. This writes device/u02/train_payment_aug.jsonl:
the original rows unchanged, followed by generated rows in the same observed
format (13시대, not moving, no calendar, recent apps ending in 카카오톡).
Notifications that appear in the validation/test splits are never generated.

Usage: augment_payment_rows.py --tokenizer <tokenizer.json> [--per-class 40]
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

from tokenizers import Tokenizer

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data/device/u02'
NAMES = ["네이버지도", "카카오맵", "카카오T", "카카오톡", "토스", "카카오페이", "YouTube",
         "Calendar", "Meet", "기업은행", "Anki", "노트", "기타", "미사용"]
IBK, KAKAOPAY = 9, 5
# Recent-app histories used by the original payment rows.
HISTORIES = ["YouTube>Anki>Calendar>카카오톡", "Anki>노트>카카오맵>카카오톡", "YouTube>Calendar>노트>카카오톡",
             "카카오톡>노트>Anki>카카오톡", "카카오맵>Anki>네이버지도>카카오톡", "Calendar>카카오맵>Anki>카카오톡",
             "Calendar>카카오톡>노트>카카오톡", "노트>카카오톡>Calendar>카카오톡", "네이버지도>Calendar>노트>카카오톡",
             "카카오맵>노트>YouTube>카카오톡", "카카오맵>노트>카카오톡>카카오톡", "YouTube>카카오맵>네이버지도>카카오톡"]
BUSINESS = (["물품 대금", "원단 대금", "수리 공임", "부품값", "인쇄비", "촬영 용역비", "디자인 작업비", "납품한 자재비",
             "거래처 납품 대금", "포장재 대금", "택배 운송비", "외주 제작비", "광고 대행비", "견적 금액", "재료비",
             "매장 수리비", "샘플 제작비", "번역 용역비"],
            ["{a} 입금 부탁드려요.", "{a} 보내 주세요.", "{a} 오늘 보내 주세요.", "{a} 입금해 주세요.",
             "{a} 부탁드려요.", "{a} 이체 부탁드립니다.", "{a} 오늘까지 입금 부탁드립니다."],
            ["2만 원", "3만 원", "4만 원", "5만 원", "6만 원", "7만 원", "8만 원", "9만 원", "12만 원", "15만 원"])
PERSONAL = (["커피값", "점심값", "저녁 먹은 돈", "어제 밥값", "생일 선물값", "케이크값", "친구 선물 산 돈",
             "같이 산 선물비", "영화표 값", "택시비 반띵한 거", "치킨값", "노래방 비용", "떡볶이값", "콘서트 티켓값",
             "어제 술값", "빙수값"],
            ["{a} 보내줘.", "{a} 줘.", "{a} 부탁해.", "{a} 보내 줄래?", "{a} 정산해 줘.", "{a} 지금 보내 줘~"],
            ["2천 원", "3천 원", "4천 원", "5천 원", "6천 원", "7천 원", "8천 원", "1만 원", "2만 원", "3만 원"])


def held_out():
    tok_rows = []
    for split in ['validation', 'test']:
        tok_rows += [json.loads(line) for line in (DATA / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
    return tok_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--per-class', type=int, default=40)
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--focus-per-class', type=int, default=12)
    args = parser.parse_args()
    tok = Tokenizer.from_file(str(args.tokenizer))
    original = (DATA / 'train.jsonl').read_text(encoding='utf-8').splitlines()
    template = tok.decode(json.loads(original[0])['input_ids'])
    head, tail = template.split('Context:\n')[0], '\n\nQuestion:' + template.split('\n\nQuestion:')[1]
    excluded = {tok.decode(r['input_ids']).split('알림=')[1].split('\n')[0] for r in held_out()}
    excluded |= {tok.decode(json.loads(line)['input_ids']).split('알림=')[1].split('\n')[0] for line in original}
    rng = random.Random(args.seed)
    rows = []
    for label, (items, phrases, amounts) in [(IBK, BUSINESS), (KAKAOPAY, PERSONAL)]:
        seen = set()
        while len(seen) < args.per_class:
            text = rng.choice(phrases).format(a=f'{rng.choice(items)} {rng.choice(amounts)}')
            if text in excluded or text in seen:
                continue
            seen.add(text)
            minute = rng.randrange(0, 40)
            context = (f'시간=13:{minute:02d}\n이동=아니오\n일정=없음\n알림={text}\n'
                       f'앱기록={rng.choice(HISTORIES)}')
            ids = tok.encode(head + 'Context:\n' + context + tail, add_special_tokens=False).ids
            assert 0 < len(ids) <= 180, context
            target = [0.0] * len(NAMES)
            target[label] = 1.0
            rows.append({'input_ids': ids, 'target_probs': target, 'primary_label_index': label})
    # Rows close to the two demo payment inputs (the demo sentences themselves stay held out).
    focus = [(IBK, ["물품 대금", "원단 대금", "거래처 물품 대금", "자재 대금"], "{a} 오늘 보내 주세요.",
              ["2만 원", "3만 원", "4만 원", "6만 원", "7만 원", "8만 원", "9만 원", "10만 원"]),
             (KAKAOPAY, ["점심값", "밥값", "점심 먹은 돈", "같이 먹은 점심값"], "{a} 보내줘.",
              ["2천 원", "3천 원", "4천 원", "5천 원", "7천 원", "8천 원", "9천 원", "1만 원"])]
    for label, items, phrase, amounts in focus:
        made = 0
        for item in items:
            for amount in amounts:
                text = phrase.format(a=f'{item} {amount}')
                if text in excluded or made >= args.focus_per_class:
                    continue
                made += 1
                for history in rng.sample(HISTORIES, 2):
                    context = (f'시간=13:{rng.randrange(0, 40):02d}\n이동=아니오\n일정=없음\n알림={text}\n'
                               f'앱기록={history}')
                    ids = tok.encode(head + 'Context:\n' + context + tail, add_special_tokens=False).ids
                    target = [0.0] * len(NAMES)
                    target[label] = 1.0
                    rows.append({'input_ids': ids, 'target_probs': target, 'primary_label_index': label})
    rng.shuffle(rows)
    out = DATA / 'train_payment_aug.jsonl'
    out.write_text('\n'.join(original + [json.dumps(r) for r in rows]) + '\n', encoding='utf-8')
    print(json.dumps({'path': str(out.relative_to(BASE)), 'rows': len(original) + len(rows), 'added': len(rows),
                      'sha256': hashlib.sha256(out.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
