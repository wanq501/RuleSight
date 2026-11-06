#!/usr/bin/env bash
# Rico for RuleSight, run in the AutoDL terminal. Route A: Voxel51/rico keeps one file per screenshot, named by its Rico id — only the
# 3,500 needed files are fetched (about 0.4 GB). Route B (only if A cannot be used): the original Rico archive, 32 connections at once.
set -u
ROOT=${ROOT:-/root/autodl-tmp/atelier}; D=$ROOT/rs_data/data; PY=${PY:-$ROOT/rs_venv/bin/python}
export MIRROR=${MIRROR:-https://hf-mirror.com}; GS=${GS:-https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz}
T=$D/rico_tmp; mkdir -p "$D/rico" "$T"; H="$D/rico_v51.py"
n_ids() { "$PY" -c "import csv,sys; print(len({r.get('rico_id') for r in csv.DictReader(open(sys.argv[1],encoding='utf-8',errors='replace')) if r.get('rico_id')}))" "$1" 2>/dev/null || echo 0; }
[ "$(n_ids "$D/uicrit_public.csv")" -lt 900 ] && [ -f "$ROOT/rulesight/bundled/uicrit_public.csv" ] && cp "$ROOT/rulesight/bundled/uicrit_public.csv" "$D/uicrit_public.csv"
[ "$(n_ids "$D/uicrit_public.csv")" -lt 900 ] && [ -f "$ROOT/rulesight.zip" ] && "$PY" -c "import zipfile,sys; open(sys.argv[2],'wb').write(zipfile.ZipFile(sys.argv[1]).read('rulesight/bundled/uicrit_public.csv'))" "$ROOT/rulesight.zip" "$D/uicrit_public.csv" 2>/dev/null
[ "$(n_ids "$D/uicrit_public.csv")" -lt 900 ] && (source /etc/network_turbo >/dev/null 2>&1; curl -sfL -m 120 -o "$D/uicrit_public.csv" https://raw.githubusercontent.com/google-research-datasets/uicrit/main/uicrit_public.csv)
echo "UICrit 屏号：$(n_ids "$D/uicrit_public.csv") 个"; [ "$(n_ids "$D/uicrit_public.csv")" -lt 900 ] && { echo "拿不到 UICrit 的屏号列表，停止"; exit 1; }
"$PY" "$H" "$D" status
watch_dl() {   # $1 = pid to watch, $2 = what is counted ("files" in rico/ or "bytes" in rico_tmp/), $3 = total
  local start_n start_b n b prev_b=0 t=0
  start_n=$(ls "$D/rico" | grep -cE '\.(jpg|png)$'); start_b=$(du -sb "$T" 2>/dev/null | cut -f1)
  while kill -0 "$1" 2>/dev/null; do
    sleep 5; t=$((t+5))
    if [ "$2" = files ]; then n=$(( $(ls "$D/rico" | grep -cE '\.(jpg|png)$') - start_n )); b=$(du -sm "$D/rico" 2>/dev/null | cut -f1); echo "  已下 $n / $3 张（rico 目录 $b MB，已用 ${t} 秒）"
    else b=$(du -sb "$T" 2>/dev/null | cut -f1); echo "  已下 $(( b / 1048576 )) / $(( $3 / 1048576 )) MB，$(( (b - prev_b) / 5 / 1024 )) KB/s，已用 ${t} 秒"; prev_b=$b; fi
  done
}
echo "=== 路线 A：Voxel51/rico，只下需要的 3,500 张（$(date +%H:%M:%S)）"
(unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; "$PY" "$H" "$D" plan); rc=$?
if [ $rc = 0 ]; then
  N=$(wc -l < "$T/plan.tsv")
  if [ "$N" -gt 0 ]; then
    ( unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; tr '\t' ' ' < "$T/plan.tsv" | xargs -P 16 -n 2 sh -c 'curl -sfL --retry 6 --retry-delay 2 -o "$1.part" "$0" && mv "$1.part" "$1" || echo "  这张没下到：$0"' ) &
    watch_dl $! files "$N"; wait
  fi
  echo "=== 层级（元素框）：samples.json（$(date +%H:%M:%S)）"
  SJ=$(unset http_proxy https_proxy; curl -s -m 60 "$MIRROR/api/datasets/Voxel51/rico/tree/main" | "$PY" -c "import json,sys; print(next((x.get('size') or 0 for x in json.load(sys.stdin) if x.get('path')=='samples.json'), 0))" 2>/dev/null || echo 0)
  if [ "${SJ:-0}" -gt 0 ] && [ "$SJ" -lt 3000000000 ]; then (unset http_proxy https_proxy; curl -fL --retry 6 -o "$T/samples.json" "$MIRROR/datasets/Voxel51/rico/resolve/main/samples.json"); else echo "  没有可用的 samples.json（大小 ${SJ:-0}）"; fi
  "$PY" "$H" "$D" hier; rm -f "$T/samples.json" "$T/plan.tsv"
else
  echo "=== 路线 A 用不了，改走路线 B：原始 Rico 包，32 个连接同时下（$(date +%H:%M:%S)）"
  SIZE=$( (source /etc/network_turbo >/dev/null 2>&1; curl -sI -m 30 "$GS") | tr -d '\r' | awk -F': ' 'tolower($1)=="content-length"{print $2}' | tail -1)
  [ -z "$SIZE" ] && { echo "原始包问不到大小，停止（把上面几行发给我）"; exit 1; }
  P=32; per=$(( (SIZE + P - 1) / P )); echo "  原始包 $(( SIZE / 1048576 )) MB，分 $P 段"
  for k in $(seq 0 $((P-1))); do a=$((k*per)); b=$((a+per-1)); [ $b -ge $SIZE ] && b=$((SIZE-1)); echo "$k $a $b"; done > "$T/ranges.txt"
  dl_parts() { while read -r k a b; do want_sz=$((b-a+1)); have_sz=$(stat -c%s "$T/part.$k" 2>/dev/null || echo 0); [ "$have_sz" -ge "$want_sz" ] && continue
      ( source /etc/network_turbo >/dev/null 2>&1; curl -sf --retry 8 --retry-delay 3 -r $((a+have_sz))-$b "$GS" >> "$T/part.$k" ) & done < "$T/ranges.txt"; wait; }
  for round in 1 2 3; do ( dl_parts ) & watch_dl $! bytes "$SIZE"; wait; done
  tot=0; for k in $(seq 0 $((P-1))); do tot=$((tot + $(stat -c%s "$T/part.$k" 2>/dev/null || echo 0))); done
  [ "$tot" -ne "$SIZE" ] && { echo "原始包只下到 $tot / $SIZE 字节，再运行本脚本会接着下"; exit 1; }
  for k in $(seq 0 $((P-1))); do cat "$T/part.$k"; done > "$T/unique_uis.tar.gz" && rm -f "$T"/part.*
  echo "=== 从原始包里取需要的屏（$(date +%H:%M:%S)）"; "$PY" "$H" "$D" untar; rm -f "$T/unique_uis.tar.gz" "$T/ranges.txt"
fi
"$PY" "$H" "$D" status
echo "=== 完成（$(date +%H:%M:%S)）。回 Windows 控制台点“继续下载并构建数据”：Rico 会直接跳过，其余几分钟内完成"
