#!/system/bin/sh
# on the phone: samples /proc/meminfo and the nntrainer_causallm process every 0.3 s until it exits; one line a sample
out=$1; : > $out
b=$(grep MemAvailable /proc/meminfo | tr -dc 0-9); echo "base MemAvailable_kB=$b" >> $out
pid=""
while true; do
  [ -z "$pid" ] && pid=$(pidof nntrainer_causallm 2>/dev/null | cut -d' ' -f1)
  ma=$(grep MemAvailable /proc/meminfo | tr -dc 0-9); mf=$(grep MemFree /proc/meminfo | tr -dc 0-9)
  if [ -n "$pid" ]; then
    s=$(cat /proc/$pid/status 2>/dev/null) || { echo "end" >> $out; break; }
    echo "$(date +%s.%N | cut -c1-14) avail=$ma free=$mf $(echo "$s" | grep -E 'VmHWM|VmRSS|RssAnon|RssFile|RssShmem' | tr -s '\t ' ' ' | tr '\n' ' ')" >> $out
  else
    echo "$(date +%s.%N | cut -c1-14) avail=$ma free=$mf (no pid)" >> $out
  fi
  sleep 0.3
done
