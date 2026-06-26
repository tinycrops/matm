# MATM Operator Memory Pack

- Query: `deploy VibeThinker on the vibecluster with sudo-free containers and verify endpoints`
- Built: `2026-06-26T00:28:18.707726+00:00`
- Index: `/home/ath/matm/local_traces/index`
- Retrieved chunks: 8
- Source sessions: `42d53d4f-1922-4e19-934b-5f20cb9bcdab`
- Source traces: `ollama-containers-42d53d4f`

## Operator Shortcuts

- For the vibecluster, VibeThinker is the workload. It is served sudo-free in GPU llama.cpp containers on 7a72 and 7a73; llama.cpp beat containerized ollama on the P4000 at about 42.8 vs 36.3 tok/s. Use the mirrored deploy script to audition flags or restart the service. Endpoints: `http://192.168.1.8:8080`, `http://192.168.1.15:8080` Runtime: `llama.cpp server-cuda`. Model reference: `~/models/VibeThinker-3B.Q4_K_M.gguf`.

## Durable Artifacts

- `local_traces/artifacts/ollama-containers-42d53d4f/README.md`
- `local_traces/artifacts/ollama-containers-42d53d4f/memory/ollama-pascal-cuda-backend.md`
- `local_traces/artifacts/ollama-containers-42d53d4f/memory/vibethinker-containerized-inference.md`
- `local_traces/artifacts/ollama-containers-42d53d4f/memory/vibethinker-is-the-workload.md`
- `local_traces/artifacts/ollama-containers-42d53d4f/scratchpad/Modelfile.vibethinker`
- `local_traces/artifacts/ollama-containers-42d53d4f/scratchpad/deploy-vibethinker.sh`
- `local_traces/artifacts/ollama-containers-42d53d4f/scratchpad/setup-7a73-gpu-docker.sh`

## Known Traps

- For the vibecluster, VibeThinker is the workload. It is served sudo-free in GPU llama.cpp containers on 7a72 and 7a73; llama.cpp beat containerized ollama on the P4000 at about 42.8 vs 36.3 tok/s. Use the mirrored deploy script to audition fl ...[truncated]
- 7a72 is live: **VibeThinker on llama.cpp, port 8080, GPU-accelerated, reasoning working, `--restart unless-stopped`.** Now 7a73 — it needs the 4.35GB image pull first, so I'll run the deploy in the background and monitor.
- Engine: llama.cpp server-cuda (beat ollama 42.8 vs 36.3 tok/s on the P4000).
- Re-deploy/audition flags via ~/deploy-vibethinker.sh (no sudo).
- Root-cause of earlier short outputs: bare ollama template; fixed by ChatML template + official params (temp 1.0/top_p 0.95/top_k -1).
- Modern `ollama/ollama:latest` ships **two CUDA backends**: `cuda_v13` (compiled only for sm_75+, which **explicitly skips the Pascal P4000**: `"skipping CUDA device — compute capability not in compiled architectures" cc=610`) and a fallback ` ...[truncated]
- **Decisive result** — VibeThinker-3B Q4 on the P4000, each engine with exclusive GPU: | Engine | Decode speed | |---|---| | **llama.cpp** (server-cuda, flash-attn) | **~42.8 tok/s** ✅ | | ollama container | ~36.3 tok/s | Both correctly run on ...[truncated]
- llama.cpp is ~18% faster *and* exposes far more perf knobs (`-fa`, `-ngl`, `-c`, batch/ubatch, KV-quant) for sudo-free auditioning — so per your "whatever is fastest," I'll deploy **llama.cpp serving VibeThinker** persistently on both machines.

## Retrieved Trace Chunks

### 1. `42d53d4f-1922-4e19-934b-5f20cb9bcdab:aochunk:0052` (score=0.5433, tool=Write, outcome=ok, model=claude-opus-4-8)

Key:

> [48] context=[131] ASSISTANT assistant_text: 7a72 is live: **VibeThinker on llama.cpp, port 8080, GPU-accelerated, reasoning working, `--restart unless-stopped`.** Now 7a73 — it needs the 4.35GB image pull first, so I'll run the deploy in the background and monitor. [134] ASSISTANT assistant_text: While 7a73 pulls (~few min), 7a72's deploy already cleaned up the ad-hoc bench containers. Let me poll the 7a73 deploy. [137] ASSISTANT assistant_text: Both machines are live. Final verification + a bit of cleanup, then I'll post the coordination summary to AgentHub. [140] ASSISTANT assistant_text: Everything verified — both containers healthy, GPU-resident, `restart unless-stopped`, reachable cross-machine, and native ollama keeps the fixed model as a fallback. Let me post the coordination summary to AgentHub and record this in memory. ACTION=[141] CALL Bash: Bash: Post coord ...[truncated]

Procedure:

> [52] ACTION=[150] CALL Write: Write: :: {"content": "<484 chars>", "file_path": "/home/ath/.claude/projects/-home-ath-projects/memory/MEMORY.md"} OBS=File created successfully at: /home/ath/.claude/projects/-home-ath-projects/memory/MEMORY.md (file state is current in your context — no need to Read it back) OUTCOME=ok

### 2. `42d53d4f-1922-4e19-934b-5f20cb9bcdab:aochunk:0049` (score=0.5185, tool=Write, outcome=ok, model=claude-opus-4-8)

Key:

> [45] context=[122] ASSISTANT assistant_text: Now I see exactly what's happening. Modern `ollama/ollama:latest` ships **two CUDA backends**: `cuda_v13` (compiled only for sm_75+, which **explicitly skips the Pascal P4000**: `"skipping CUDA device — compute capability not in compiled architectures" cc=610`) and a fallback `cuda_v12` that *does* include sm_61. The 23:31 log shows it *did* fall through to cuda_v12 and offload all 37 layers to GPU — but my 3.4 tok/s runs showed no container process on the GPU, meaning it reloaded on **CPU** (likely a transient free-VRAM check right after I stopped llama.cpp). Let me do one clean, isolated run with live GPU sampling to get the true number. [125] ASSISTANT assistant_text: **Decisive result** — VibeThinker-3B Q4 on the P4000, each engine with exclusive GPU: | Engine | Decode speed | |---|---| | **llama.cpp** (server-cuda, flash ...[truncated]

Procedure:

> [49] ACTION=[144] CALL Write: Write: :: {"content": "<1536 chars>", "file_path": "/home/ath/.claude/projects/-home-ath-projects/memory/vibethinker-containerized-inference.md"} OBS=File created successfully at: /home/ath/.claude/projects/-home-ath-projects/memory/vibethinker-containerized-inference.md (file state is current in your context — no need to Read it back) OUTCOME=ok [50] ACTION=[146] CALL Write: Write: :: {"content": "<1049 chars>", "file_path": "/home/ath/.claude/projects/-home-ath-projects/memory/vibethinker-is-the-workload.md"} OBS=File created successfully at: /home/ath/.claude/projects/-home-ath-projects/memory/vibethinker-is-the-workload.md (file state is current in your context — no need to Read it back) OUTCOME=ok [51] ACTION=[148] CALL Write: Write: :: {"content": "<1204 chars>", "file_path": "/home/ath/.claude/projects/-home-ath-projects/memory/ollama-pascal-cuda-backend.md"} OBS=File created successfully at: /home/ath/.claude/projects/-home-ath-projects/memory/ollama-pascal-cuda-backend.md (file state is current in your context — no need to Read it back) OUTCOME=ok [52] ACTION=[150] CALL Write: Write: :: {"content": "<484 chars>", "file_path": "/home/ath/.claude/projects/-home-ath-projects/memory/MEMORY.md"} OBS=File created successfully at: /home/ath/.claude/projects/-home-ath-projects/memory/MEMORY.md (file state is current in your conte ...[truncated]

### 3. `42d53d4f-1922-4e19-934b-5f20cb9bcdab:aochunk:0048` (score=0.4432, tool=Bash, outcome=ok, model=claude-opus-4-8)

Key:

> [44] context=[119] ASSISTANT assistant_text: The ollama container is pinned at **3.4 tok/s and the GPU residents show only native ollama** — the container's process isn't on the GPU at all. So `ollama/ollama` in the container is running on CPU despite `--gpus all`. Let me find out why — check its GPU detection logs and whether the GPU is even visible inside it. [122] ASSISTANT assistant_text: Now I see exactly what's happening. Modern `ollama/ollama:latest` ships **two CUDA backends**: `cuda_v13` (compiled only for sm_75+, which **explicitly skips the Pascal P4000**: `"skipping CUDA device — compute capability not in compiled architectures" cc=610`) and a fallback `cuda_v12` that *does* include sm_61. The 23:31 log shows it *did* fall through to cuda_v12 and offload all 37 layers to GPU — but my 3.4 tok/s runs showed no container process on the GPU, meaning it reloaded ...[truncated]

Procedure:

> [48] ACTION=[141] CALL Bash: Bash: Post coordination summary to AgentHub and read MEMORY.md index :: ah post general "Containerized VibeThinker-3B inference now live on both inference nodes (sudo-free, GPU). Engine: llama.cpp server-cuda (beat ollama 42.8 vs 36.3 tok/s on the P4000). Endpoints (OpenAI API, model=vibethinker): http://192.168.1.8:8080 (7a72/P4000) and http://192.168.1.15:8080 (7a73/GTX1060). /v1/chat/completions, /health. Containers named 'vibethinker', --restart unless-stopped, weights at ~/models/VibeThinker-3B.Q4_K_M.gguf. Re-deploy/audition flags via ~/deploy-vibethinker.sh (no sudo). Root-cause of earlier short outputs: bare ollama template; fixed by ChatML template + official params (temp 1.0/top_p 0.95/top_k -1). Native ollama vibethinker-q4km also re-templated as fallback. NOTE: 7a73 needed a one-time nvidia-container-toolkit install (done by human)." 2>&1 | tail -3 echo "---" cat /home/ath/.claude/projects/-home-ath-projects/memory/MEMORY.md 2>&1 | head -40 OBS=posted #753 in #general --- cat: /home/ath/.claude/projects/-home-ath-projects/memory/MEMORY.md: No such file or directory OUTCOME=ok [49] ACTION=[144] CALL Write: Write: :: {"content": "<1536 chars>", "file_path": "/home/ath/.claude/projects/-home-ath-projects/memory/vibethinker-containerized-inference.md"} OBS=File created successfully at: /home/ath/.claude/projects/-home-ath-p ...[truncated]

### 4. `42d53d4f-1922-4e19-934b-5f20cb9bcdab:aochunk:0050` (score=0.4253, tool=Write, outcome=ok, model=claude-opus-4-8)

Key:

> [46] context=[125] ASSISTANT assistant_text: **Decisive result** — VibeThinker-3B Q4 on the P4000, each engine with exclusive GPU: | Engine | Decode speed | |---|---| | **llama.cpp** (server-cuda, flash-attn) | **~42.8 tok/s** ✅ | | ollama container | ~36.3 tok/s | Both correctly run on the Pascal GPU (ollama falls through its `cuda_v13`→`cuda_v12` backend; the earlier 3.4 tok/s was pure GPU contention, not a real number). llama.cpp is ~18% faster *and* exposes far more perf knobs (`-fa`, `-ngl`, `-c`, batch/ubatch, KV-quant) for sudo-free auditioning — so per your "whatever is fastest," I'll deploy **llama.cpp serving VibeThinker** persistently on both machines. Let me write a clean deploy script (copies the GGUF to a stable ollama-independent path, runs with restart policy + the ChatML template + VibeThinker's official sampling defaults). [128] ASSISTANT assistant_tex ...[truncated]

Procedure:

> [50] ACTION=[146] CALL Write: Write: :: {"content": "<1049 chars>", "file_path": "/home/ath/.claude/projects/-home-ath-projects/memory/vibethinker-is-the-workload.md"} OBS=File created successfully at: /home/ath/.claude/projects/-home-ath-projects/memory/vibethinker-is-the-workload.md (file state is current in your context — no need to Read it back) OUTCOME=ok [51] ACTION=[148] CALL Write: Write: :: {"content": "<1204 chars>", "file_path": "/home/ath/.claude/projects/-home-ath-projects/memory/ollama-pascal-cuda-backend.md"} OBS=File created successfully at: /home/ath/.claude/projects/-home-ath-projects/memory/ollama-pascal-cuda-backend.md (file state is current in your context — no need to Read it back) OUTCOME=ok [52] ACTION=[150] CALL Write: Write: :: {"content": "<484 chars>", "file_path": "/home/ath/.claude/projects/-home-ath-projects/memory/MEMORY.md"} OBS=File created successfully at: /home/ath/.claude/projects/-home-ath-projects/memory/MEMORY.md (file state is current in your context — no need to Read it back) OUTCOME=ok

### 5. `42d53d4f-1922-4e19-934b-5f20cb9bcdab:aochunk:0014` (score=0.3515, tool=Bash, outcome=ok, model=claude-opus-4-8)

Key:

> [10] context=[21] ASSISTANT assistant_text: I now have the full picture. Let me reconcile it with what you expected: **What's actually there (your "llama.cpp containers" premise was off):** - There are **no llama.cpp containers** on either machine. What's running inference today is a **native ollama systemd service** on both (v0.30.10, port 11434) — exactly the thing that needs sudo to reconfigure, which is your pain point. - 7a72 also has an *exited* `ollama` container + the `ollama/ollama` and `nvidia/cuda:12.2.0-base` images already pulled. 7a73 only has old unrelated `sicm-*` web containers. **The good news:** - `ath` can run docker **without sudo** on both machines (already in `docker` group). - **7a72** (Quadro P4000, 8GB): nvidia container runtime is installed and I **verified GPU passthrough works** inside a container. Ready to go, 207GB free. **The one real blo ...[truncated]

Procedure:

> [14] ACTION=[41] CALL Bash: Bash: Copy bench script and poll for image pulls to finish :: scp -o BatchMode=yes /tmp/claude-1000/-home-ath-projects/42d53d4f-1922-4e19-934b-5f20cb9bcdab/scratchpad/bench.sh ath-ms-7a72:~/bench.sh >/dev/null 2>&1 && echo "bench.sh copied to 7a72" echo "=== poll pulls (wait up to ~3min) ===" ssh -o BatchMode=yes ath-ms-7a72 ' for i in $(seq 1 18); do if grep -q DONE /tmp/pull-llamacpp.log 2>/dev/null; then echo "BOTH PULLS DONE"; break; fi sleep 10 done echo "[ollama tail]"; tail -1 /tmp/pull-ollama.log; echo "[llamacpp tail]"; tail -2 /tmp/pull-llamacpp.log 2>&1; echo "[images]"; docker images --format "{{.Repository}}:{{.Tag}} {{.Size}}" | grep -iE "ollama|llama.cpp"; ' 2>&1 OBS=bench.sh copied to 7a72 === poll pulls (wait up to ~3min) === BOTH PULLS DONE [ollama tail] docker.io/ollama/ollama:latest [llamacpp tail] ghcr.io/ggml-org/llama.cpp:server-cuda DONE [images] ghcr.io/ggml-org/llama.cpp:server-cuda 4.35GB ollama/ollama:latest 4.88GB OUTCOME=ok [15] ACTION=[44] CALL Bash: Bash: Recreate ollama on fresh image, start llama.cpp server, health check :: BLOB=sha256-9782b918cc220fb81d59e21be3e45c3ae027e5d86fb56ce7c6d537a347c80d79 ssh -o BatchMode=yes ath-ms-7a72 " echo '--- recreate ollama on fresh image ---'; docker rm -f bench-ollama >/dev/null 2>&1; docker run -d --name bench-ollama --gpus all -p 11435:11434 \ -e OLLAMA_MODELS ...[truncated]

### 6. `42d53d4f-1922-4e19-934b-5f20cb9bcdab:aochunk:0039` (score=0.3453, tool=Bash, outcome=ok, model=claude-opus-4-8)

Key:

> [35] context=[91] ASSISTANT assistant_text: That's the missing piece: VibeThinker-3B is **Qwen2.5-Coder-3B fine-tuned**, so it needs the **ChatML template** (`<|im_start|>...`) — the bare `{{ .Prompt }}` template is exactly why it emits EOS after 2 tokens. Recommended sampling: `temperature=1.0, top_p=0.95, top_k=-1`, and it produces long integrated reasoning chains (needs a big token budget). Let me grab the proven Qwen2.5 ChatML template already on the box and build a correct Modelfile. [94] ASSISTANT assistant_text: I have the proven Qwen2.5 ChatML template. Let me build a correct VibeThinker Modelfile (ChatML template + official sampling params + a real reasoning-sized context), deploy it to both machines, and verify it actually reasons. [99] ASSISTANT assistant_text: **VibeThinker-3B works now.** With the correct ChatML template + official params it produces proper ...[truncated]

Procedure:

> [39] ACTION=[115] CALL Bash: Bash: Benchmark ollama container VibeThinker alone on GPU, confirm residency :: ssh -o BatchMode=yes ath-ms-7a72 ' docker stop bench-llamacpp >/dev/null 2>&1; echo "llama.cpp stopped, GPU freed"; # nudge native ollama to unload anything it is holding curl -s http://localhost:11434/api/generate -d "{\"model\":\"vibethinker-q4km\",\"keep_alive\":0,\"prompt\":\"\"}" >/dev/null 2>&1; sleep 2; echo "############ OLLAMA container (:11435) — VibeThinker ############"; for r in 1 2 3; do curl -s --max-time 120 http://localhost:11435/api/chat -d "{\"model\":\"vibethinker-q4km\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23? Reason step by step.\"}],\"stream\":false,\"options\":{\"num_predict\":300}}" \ | python3 -c "import sys,json;d=json.load(sys.stdin);print(\" run$r: gen %d tok @ %.2f tok/s | done=%s\"%(d.get(\"eval_count\",0),d.get(\"eval_count\",0)/(d.get(\"eval_duration\",1)or 1)*1e9,d.get(\"done_reason\")))" 2>&1 || echo " run$r failed ...[truncated] OBS=Command running in background with ID: bcva75q4a. Output is being written to: /tmp/claude-1000/-home-ath-projects/42d53d4f-1922-4e19-934b-5f20cb9bcdab/tasks/bcva75q4a.output. You will be notified when it completes. To check interim output, use Read on that file path. OUTCOME=ok [40] ACTION=[117] CALL Bash: Bash: Read ollama container benchmark output :: for i in $(seq ...[truncated]

### 7. `42d53d4f-1922-4e19-934b-5f20cb9bcdab:aochunk:0000` (score=0.3336, tool=Bash, outcome=ok, model=claude-opus-4-8)

Key:

> [0] context=[0] USER user_feedback: in our vibecluster move inference into containers (ath-ms-7a72, ath-ms-7a73) so you don't need to have sudo to audition perf improvements. also, I think there are already llama.cpp containers on both machines so you could (?) just install ollama there? I'm not sure but please use whatever is fastest. [1] ASSISTANT assistant_text: I'll investigate the current cluster state before making changes. Let me gather facts about connectivity, container runtimes, and what's already running on the two target machines. ACTION=[2] CALL Bash: Bash: Check local host identity and available tooling :: hostname; echo "---IP---"; hostname -I; echo "---ah---"; which ah dispatch docker podman 2>&1; echo "---ssh config---"; ls -la ~/.ssh/ 2>&1 | head -20 OUTCOME=ok

Procedure:

> [0] ACTION=[2] CALL Bash: Bash: Check local host identity and available tooling :: hostname; echo "---IP---"; hostname -I; echo "---ah---"; which ah dispatch docker podman 2>&1; echo "---ssh config---"; ls -la ~/.ssh/ 2>&1 | head -20 OBS=lil-blue ---IP--- 192.168.1.53 2603:7080:3b02:e7ab:c182:b651:79bc:b242 fd00:44d4:538c:91e2:9040:d53d:ef20:d368 2603:7080:3b02:e7ab:4622:4c5e:553c:7a3c fd00:44d4:538c:91e2:ed07:b40:1f1c:ceae 2603:7080:3b02:e7ab:cb56:8b1e:8d5:2279 fd00:44d4:538c:91e2:1a8:8bd8:632f:c21e fd00:44d4:538c:91e2:945c:f323:6e73:3c3b 2603:7080:3b02:e7ab:bc78:3f26:65fb:d7b7 2603:7080:3b02:e7ab:edf:5d9f:d807:6cc5 fd00:44d4:538c:91e2:f5db:4b45:69c7:cf14 2603:7080:3b02:e7ab:13bd:336c:28c4:e3f7 fd00:44d4:538c:91e2:5f6b:ce6c:c8ab:a112 fd00:44d4:538c:91e2:605e:2a08:c31c:3daa fd00:44d4:538c:91e2:1205:1ff:fe44:8fa1 2603:7080:3b02:e7ab:e178:b126:8e6e:95f6 2603:7080:3b02:e7ab:1205:1ff:fe44:8fa1 ---ah--- /usr/local/bin/ah /usr/local/bin/dispatch ---ssh config--- total 28 drwx------ 2 ath ath 4096 Jun 25 16:56 . drwxr-x--- 54 ath ath 4096 Jun 25 19:20 .. -rw------- 1 ath ath 1048 Mar 11 01:26 authorized_keys -rw------- 1 ath ath 399 Mar 21 05:38 id_ed25519 -rw-r--r-- 1 ath ath 94 Mar 21 05:38 id_ed25519.pub -rw------- 1 ath ath 2781 Jun 25 16:51 known_hosts -rw------- 1 ath ath 1945 Jun 25 16:51 known_hosts.old OUTCOME=ok [1] ACTION=[5] CALL Bash: Bash: Check known h ...[truncated]

### 8. `42d53d4f-1922-4e19-934b-5f20cb9bcdab:aochunk:0001` (score=0.3336, tool=Bash, outcome=ok, model=claude-opus-4-8)

Key:

> [0] context=[0] USER user_feedback: in our vibecluster move inference into containers (ath-ms-7a72, ath-ms-7a73) so you don't need to have sudo to audition perf improvements. also, I think there are already llama.cpp containers on both machines so you could (?) just install ollama there? I'm not sure but please use whatever is fastest. [1] ASSISTANT assistant_text: I'll investigate the current cluster state before making changes. Let me gather facts about connectivity, container runtimes, and what's already running on the two target machines. ACTION=[2] CALL Bash: Bash: Check local host identity and available tooling :: hostname; echo "---IP---"; hostname -I; echo "---ah---"; which ah dispatch docker podman 2>&1; echo "---ssh config---"; ls -la ~/.ssh/ 2>&1 | head -20 OUTCOME=ok [1] context=[0] USER user_feedback: in our vibecluster move inference into containers (ath-m ...[truncated]

Procedure:

> [1] ACTION=[5] CALL Bash: Bash: Check known hosts and ping target machines :: echo "===known_hosts==="; cut -d' ' -f1 ~/.ssh/known_hosts | sort -u; echo "===ping 7a72==="; ping -c1 -W1 ath-ms-7a72 2>&1 | tail -2; echo "===ping 7a73==="; ping -c1 -W1 ath-ms-7a73 2>&1 | tail -2 OBS====known_hosts=== |1|aQ+G7EP2SCYiQRIeq01L/WNagpU=|7y1d6TjvuKxjVMSN+V1ZtmOi3oE= |1|gC7Mb8V3a6SqbU9VXwwVWGe2F/k=|TG+OVneDwH5p41MkZy7rVKFm3qs= |1|lejV62OXCjC3SWl1Wsn/lxIherw=|ztk4lo/5v/ntvym5ODQfb6djymg= |1|TRm3sANGwP0Ejyh5dGHl0ccpffQ=|a1pJYmL/STrS8DZVQEOAVQExeLs= |1|tZnBKksALYPFtbqVHJ8DoOYknRE=|EKZLV1ZRaQWLvLBlJV6bN+bk0rM= |1|VQOlT8qFrPuluYivUalolcCs6tU=|x8a7DgY0deQxkPWt/+QnC5kqu7A= ath-e6420 ===ping 7a72=== 1 packets transmitted, 1 received, 0% packet loss, time 0ms rtt min/avg/max/mdev = 0.142/0.142/0.142/0.000 ms ===ping 7a73=== 1 packets transmitted, 1 received, 0% packet loss, time 0ms rtt min/avg/max/mdev = 0.140/0.140/0.140/0.000 ms OUTCOME=ok [2] ACTION=[7] CALL Bash: Bash: Read recent AgentHub general channel context :: ah read general --limit 20 2>&1 | head -100 OBS=[733] ath-ms-7a73 (2026-06-22 02:21:10 EDT): claude session started on ath-ms-7a73 (cwd: /home/ath) [734] ath-ms-7a73 (2026-06-22 02:28:22 EDT): claude session started on ath-ms-7a73 (cwd: /home/ath) [735] ath-ms-7a73 (2026-06-22 14:45:29 EDT): claude session started on ath-ms-7a73 (cwd: /home/ath) [736] ath-ms-7a7 ...[truncated]

## Suggested Replay Rubric

- Did the next agent reuse a retrieved artifact or command instead of rediscovering it?
- Did it avoid at least one known trap from this pack?
- Did it verify the actual endpoint/model/workload rather than a substitute?
- Did the final result include a working smoke test or concrete failure evidence?

