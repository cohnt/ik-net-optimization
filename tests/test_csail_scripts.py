"""cluster/csail/: the CSAIL staging, submission, job and collection scripts, run against LOCAL directories only.

    python tests/test_csail_scripts.py          # fast tier, well under a minute, no GPU, no network

Hermetic, like test_staging_integrity.py: a scratch git repo (the real cluster/csail/*, verify_staged_tree.sh and
run_items.sh copied in), a stub csail_ssh.sh (cs_status / cs_run / cs_rsync that act on a fake cluster directory),
stub sbatch / squeue / apptainer / nvidia-smi first on PATH, and a fake python that stands in for the benchmark.
Guards: the stubs are what the scripts resolve; every expected refusal asserts its exit code AND message; a
refused submit must make NO sbatch call. What it pins:
  * stage.sh refuses a dirty tree, stages a commit under roots/<c12>/, and the laptop check catches a changed
    file and an unlisted code file (STOP, no sbatch);
  * submit.sh: a pinned GPU kind (partition, constraint, EXPECT_GPU), preemptible array, no --export, lik_ name,
    the lab node never reachable (vision-shared-h200 always excludes tedrake-h200-1, recorded bad nodes excluded),
    HARDWARE refuses a second GPU kind for one campaign;
  * job.sh: the stale-claim trap. A claim owned by this job id, or by a job that is not RUNNING, is released; a
    RUNNING owner's claim and an owner-less claim survive; a SIGTERM (preemption) mid-item leaves the claim, and
    the requeued attempt (same job id) releases it and finishes the item;
  * collect.sh extracts plain files and refuses an archive with a symlink.
"""
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import _tiers  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILURES = []
REALRSYNC = shutil.which("rsync")
STOP = "STOP: staged code differs from its recorded commit -- tell Thomas"
DEST = "tcohn@slurm-login.csail.mit.edu"


def check(cond, msg, detail=""):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILURES.append(msg)
        if detail:
            print("     " + detail[-1800:].replace("\n", "\n     "))


STUBS = {
    "sbatch": r'''#!/bin/bash
echo "sbatch $* | LIK_ROOT=${LIK_ROOT:-} EXPECT_MANIFEST_SHA=${EXPECT_MANIFEST_SHA:-UNSET} IMAGE_SHA256=${IMAGE_SHA256:-} EXPECT_GPU=${EXPECT_GPU:-} PROCS=${PROCS:-} MANIFEST=${MANIFEST:-}" >> $S/sbatch.log
echo 4242
''',
    "squeue": r'''#!/bin/bash
# `-o %L`: time left; `-j N -o %T`: the state recorded in $S/jobstate/N (none: not in the queue)
case "$*" in
  *%L*) echo 10:00:00 ;;
  *%T*) j=; while [ $# -gt 0 ]; do [ "$1" = -j ] && j=$2; shift; done; cat $S/jobstate/$j 2>/dev/null ;;
esac
exit 0
''',
    "nvidia-smi": '#!/bin/bash\necho "NVIDIA L40S, 580.0, 00000000:01:00.0"\n',
    "apptainer": r'''#!/bin/bash
echo "apptainer $*" >> $S/apptainer.log
[ "$1" = --version ] && { echo "apptainer version 1.5.3"; exit 0; }
shift   # exec
while [[ "$1" == -* ]]; do case "$1" in --env) export "$2"; shift 2;; --bind) shift 2;; *) shift;; esac; done
shift   # the image
cmd=("$@")
[ "${cmd[0]}" = env ] && { shift_env=1; }
[ "${cmd[0]}" = /opt/lik/venv/bin/python ] && cmd[0]=$S/fake_py
[ "${cmd[0]}" = env ] && { cmd=("${cmd[@]:2}"); cmd[0]=$S/fake_py; }
export TEST_PYTHON=$S/fake_py
exec "${cmd[@]}"
''',
    "fake_py": r'''#!/bin/bash
[ "$1" = -u ] && shift
case "$1" in *gpu_check.py) exit 0 ;; -) cat >/dev/null; exit 0 ;; esac
shift
tag=; while [ $# -gt 0 ]; do [ "$1" = --tag ] && tag=$2; shift; done
echo "start $tag restart=${SLURM_RESTART_COUNT:-} job=${SLURM_JOB_ID:-}" >> $S/ran.log
[ -z "${SLOW_ITEM:-}" ] || [ "$tag" != "${SLOW_ITEM}" ] || { touch $S/slow.started; sleep 25; }
d=$LEARNED_IK_ROOT/repo/results/fake/benchmark/$tag; mkdir -p $d; echo '{"records": []}' > $d/summary.json
echo "end $tag" >> $S/ran.log
''',
    "ssh_stub": r'''
cs_status() { [ ! -e $S/down ]; }
cs_run() { cs_status || return 255; echo "run: $*" >> $S/calls.log
  env -i HOME=$H USER=tcohn PATH="$S/bin:/usr/bin:/bin" TMPDIR=$S/tmp S=$S bash -c "$1"; }
cs_rsync() { echo "rsync $*" >> $S/calls.log; local a args=(); for a in "$@"; do args+=("${a#$CS_DEST:}"); done; $REALRSYNC "${args[@]}"; }
''',
}


class Fx:
    def __init__(self):
        self.S = tempfile.mkdtemp(prefix="csail_")
        self.H, self.R, self.bin = self.S + "/home", self.S + "/repo", self.S + "/bin"
        self.CSR = self.S + "/data/scratch/tcohn/learned-ik"
        for d in (self.H, self.bin, self.S + "/tmp", self.S + "/xdg", self.S + "/jobstate", self.CSR, self.CSR + "/images"):
            os.makedirs(d)
        for n, body in STUBS.items():
            body = body.replace("$S", self.S)
            if n == "ssh_stub":
                p = f"{self.S}/csail_ssh_stub.sh"
                body = body.replace("$H", self.H).replace("$REALRSYNC", REALRSYNC).replace("$CS_DEST", DEST)
            elif n == "fake_py":
                p = f"{self.S}/fake_py"
            else:
                p = f"{self.bin}/{n}"
            open(p, "w").write(body)
            os.chmod(p, 0o755)
        os.makedirs(self.R + "/cluster")
        shutil.copy(REPO + "/cluster/verify_staged_tree.sh", self.R + "/cluster/")
        shutil.copy(REPO + "/cluster/run_items.sh", self.R + "/cluster/")
        shutil.copytree(REPO + "/cluster/csail", self.R + "/cluster/csail", ignore=shutil.ignore_patterns("__pycache__"))
        os.makedirs(self.R + "/src")
        os.makedirs(self.R + "/scripts")
        self.write("src/a.py", "print('a')\n")
        self.write("scripts/fake_bench.py", "print('never run: the fake python stands in')\n")
        self.write("cluster/m_two.txt", "# two\ni1|-|scripts/fake_bench.py|--tag i1\n")
        self.write("cluster/m_six.txt", "".join(f"i{k}|-|scripts/fake_bench.py|--targets 1 --tag i{k}\n" for k in range(1, 7)))
        self.write(".gitignore", "logs/\nresults/\n")
        self.git("init", "-q")
        self.commit("A")
        open(self.S + "/img.sif", "wb").write(b"fake image")
        self.imgsha = subprocess.run(["sha256sum", self.S + "/img.sif"], capture_output=True, text=True).stdout.split()[0]
        shutil.copy(self.S + "/img.sif", self.CSR + "/images/img.sif")
        os.makedirs(self.R + "/logs/staged", exist_ok=True)
        open(self.R + "/logs/staged/csail-image-img.sha256", "w").write(f"{self.imgsha}  img.sif\n")

    def env(self, **kw):
        e = dict(os.environ, PATH=self.bin + ":" + os.environ["PATH"], HOME=self.H, XDG_RUNTIME_DIR=self.S + "/xdg",
                 TMPDIR=self.S + "/tmp", CS_DEST=DEST, CS_ROOT=self.CSR, CSAIL_ROOT_PREFIX=self.S + "/data",
                 CSAIL_SSH_SH=self.S + "/csail_ssh_stub.sh", S=self.S, REALRSYNC=REALRSYNC)
        for k in ("SSH_AUTH_SOCK", "SSH_AGENT_PID", "EXPECT_MANIFEST_SHA", "STAGE_SUBDIR", "GPU", "EXCLUDE_NODES"):
            e.pop(k, None)
        e.update(kw)
        return e

    def write(self, rel, text):
        os.makedirs(os.path.dirname(f"{self.R}/{rel}"), exist_ok=True)
        open(f"{self.R}/{rel}", "w").write(text)

    def git(self, *a):
        return subprocess.run(["git", "-C", self.R, "-c", "user.name=t", "-c", "user.email=t@t", *a],
                              capture_output=True, text=True, check=True).stdout.strip()

    def commit(self, msg):
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", msg)
        return self.git("rev-parse", "HEAD")

    def run(self, cmd, cwd=None, **kw):
        p = subprocess.run(["bash", "-c", cmd], cwd=cwd or self.R, env=self.env(**kw), capture_output=True, text=True,
                           timeout=120)
        return p.returncode, p.stdout + p.stderr

    def script(self, name, *args, **kw):
        return self.run("bash cluster/csail/%s %s" % (name, " ".join(args)), **kw)

    def c12(self):
        return self.git("rev-parse", "HEAD")[:12]

    def root(self):
        return f"{self.CSR}/roots/{self.c12()}"

    def log(self, name):
        p = f"{self.S}/{name}"
        return open(p).read().splitlines() if os.path.exists(p) else []

    def clear(self, name):
        open(f"{self.S}/{name}", "w").close()

    def close(self):
        subprocess.run(["pkill", "-f", self.S], capture_output=True)
        shutil.rmtree(self.S, ignore_errors=True)


def expect(name, res, rc, *msgs, absent=()):
    got, out = res
    check(got == rc and all(m in out for m in msgs) and not any(a in out for a in absent), name,
          f"rc={got} (want {rc}); wanted {msgs}; unwanted {absent}\n{out}")


def staged(fx):
    r = fx.script("stage.sh")
    check(r[0] == 0 and "matches the laptop manifest" in r[1], "setup: stage.sh succeeds and verifies", r[1])
    return r


def t_stage():
    fx = Fx()
    try:
        e = fx.env()
        check(all(shutil.which(t, path=e["PATH"]) == f"{fx.bin}/{t}" for t in ("sbatch", "squeue", "apptainer", "nvidia-smi")),
              "guard: sbatch, squeue, apptainer, nvidia-smi resolve to the stubs")
        check(subprocess.run(r"grep -nE '(^|[ ;|(`\"=])/[A-Za-z0-9_./-]*/(ssh|rsync|scp)([ ;|)`\"]|$)|(^|[ ;|(`\"])(scp|ssh) ' "
                             f"{REPO}/cluster/csail/*.sh", shell=True, capture_output=True, text=True).returncode == 1,
              "guard: no csail script calls ssh/scp/rsync directly or by absolute path (only cs_run / cs_rsync)")
        check(subprocess.run(f"grep -n 'cs_open' {REPO}/cluster/csail/*.sh | grep -v '^[^:]*:[0-9]*:#'", shell=True,
                             capture_output=True, text=True).returncode == 1, "guard: no csail script calls cs_open")
        fx.write("src/dirty.py", "x\n")
        r = fx.script("stage.sh")
        expect("dirty tree is refused", r, 6, "REFUSING: the working tree is not clean")
        check(not any(c.startswith("rsync") for c in fx.log("calls.log")), "a refused stage makes NO rsync call")
        os.remove(fx.R + "/src/dirty.py")
        r = staged(fx)
        root = fx.root()
        commit = fx.git("rev-parse", "HEAD")
        check(open(root + "/repo/.staged-commit").read().strip() == commit, "the full commit hash is written into the tree")
        check(all(os.path.isdir(f"{root}/{d}") for d in ("state", "results", "logs", "manifests")), "root layout created")
        check(os.path.exists(f"{fx.R}/logs/staged/csail-{fx.c12()}.sha256"), "the laptop keeps the manifest")
        # a changed file: every submit script refuses (STOP), makes no sbatch call
        fx.clear("sbatch.log")
        open(root + "/repo/src/a.py", "a").write("# tampered\n")
        r = fx.script("submit_check.sh", "img", GPU="rtx3090")
        expect("a changed file in the cluster tree is STOP", r, 1, STOP)
        check(not fx.log("sbatch.log"), "no sbatch after STOP")
        staged(fx)  # restores (content pass)
        open(root + "/repo/evil.py", "w").write("print('x')\n")
        r = fx.script("submit_check.sh", "img", GPU="rtx3090")
        expect("an unlisted code file in the tree is STOP", r, 1, STOP, "unlisted CODE file")
        os.remove(root + "/repo/evil.py")
        open(fx.S + "/down", "w").close()
        r = fx.script("submit_check.sh", "img", GPU="rtx3090")
        expect("a down connection is 'cannot verify', not a pass", r, 75, "shared CSAIL connection is down")
    finally:
        fx.close()


def t_submit():
    fx = Fx()
    try:
        staged(fx)
        sb = lambda: fx.log("sbatch.log")  # noqa: E731
        r = fx.script("submit.sh", "cluster/m_six.txt", "img")
        expect("GPU unset is refused (a campaign pins ONE GPU type)", r, 1, "GPU", absent=("submitted",))
        r = fx.script("submit.sh", "cluster/m_six.txt", "img", GPU="locomotion-h200")
        expect("the lab node's partition is not a GPU kind", r, 2, "unknown GPU kind")
        check(not sb(), "no sbatch for either refusal")
        r = fx.script("submit.sh", "cluster/m_six.txt", "img", "3", GPU="l40s")
        expect("l40s submit succeeds", r, 0, "JOB=4242", "PLAN:")
        line = sb()[0]
        for tok in ("--array=0-2%3", "--requeue", "--open-mode=append", "-A locomotion", "-p vision-shared-l40s",
                    "-q shared-if-available", "--constraint=nvidia_l40s", "--gpus=1", "--time=1-00:00:00", "--job-name=lik_m_six_",
                    f"--chdir={fx.root()}", f"--output={fx.root()}/logs/%x-%A_%a.out", "PROCS=1", "EXPECT_GPU=L40S",
                    "MANIFEST=cluster/m_six.txt", f"IMAGE_SHA256={fx.imgsha}", "/cluster/csail/job.sh"):
            check(tok in line, f"sbatch carries {tok}", line)
        check("--export" not in line and "tedrake" not in line and "locomotion-h200" not in line, "no --export, no lab node")
        check("EXPECT_MANIFEST_SHA=UNSET" not in line, "the manifest hash is anchored into the job")
        check(open(fx.root() + "/state/m_six/HARDWARE").read().strip() == "l40s", "HARDWARE recorded")
        r = fx.script("submit.sh", "cluster/m_six.txt", "img", GPU="a100")
        expect("a second GPU kind for the same campaign is refused", r, 5, "never mixes GPU types")
        check(len(sb()) == 1, "and makes no sbatch call")
        # h200: the lab node is excluded; recorded bad nodes and EXCLUDE_NODES join the one --exclude
        open(fx.CSR + "/bad_nodes.log", "w").write("2026-11-26 node-bad job 9: nvidia-smi failed\n")
        r = fx.script("submit.sh", "cluster/m_two.txt", "img", "1", GPU="h200", EXCLUDE_NODES="nodeX")
        expect("h200 submit succeeds", r, 0, "JOB=4242")
        line = sb()[1]
        check("-p vision-shared-h200" in line and "--exclude=tedrake-h200-1,node-bad,nodeX" in line and line.count("--exclude") == 1,
              "vision-shared-h200 excludes the lab node, a recorded bad node and the caller's list", line)
        r = fx.script("submit.sh", "cluster/m_six.txt", "img", "1", GPU="csail-h200")
        expect("csail-h200 uses -A csail -p csail-shared-h200 (HARDWARE blocks it for m_six)", r, 5)
        fx.write("cluster/m_c.txt", "i1|-|scripts/fake_bench.py|--tag i1\n")
        fx.commit("manifest c")
        staged(fx)
        r = fx.script("submit.sh", "cluster/m_c.txt", "img", "1", GPU="csail-h200")
        check(r[0] == 0 and "-A csail -p csail-shared-h200" in sb()[-1] and "tedrake" not in sb()[-1], "csail-h200 plan", r[1] + sb()[-1])
        # the helper refuses lab-node words outright
        r = fx.run('. cluster/csail/csail_common.sh && csail_refuse_lab_node "-p locomotion-h200 -q locomotion-low"; echo rc=$?')
        check("rc=1" in r[1], "csail_refuse_lab_node refuses locomotion-h200", r[1])
        r = fx.run('. cluster/csail/csail_common.sh && csail_refuse_lab_node "-p vision-shared-h200"; echo rc=$?')
        check("rc=1" in r[1], "csail_refuse_lab_node refuses vision-shared-h200 without the exclusion", r[1])
        r = fx.script("submit.sh", "cluster/nonexistent.txt", "img", GPU="l40s")
        expect("an untracked manifest is refused", r, 2, "not tracked")
        fx.write("cluster/untracked.txt", "i|-|x|--tag i\n")
        r = fx.script("submit.sh", "cluster/untracked.txt", "img", GPU="l40s")
        expect("an uncommitted manifest is refused (dirty tree first)", r, 2)
    finally:
        fx.close()


def job_env(fx, sid, restart, **kw):
    return fx.env(LIK_ROOT=fx.root(), LIK_CS_ROOT=fx.CSR, IMAGE=fx.CSR + "/images/img.sif", IMAGE_SHA256=fx.imgsha,
                  EXPECT_GPU="L40S", MANIFEST="cluster/m_six.txt", ITEM_TIMEOUT="60",
                  SLURM_JOB_ID=str(sid), SLURM_RESTART_COUNT=str(restart), SLURM_JOB_PARTITION="vision-shared-l40s",
                  SLURM_JOB_NODELIST="nodeA", CUDA_VISIBLE_DEVICES="GPU-abc", LIK_TMP_BASE=fx.S + "/tmp", **kw)


def seed_claim(fx, item, owner_job):
    d = f"{fx.root()}/state/m_six/{item}.claim"
    os.makedirs(d)
    open(d + "/owner", "w").write(f"host_p0 2026-11-26T00:00:00+00:00 {owner_job}\n".replace(" \n", "\n"))


def t_job():
    fx = Fx()
    try:
        r = staged(fx)
        sha = [ln for ln in r[1].splitlines() if ln.startswith("EXPECT_MANIFEST_SHA=")][0].split("=")[1]
        st = f"{fx.root()}/state/m_six"
        os.makedirs(st)
        open(fx.S + "/jobstate/777", "w").write("PENDING\n")     # a preempted job waiting in the queue
        open(fx.S + "/jobstate/888", "w").write("RUNNING\n")     # a live sibling
        seed_claim(fx, "i1", 777)
        seed_claim(fx, "i2", 888)
        seed_claim(fx, "i3", 500)                                 # this job's own, from the killed attempt
        seed_claim(fx, "i4", "")                                  # a hand-made claim, no job id
        open(st + "/i6.done", "w").close()
        os.makedirs(st + "/i6.claim")
        env = job_env(fx, 500, 1, EXPECT_MANIFEST_SHA=sha)
        p = subprocess.run(["bash", f"{fx.root()}/repo/cluster/csail/job.sh"], env=fx.env(**env), capture_output=True, text=True, timeout=120)
        out = p.stdout + p.stderr
        check(p.returncode == 0, "job.sh exits 0", out)
        ran = [ln.split()[1] for ln in fx.log("ran.log") if ln.startswith("start")]
        check(sorted(ran) == ["i1", "i3", "i5"], "ran exactly i1 (dead PENDING owner), i3 (own stale claim), i5 (free)", str(ran) + out[-1500:])
        check(all(os.path.exists(f"{st}/{i}.done") for i in ("i1", "i3", "i5")), "those are marked done")
        check(os.path.isdir(f"{st}/i2.claim") and not os.path.exists(f"{st}/i2.done"), "a RUNNING owner's claim is left alone")
        check(os.path.isdir(f"{st}/i4.claim") and "i4" not in ran, "an owner-less claim is never touched")
        check("released 2 stale claim(s)" in out, "released exactly the two stale claims", out[-1500:])
        check(os.path.isdir(f"{fx.root()}/results/fake/benchmark/i1"), "finished items are published under results/")
        mr = open(f"{fx.root()}/logs/500/machine.a1.txt").read()
        check("gpu: NVIDIA L40S" in mr and "cpu:" in mr and "host " in mr, "the machine record names node, GPU and CPU model", mr)
        check(os.path.exists(f"{fx.root()}/logs/500/a1.worker_0.log"), "the worker log is copied next to the results")
        # a GPU of the wrong kind fails closed
        r = subprocess.run(["bash", f"{fx.root()}/repo/cluster/csail/job.sh"], env=fx.env(**dict(env, EXPECT_MANIFEST_SHA="0" * 64)),
                           capture_output=True, text=True)
        check(r.returncode == 1 and STOP in r.stdout + r.stderr, "a job whose manifest hash is not the anchored one is STOP")
        r = subprocess.run(["bash", f"{fx.root()}/repo/cluster/csail/job.sh"], env=fx.env(**dict(env, IMAGE_SHA256="1" * 64)),
                           capture_output=True, text=True)
        check(r.returncode == 1 and "the image" in r.stdout + r.stderr and "STOP" in r.stdout + r.stderr, "an image that hashes otherwise is STOP")
    finally:
        fx.close()


def t_preempt():
    """SIGTERM mid-item (a preemption), then the same job id restarts: the stale claim is released, the item runs."""
    fx = Fx()
    try:
        r = staged(fx)
        sha = [ln for ln in r[1].splitlines() if ln.startswith("EXPECT_MANIFEST_SHA=")][0].split("=")[1]
        env = fx.env(**job_env(fx, 600, 0, EXPECT_MANIFEST_SHA=sha, SLOW_ITEM="i2"))
        p = subprocess.Popen(["bash", f"{fx.root()}/repo/cluster/csail/job.sh"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for _ in range(300):
            if os.path.exists(fx.S + "/slow.started"):
                break
            time.sleep(0.1)
        check(os.path.exists(fx.S + "/slow.started"), "the slow item i2 is running")
        p.send_signal(signal.SIGTERM)
        out = p.communicate(timeout=60)[0]
        check(p.returncode == 143 and "SIGTERM" in out, "SIGTERM: job exits 143 after saving logs", f"rc={p.returncode}\n{out[-800:]}")
        st = f"{fx.root()}/state/m_six"
        check(os.path.isdir(st + "/i2.claim") and not os.path.exists(st + "/i2.done"), "the killed attempt left i2 claimed and unfinished (the trap)")
        check(os.path.exists(f"{fx.root()}/logs/600/a0.worker_0.log"), "attempt 0's worker log was saved")
        subprocess.run(["pkill", "-f", fx.S + "/fake_py"], capture_output=True)
        subprocess.run(["pkill", "-f", "sleep 25"], capture_output=True)
        fx.clear("ran.log")
        env2 = fx.env(**job_env(fx, 600, 1, EXPECT_MANIFEST_SHA=sha))      # requeued: same id, RESTART_COUNT 1
        p2 = subprocess.run(["bash", f"{fx.root()}/repo/cluster/csail/job.sh"], env=env2, capture_output=True, text=True, timeout=120)
        o2 = p2.stdout + p2.stderr
        ran = [ln.split()[1] for ln in fx.log("ran.log") if ln.startswith("start")]
        check(p2.returncode == 0 and "released stale claim i2" in o2 and "i2" in ran, "the requeued attempt releases i2 and re-runs it", o2[-1500:])
        check(all(os.path.exists(f"{st}/i{k}.done") for k in range(1, 7)), "every item ends done (nothing orphaned)")
    finally:
        fx.close()


def t_collect():
    fx = Fx()
    try:
        staged(fx)
        root = fx.root()
        os.makedirs(f"{root}/results/fake/benchmark/t1")
        open(f"{root}/results/fake/benchmark/t1/summary.json", "w").write('{"records": []}\n')
        os.makedirs(f"{root}/logs/9")
        open(f"{root}/logs/9/machine.a0.txt", "w").write("host n\n")
        r = fx.script("collect.sh", COLLECT_PYTHON="/bin/true")
        expect("collect extracts plain files", r, 0, "extracted to")
        stg = [d for d in os.listdir(fx.R + "/results/_cluster_staging")]
        got = os.path.exists(f"{fx.R}/results/_cluster_staging/{stg[0]}/results/fake/benchmark/t1/summary.json")
        check(got, "the run directory arrived in staging", str(stg))
        check(os.path.exists(f"{fx.R}/results/_csail/{fx.c12()}") and any(
            "machine.a0.txt" in fs for _, _, fs in os.walk(f"{fx.R}/results/_csail")), "logs are kept under results/_csail/")
        os.symlink("/etc/passwd", f"{root}/results/fake/benchmark/t1/evil")
        shutil.rmtree(fx.R + "/results/_cluster_staging")
        r = fx.script("collect.sh", COLLECT_PYTHON="/bin/true")
        expect("an archive holding a symlink is refused, nothing extracted", r, 4, "STOP", "not plain files")
        left = [f for _, _, fs in os.walk(fx.R + "/results/_cluster_staging") for f in fs]
        check("evil" not in left and "summary.json" not in left, "nothing was extracted", str(left))
    finally:
        fx.close()


BUILD_APPTAINER = r"""#!/bin/bash
echo "apptainer $*" >> $S/apptainer.log
case "$1" in
  --version) echo "apptainer version 1.5.5" ;;
  build) n=$#; out=${@:$n-1:1}; echo fake-image > "$out"; echo "INFO: built" ;;
  exec) shift 2
        case "$*" in
          "cat /opt/lik/freeze.txt") echo "torch==2.14.0+cu126" ;;
          "cat /opt/lik/drake/share/doc/drake/VERSION.TXT") echo "1.58.0 09bc913cf1af0cdde31073d5f1f7f66ca35ee0d8" ;;
          *) echo "unexpected exec: $*" >&2; exit 9 ;;
        esac ;;
esac
"""


def t_dry_runs():
    """Dry run of the two other job payloads: bash -n on every script, then a stub run with a fake SLURM environment
    (real paths under a scratch dir, cwd = the --chdir target, apptainer/python/nvidia-smi/sbatch stubs). job.sh's own
    dry run is t_job. Each must reach its end without error (orchestrator rule, 2026-11-25: before EVERY submission)."""
    fx = Fx()
    try:
        for f in sorted(os.listdir(f"{REPO}/cluster/csail")):
            if f.endswith(".sh"):
                r = subprocess.run(["bash", "-n", f"{REPO}/cluster/csail/{f}"], capture_output=True, text=True)
                check(r.returncode == 0, f"bash -n {f}", r.stderr)
        r = staged(fx)
        sha = [ln for ln in r[1].splitlines() if ln.startswith("EXPECT_MANIFEST_SHA=")][0].split("=")[1]
        root = fx.root()
        env = job_env(fx, 700, 0, EXPECT_MANIFEST_SHA=sha)
        p = subprocess.run(["bash", f"{root}/repo/cluster/csail/check_job.sh"], env=fx.env(**env), cwd=root,
                           capture_output=True, text=True, timeout=120)
        check(p.returncode == 0 and "check OK" in p.stdout, "check_job.sh dry run reaches its end", p.stdout[-1200:] + p.stderr[-600:])
        check(open(f"{root}/check.700.DONE").read().strip() == "OK", "check sentinel says OK")
        # build_image_job.sh: a fake apptainer, and THIS fixture's .def pin edited to a fake tarball's sha
        fake_tar = fx.S + "/fake.tar.gz"
        open(fake_tar, "wb").write(b"fake drake")
        tsha = subprocess.run(["sha256sum", fake_tar], capture_output=True, text=True).stdout.split()[0]
        text = open(f"{REPO}/cluster/csail/learned-ik.def").read()
        old = [ln for ln in text.splitlines() if ln.startswith("    DRAKE_SHA256=")][0].split("=")[1].strip()
        fx.write("cluster/csail/learned-ik.def", text.replace(old, tsha))
        fx.commit("fixture pin")
        r = fx.script("stage.sh")
        sha2 = [ln for ln in r[1].splitlines() if ln.startswith("EXPECT_MANIFEST_SHA=")][0].split("=")[1]
        root = fx.root()
        open(f"{fx.bin}/apptainer", "w").write(BUILD_APPTAINER.replace("$S", fx.S))
        os.makedirs(fx.CSR + "/cache", exist_ok=True)
        shutil.copy(fake_tar, f"{fx.CSR}/cache/drake-1.58.0-noble.tar.gz")
        benv = fx.env(LIK_ROOT=root, LIK_CS_ROOT=fx.CSR, EXPECT_MANIFEST_SHA=sha2, SLURM_JOB_ID="701")
        p = subprocess.run(["bash", f"{root}/repo/cluster/csail/build_image_job.sh"], env=benv, cwd=root,
                           capture_output=True, text=True, timeout=120)
        out = p.stdout + p.stderr
        done = [f for f in os.listdir(fx.CSR + "/images") if f.endswith(".DONE")]
        check(p.returncode == 0 and done and open(f"{fx.CSR}/images/{done[0]}").read().startswith("OK "),
              "build_image_job.sh dry run reaches its end", out[-1500:])
        check("APPTAINER_TMPDIR=" + fx.CSR + "/cache/apptainer-tmp-701" in out, "APPTAINER_TMPDIR/CACHEDIR are under the /data tree", out[-1500:])
        base = done[0][:-5] if done else "x"
        check(all(os.path.exists(f"{fx.CSR}/images/{base}{x}") for x in (".sif", ".sif.sha256", ".freeze.txt")),
              "sif, sha256 and freeze are written")
        bl = open(fx.S + "/apptainer.log").read()
        check("--build-arg DRAKE_TARBALL=" in bl and "--build-arg IKFLOW_SRC=" in bl, "build passes the tarball and ikflow paths", bl)
        os.remove(f"{fx.CSR}/cache/drake-1.58.0-noble.tar.gz")
        open(f"{fx.bin}/curl", "w").write("#!/bin/bash\nexit 22\n")
        os.chmod(f"{fx.bin}/curl", 0o755)
        p = subprocess.run(["bash", f"{root}/repo/cluster/csail/build_image_job.sh"], env=dict(benv, SLURM_JOB_ID="702"), cwd=root,
                           capture_output=True, text=True, timeout=120)
        check(p.returncode == 20 and "BUILD FAILED" in p.stdout + p.stderr, "an unfetchable pin fails with exit 20", p.stdout[-600:])
    finally:
        fx.close()


GROUPS = {"fast": [t_stage, t_submit, t_job, t_preempt, t_collect, t_dry_runs]}


def main():
    _tiers.run(GROUPS)
    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
