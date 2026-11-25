"""cluster/engaging/ (MIT ORCD Engaging), run against LOCAL directories only; nothing here touches a cluster.

    python tests/test_engaging_scripts.py          # fast tier, a few seconds, no slow group

Hermetic: a scratch git repo holding the real cluster/ scripts, a fake HOME standing in for the Engaging
account, a stub of the engaging-cluster skill's functions (eg_status / eg_run / eg_rsync run LOCALLY
against the fake HOME; the real skill is never sourced, so no connection can be made), and a stub `sbatch`
that logs its arguments and environment. Checked:
  1. every script parses (bash -n), and lik.def is the project's pin: Drake 1.58.0 release tarball with a
     published-.sha256 check (not the PyPI wheel, not a nightly), torch 2.14.0+cu126, and every pip pin that
     setup_supercloud.sh carries;
  2. stage.sh: refuses a dirty tree, stages an immutable per-commit tree with the sha256 manifest and
     passes the same cluster-side check as SuperCloud (a tampered file is STOP);
  3. submit.sh: pins the GPU type and the CPU model, refuses an unpinned or mixed campaign, B200, and more
     tasks than GPUs; --requeue only on mit_preemptable; the integrity hash and image hash reach the job;
  4. THE REQUEUE TRAP: run_items.sh writes the claiming job's id into the claim (env-gated: SuperCloud's
     claims and file names are unchanged), and eg_release_own_claims releases exactly the dead attempt's
     claims, never another job's or a finished item's;
  5. the job's hardware pin refuses a node whose GPU or CPU model is not the pinned one.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import _tiers  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILURES = []


def check(cond, msg, detail=""):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILURES.append(msg)
        if detail:
            print("     " + detail[-1500:].replace("\n", "\n     "))


SKILL_STUB = r'''
eg_status() { return 0; }
eg_run() { echo "eg_run $*" >> "$S/calls.log"; bash -c "$*"; }
eg_rsync() { local a args=(); for a in "$@"; do [[ "$a" == "$EG_DEST":* ]] && a="$HOME/${a#*:}"; args+=("$a"); done
             echo "rsync ${args[*]}" >> "$S/calls.log"; rsync "${args[@]}"; }
'''
SBATCH_STUB = '#!/bin/bash\necho "sbatch $* | LIK_CROOT=${LIK_CROOT:-} MANIFEST=${MANIFEST:-} EXPECT_MANIFEST_SHA=${EXPECT_MANIFEST_SHA:-UNSET} EXPECT_SIF_SHA=${EXPECT_SIF_SHA:-UNSET} LIK_EXPECT_GPU=${LIK_EXPECT_GPU:-} LIK_EXPECT_CPU=${LIK_EXPECT_CPU:-} ITEM_TIMEOUT=${ITEM_TIMEOUT:-}" >> "$S/sbatch.log"\necho 4242\n'


class Fx:
    def __init__(self):
        self.root = tempfile.mkdtemp(prefix="engtest_")
        self.R = os.path.join(self.root, "repo")          # the scratch git repo (the "workstation")
        self.H = os.path.join(self.root, "home")          # the fake Engaging account
        self.S = os.path.join(self.root, "stubs")
        for d in (self.R, self.H, self.S, os.path.join(self.root, "bin")):
            os.makedirs(d)
        shutil.copytree(os.path.join(REPO, "cluster"), os.path.join(self.R, "cluster"),
                        ignore=shutil.ignore_patterns("manifest_stage*", "__pycache__", "*.txt", "*.md"))
        for m in ("manifest_engaging_smoke.txt", "manifest_engaging_requeue.txt"):
            shutil.copy(os.path.join(REPO, "cluster", "engaging", m), os.path.join(self.R, "cluster", "engaging", m))
        os.makedirs(os.path.join(self.R, "models", "panda"))
        open(os.path.join(self.R, "models", "panda", "a.yaml"), "w").write("a: 1\n")
        open(os.path.join(self.R, ".gitignore"), "w").write("logs/\nresults/\n")
        open(os.path.join(self.S, "skill.sh"), "w").write(SKILL_STUB)
        sb = os.path.join(self.root, "bin", "sbatch")
        open(sb, "w").write(SBATCH_STUB)
        os.chmod(sb, 0o755)
        self.git("init", "-q"); self.git("add", "-A")
        self.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
        self.commit = self.git("rev-parse", "HEAD").strip()
        self.env = dict(os.environ, HOME=self.H, S=self.S, EG_SKILL=os.path.join(self.S, "skill.sh"),
                        EG_DEST="tcohn@orcd-login.mit.edu", PATH=os.path.join(self.root, "bin") + ":" + os.environ["PATH"],
                        TMPDIR=self.root)
        for k in ("SSH_AUTH_SOCK", "SSH_AGENT_PID", "EXPECT_MANIFEST_SHA"):
            self.env.pop(k, None)

    def git(self, *a):
        return subprocess.run(["git", "-C", self.R, *a], capture_output=True, text=True).stdout

    def sh(self, cmd, extra=None, cwd=None):
        env = dict(self.env, **(extra or {}))
        return subprocess.run(["bash", "-c", cmd], env=env, cwd=cwd or self.R, capture_output=True, text=True)

    def script(self, name, *args, extra=None):
        return self.sh(f"bash {self.R}/cluster/engaging/{name} {' '.join(args)}", extra)

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


def t_syntax_and_pin():
    for f in sorted(os.listdir(os.path.join(REPO, "cluster", "engaging"))):
        if f.endswith(".sh"):
            r = subprocess.run(["bash", "-n", os.path.join(REPO, "cluster", "engaging", f)], capture_output=True, text=True)
            check(r.returncode == 0, f"bash -n {f}", r.stderr)
    d = open(os.path.join(REPO, "cluster", "engaging", "lik.def")).read()
    sc = open(os.path.join(REPO, "cluster", "setup_supercloud.sh")).read()
    check(re.search(r'^\s*DRAKE_VERSION="1\.58\.0"', d, re.M) and re.search(r'DRAKE_SHA256="[0-9a-f]{64}"', d),
          "lik.def pins Drake release 1.58.0 with a sha256")
    check("drake-${DRAKE_VERSION}-noble.tar.gz" in d and '$DRAKE_URL.sha256' in d,
          "lik.def fetches the noble tarball and checks the PUBLISHED .sha256")
    check(not re.search(r"pip install[^\n]*\bdrake==", d), "lik.def does not install the PyPI wheel")
    check("index-url https://download.pytorch.org/whl/cu126 torch==2.14.0" in d, "lik.def: torch 2.14.0 from the cu126 index")
    pins = set(re.findall(r"([A-Za-z0-9_.\[\]-]+==[0-9][0-9A-Za-z.+]*)", sc.split("[5/6]")[1].split("[6/6]")[0]))
    pins = {p.replace('"', "") for p in pins if not p.startswith("torch==2.")}
    missing = sorted(p for p in pins if p not in d)
    check(not missing, "every pip pin of setup_supercloud.sh section 5 is in lik.def", str(missing))


def t_stage_and_submit():
    fx = Fx()
    try:
        calls = lambda: open(os.path.join(fx.S, "calls.log")).read() if os.path.exists(os.path.join(fx.S, "calls.log")) else ""
        open(os.path.join(fx.R, "dirty.txt"), "w").write("x")
        r = fx.script("stage.sh")
        check(r.returncode == 6 and "not clean" in r.stderr and "rsync" not in calls(), "stage.sh refuses a dirty tree and ships nothing", r.stderr)
        os.remove(os.path.join(fx.R, "dirty.txt"))
        r = fx.script("stage.sh")
        croot = os.path.join(fx.H, "lik-engaging", "c-" + fx.commit[:12])
        check(r.returncode == 0 and "STAGED" in r.stdout, "stage.sh stages a clean commit", r.stdout + r.stderr)
        check(os.path.isfile(croot + "/repo/.staged-manifest.sha256") and open(croot + "/repo/.staged-commit").read().strip() == fx.commit,
              "the tree is keyed by the commit and carries its manifest and full hash")
        check("matches the laptop manifest" in r.stdout, "the cluster-side sha256sum matched the laptop manifest")
        # ---- submit.sh
        sb = os.path.join(fx.S, "sbatch.log")
        r = fx.script("submit.sh", "bench", "cluster/engaging/manifest_engaging_smoke.txt")
        check(r.returncode == 4 and "pin the CPU" in r.stderr and not os.path.exists(sb), "bench without a CPU pin is refused (no sbatch)", r.stderr)
        r = fx.script("submit.sh", "bench", "cluster/engaging/manifest_engaging_smoke.txt", extra={"GPU": "b200", "ANY_CPU": "1"})
        check(r.returncode == 4 and "sm_100" in r.stderr, "B200 is refused (cu126 has no sm_100)", r.stderr)
        r = fx.script("submit.sh", "bench", "cluster/engaging/manifest_engaging_smoke.txt", extra={"ANY_CPU": "1", "NTASKS": "3"})
        check(r.returncode == 4 and "exceeds" in r.stderr, "more tasks than the partition's GPU limit is refused", r.stderr)
        r = fx.script("submit.sh", "bench", "cluster/engaging/manifest_engaging_smoke.txt", extra={"ANY_CPU": "1"})
        check(r.returncode == 4 and "anchored image hash" in r.stderr, "bench without an anchored image hash is refused", r.stderr)
        os.makedirs(os.path.join(fx.R, "logs", "staged"), exist_ok=True)
        open(os.path.join(fx.R, "logs", "staged", "engaging-sif.sha256"), "w").write("ab" * 32 + "\n")
        r = fx.script("submit.sh", "bench", "cluster/engaging/manifest_engaging_requeue.txt",
                      extra={"PARTITION": "mit_preemptable", "NODES": "node4100-4102,node4200", "CPU_MODEL": "EPYC 9474F"})
        log = open(sb).read() if os.path.exists(sb) else ""
        check(r.returncode == 0 and "submitted job 4242" in r.stdout, "preemptable campaign submit succeeds", r.stdout + r.stderr)
        for want, why in (("-G l40s:1", "GPU pinned (default l40s)"), ("-p mit_preemptable", "partition"), ("--requeue", "--requeue on preemptable"),
                          ("--array=0-3%4", "one task per GPU, 4 GPUs"), ("-J lik_bench_", "job named lik_*"),
                          ("--constraint=node4100|node4101|node4102|node4200", "CPU pinned by a node-name OR-constraint"),
                          ("LIK_EXPECT_CPU=EPYC 9474F", "CPU model passed for the in-job check"), ("LIK_EXPECT_GPU=L40S", "GPU name passed for the in-job check"),
                          ("EXPECT_SIF_SHA=" + "ab" * 32, "image hash anchored"), ("-t 48:00:00", "48 h walltime"), ("ITEM_TIMEOUT=28800", "item cap")):
            check(want in log, "sbatch line: " + why, log)
        mh = open(os.path.join(fx.H, "lik-engaging", "c-" + fx.commit[:12], "state", "submissions.log")).read()
        check("gpu=l40s" in mh and "EPYC 9474F" in mh and "mit_preemptable" in mh, "the pins are recorded in state/submissions.log", mh)
        m = re.search(r"EXPECT_MANIFEST_SHA=([0-9a-f]{64})", log)
        check(bool(m), "the integrity hash is handed to the job")
        os.remove(sb)
        r = fx.script("submit.sh", "plumb", extra={"ANY_CPU": "1"})
        log = open(sb).read()
        check("-p mit_normal_gpu" in log and "--requeue" not in log and "--array" not in log and "-J lik_plumb" in log,
              "plumb: one GPU on mit_normal_gpu, no requeue, no array", log)
        # ---- a tampered tree is STOP before any submit
        open(croot + "/repo/cluster/engaging/py_in_image.sh", "a").write("\necho owned\n")
        os.remove(sb)
        r = fx.script("submit.sh", "plumb", extra={"ANY_CPU": "1"})
        check(r.returncode == 1 and "STOP: staged code differs" in r.stdout and not os.path.exists(sb), "a tampered staged file is STOP and nothing is submitted", r.stdout + r.stderr)
        r = fx.script("collect.sh", "--reclaim", "'x;rm'")
        check(r.returncode == 2, "collect.sh --reclaim rejects a malformed manifest name", r.stderr)
    finally:
        fx.close()


FAKE_BENCH = "import os, sys\nsys.exit(0)\n"


def t_requeue_claims():
    fx = Fx()
    try:
        st = os.path.join(fx.root, "state"); os.makedirs(st)
        def claim(i, owner):
            os.makedirs(os.path.join(st, i + ".claim"))
            if owner is not None:
                open(os.path.join(st, i + ".claim", "owner"), "w").write(owner)
        claim("dead", "host_p0_j77 2026-11-25T00:00:00 job=77")        # this job's previous attempt
        claim("dead2", "host_p0_j77 2026-11-25T00:00:00 job=77")
        claim("other", "host_p0_j78 2026-11-25T00:00:00 job=78")       # another live task
        claim("other7", "host_p0_j770 2026-11-25T00:00:00 job=770")    # id with a shared prefix
        claim("finished", "host_p0_j77 2026-11-25T00:00:00 job=77"); open(os.path.join(st, "finished.done"), "w").close()
        claim("sc", "host_p0 2026-11-25T00:00:00")                      # a SuperCloud-style claim: never touched
        r = fx.sh(f"export LIK_CROOT={fx.root}/x; source {fx.R}/cluster/engaging/job_lib.sh && SLURM_JOB_ID=77 eg_release_own_claims {st}")
        left = sorted(f[:-6] for f in os.listdir(st) if f.endswith(".claim"))
        check(r.returncode == 0 and left == ["finished", "other", "other7", "sc"],
              "eg_release_own_claims releases exactly this job's dead claims (not another job's, not a finished item's)", r.stdout + r.stderr + str(left))
        # run_items.sh: the owner line and file names, env-gated
        root = os.path.join(fx.root, "ri"); repo, state, tmp = (os.path.join(root, d) for d in ("repo", "state", "tmp"))
        for d in (repo, state, tmp, os.path.join(root, "home", ".cache")):
            os.makedirs(d)
        open(os.path.join(repo, "fake.py"), "w").write(FAKE_BENCH)
        man = os.path.join(root, "manifest_T.txt"); open(man, "w").write("itemA|-|fake.py|x\n")
        base = dict(MANIFEST=man, PROCS="1", LEARNED_IK_ROOT=root, TEST_REPO=repo, TEST_STATE_DIR=state, TEST_PYTHON=sys.executable,
                    LEARNED_IK_RESULTS=os.path.join(root, "results"), TMPDIR=tmp)
        r = fx.sh(f"bash {REPO}/cluster/run_items.sh", dict(base, LIK_CLAIM_JOB="5", LIK_TAG_SUFFIX="_j5"))
        own = open(os.path.join(state, "manifest_T", "itemA.claim", "owner")).read()
        check(re.search(r" job=5$", own.strip()) is not None, "run_items.sh records job=<id> in the claim (LIK_CLAIM_JOB)", own + r.stderr)
        check(any(f.endswith("_p0_j5.SENTINEL") for f in os.listdir(os.path.join(state, "manifest_T"))), "per-task sentinel names (LIK_TAG_SUFFIX)")
        shutil.rmtree(state); os.makedirs(state)
        r = fx.sh(f"bash {REPO}/cluster/run_items.sh", base)
        own = open(os.path.join(state, "manifest_T", "itemA.claim", "owner")).read()
        check("job=" not in own and any(re.search(r"_p0\.SENTINEL$", f) for f in os.listdir(os.path.join(state, "manifest_T"))),
              "without the env, claims and file names are exactly SuperCloud's", own)
    finally:
        fx.close()


def t_hardware_pin():
    fx = Fx()
    try:
        bin_ = os.path.join(fx.root, "bin")
        open(os.path.join(bin_, "nvidia-smi"), "w").write("#!/bin/bash\necho 'NVIDIA H200'\n")
        open(os.path.join(bin_, "lscpu"), "w").write("#!/bin/bash\necho 'Model name:  AMD EPYC 9654 96-Core'\n")
        for f in ("nvidia-smi", "lscpu"):
            os.chmod(os.path.join(bin_, f), 0o755)
        run = lambda extra: fx.sh(f"export LIK_CROOT={fx.root}/x; source {fx.R}/cluster/engaging/job_lib.sh && eg_hardware_check", extra)
        r = run({"LIK_EXPECT_GPU": "L40S"})
        check(r.returncode == 1 and "never mixes hardware" in r.stderr, "a node with the wrong GPU is refused", r.stdout + r.stderr)
        r = run({"LIK_EXPECT_GPU": "H200", "LIK_EXPECT_CPU": "EPYC 9474F"})
        check(r.returncode == 1 and "CPU [AMD EPYC 9654" in r.stderr, "a node with the wrong CPU model is refused", r.stdout + r.stderr)
        r = run({"LIK_EXPECT_GPU": "H200", "LIK_EXPECT_CPU": "EPYC 9654"})
        check(r.returncode == 0 and "HARDWARE host=" in r.stdout and "gpu=[NVIDIA H200]" in r.stdout and "cpu=[AMD EPYC 9654" in r.stdout,
              "the right hardware passes and is recorded (host, GPU, CPU model)", r.stdout + r.stderr)
    finally:
        fx.close()


GROUPS = {"fast": [t_syntax_and_pin, t_stage_and_submit, t_requeue_claims, t_hardware_pin]}


def main():
    _tiers.run(GROUPS)
    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
