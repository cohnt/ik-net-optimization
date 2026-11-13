"""Cluster code integrity (cluster/README.md, "Code integrity"): stage_code.sh, verify_staged_tree.sh,
swap_staged_tree.sh, submit_job.sh and the submit scripts, run against LOCAL directories only.

    python tests/test_staging_integrity.py          # fast tier, about 25 s, no slow group

Hermetic: a scratch git repo (the real cluster/*.sh copied in, a few tracked files, a gitignored
checkpoint), a fake HOME standing in for the cluster account, stub `ssh` / `rsync` / `LLsub` / `sbatch`
first on PATH. Guards: (1) `command -v` of those four resolves to the stubs, else FAIL; (2)
SSH_AUTH_SOCK / SSH_AGENT_PID unset and HOME fake; (3) every stub logs its calls and the expected ones
are asserted (a refused stage must make NO rsync call); (4) no script under test names ssh / rsync / scp
by absolute path. Every expected failure asserts its exit code AND its message.
"""
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import _tiers  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILURES = []
REALRSYNC = shutil.which("rsync")
STOP = "STOP: staged code differs from its recorded commit -- tell Thomas"
CANNOT = "CANNOT VERIFY"


def check(cond, msg, detail=""):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILURES.append(msg)
        if detail:
            print("     " + detail[-1500:].replace("\n", "\n     "))


STUBS = {
    "ssh": r'''#!/bin/bash
echo "ssh $*" >> $S/calls.log
[ -e $S/ssh_down ] && exit 255
while [[ "$1" == -* ]]; do case "$1" in -F|-o) shift 2;; *) shift;; esac; done
shift
exec env -i HOME=$H USER=tcohn PATH="$PATH" TMPDIR="$TMPDIR" bash -c "$*"   # like real ssh: the laptop environment does not travel
''',
    "rsync": r'''#!/bin/bash
echo "rsync $*" >> $S/calls.log
args=(); skip=0
for a in "$@"; do
  if [ $skip = 1 ]; then skip=0; continue; fi
  if [ "$a" = -e ]; then skip=1; continue; fi
  if [[ "$a" =~ ^[^/\ ]+@[^:]+:(.*)$ ]]; then a=$H/${BASH_REMATCH[1]}; fi
  args+=("$a")
done
exec $REALRSYNC "${args[@]}"
''',
    "LLsub": '#!/bin/bash\necho "LLsub $* | EXPECT_MANIFEST_SHA=${EXPECT_MANIFEST_SHA:-UNSET}" >> $S/calls.log\necho "Submitted batch job 1"\n',
    "sbatch": '#!/bin/bash\necho "sbatch $* | EXPECT_MANIFEST_SHA=${EXPECT_MANIFEST_SHA:-UNSET}" >> $S/calls.log\necho "Submitted batch job 1"\n',
    "squeue": "#!/bin/bash\nexit 0\n",
}


class Fx:
    """One scratch repo + fake cluster account."""

    def __init__(self):
        self.S = tempfile.mkdtemp(prefix="s4min_")
        self.H = self.S + "/home"
        self.R = self.S + "/repo"          # the laptop checkout
        self.T = self.H + "/learned-ik/repo"  # the cluster tree
        self.bin = self.S + "/bin"
        os.makedirs(self.bin)
        os.makedirs(self.H + "/learned-ik")
        os.makedirs(self.S + "/xdg")
        for n, body in STUBS.items():
            p = f"{self.bin}/{n}"
            open(p, "w").write(body.replace("$S", self.S).replace("$H", self.H).replace("$REALRSYNC", REALRSYNC))
            os.chmod(p, 0o755)
        os.makedirs(self.R + "/cluster")
        for f in glob.glob(REPO + "/cluster/*.sh") + [REPO + "/cluster/chart_runs.txt"]:
            shutil.copy(f, self.R + "/cluster/")
        os.makedirs(self.R + "/src")
        os.makedirs(self.R + "/models/r")
        self.write("src/a.py", "print('a')\n")
        self.write("README", "v1\n")
        for f in ("src/generic_program.py", "src/svgd/solver.py", "cluster/gen_manifest.py", "cpp/build.sh"):   # swap_staged_tree.sh cmp's these
            self.write(f, "x = 1\n")
        self.write("cluster/manifest_t.txt", "# t\nitem1|-|src/a.py|--x\n")
        self.write(".gitignore", "models/r/*.pkl\nlogs/\n*.npz\n")
        open(self.R + "/models/r/a.pkl", "wb").write(b"c" * 200)   # gitignored payload
        self.git("init", "-q")
        self.commit("A")

    def env(self, **kw):
        e = dict(os.environ, PATH=self.bin + ":" + os.environ["PATH"], HOME=self.H, XDG_RUNTIME_DIR=self.S + "/xdg",
                 SC_ROOT="learned-ik", TMPDIR=self.S)
        for k in ("SSH_AUTH_SOCK", "SSH_AGENT_PID", "EXPECT_MANIFEST_SHA", "STAGE_SUBDIR", "LIK_LARGE_BYTES"):
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
        p = subprocess.run(["bash", "-c", cmd], cwd=cwd or self.R, env=self.env(**kw), capture_output=True,
                           text=True, timeout=120)
        return p.returncode, p.stdout + p.stderr

    def script(self, name, *args, **kw):
        return self.run("bash cluster/%s %s" % (name, " ".join(args)), **kw)

    def stage(self, **kw):
        return self.script("stage_code.sh", **kw)

    def calls(self):
        p = self.S + "/calls.log"
        return open(p).read().splitlines() if os.path.exists(p) else []

    def clear_calls(self):
        open(self.S + "/calls.log", "w").close()

    def expect_sha(self, sub="repo"):
        return subprocess.run(f"sha256sum < {self.H}/learned-ik/{sub}/.staged-manifest.sha256 | cut -d' ' -f1",
                              shell=True, capture_output=True, text=True).stdout.strip()

    def job(self, body, tree=None, **kw):
        """Run body inside bash with the verifier sourced from the cluster tree, EXPECT_MANIFEST_SHA set."""
        tree = tree or self.T
        kw.setdefault("EXPECT_MANIFEST_SHA", self.expect_sha())
        return self.run(f'source {tree}/cluster/verify_staged_tree.sh; {body}', cwd=self.S, **kw)

    def close(self):
        shutil.rmtree(self.S, ignore_errors=True)


def expect(name, res, rc, *msgs, absent=()):
    got, out = res
    ok = got == rc and all(m in out for m in msgs) and not any(a in out for a in absent)
    check(ok, name, f"rc={got} (want {rc}); wanted {msgs}; unwanted {absent}\n{out}")


def guards(fx):
    e = fx.env()
    bad = [t for t in ("ssh", "rsync", "LLsub", "sbatch") if shutil.which(t, path=e["PATH"]) != f"{fx.bin}/{t}"]
    check(not bad, "guard 1: ssh, rsync, LLsub, sbatch resolve to the stubs", str(bad))
    check("SSH_AUTH_SOCK" not in e and "SSH_AGENT_PID" not in e and e["HOME"] == fx.H,
          "guard 2: no ssh agent, fake HOME")
    absolute = subprocess.run(
        r"grep -nE '(^|[ ;|(`\"=])/[A-Za-z0-9_./-]*/(ssh|rsync|scp)([ ;|)`\"]|$)|(^|[ ;|(`\"])scp ' "
        f"{REPO}/cluster/stage_code.sh {REPO}/cluster/ssh_common.sh {REPO}/cluster/verify_staged_tree.sh "
        f"{REPO}/cluster/swap_staged_tree.sh {REPO}/cluster/submit_job.sh", shell=True, capture_output=True, text=True)
    check(absolute.returncode == 1, "guard 4: no script under test names ssh/rsync/scp by absolute path or uses scp",
          absolute.stdout)


def staged(fx, **kw):
    r = fx.stage(**kw)
    check(r[0] == 0, "setup: stage succeeds", r[1])
    return r


def t_stage():
    fx = Fx()
    try:
        guards(fx)
        head = fx.git("rev-parse", "HEAD")
        res = fx.stage()
        expect("clean stage passes and ends with the laptop check", res, 0, "matches the laptop manifest")
        t = fx.T
        check(open(t + "/.staged-commit").read().strip() == head and len(head) == 40, "full commit hash pushed")
        man = open(t + "/.staged-manifest.sha256").read()
        check("src/a.py" in man and "models/r/a.pkl" in man, "manifest lists the code and the allowlisted checkpoint")
        key = fx.expect_sha()
        sd = fx.R + "/logs/staged"
        check(os.path.exists(f"{sd}/learned-ik-repo.sha256") and os.path.exists(f"{sd}/by-manifest/{key}.sha256"),
              "laptop copy kept under logs/staged/, also keyed by the manifest's sha256")
        rs = [c for c in fx.calls() if c.startswith("rsync") and "@" in c]
        check(len(rs) == 2 and "--delete" in rs[0] and "--checksum" in rs[1] and "--max-size=" in rs[1],
              "an rsync with --delete, then the --checksum content pass", "\n".join(rs))

        # refusals: nothing may be sent
        for name, mut, rc, msg in [
            ("dirty tracked file", lambda: fx.write("src/a.py", "print('b')\n"), 6, "working tree is not clean"),
            ("untracked file", lambda: fx.write("new.py", "x=1\n"), 6, "working tree is not clean"),
        ]:
            fx.git("checkout", "-q", ".")
            subprocess.run(f"cd {fx.R} && git clean -fdq", shell=True)
            mut()
            fx.clear_calls()
            res = fx.stage()
            expect(f"{name}: stage refused", res, rc, msg)
            check(fx.calls() == [], f"{name}: no ssh or rsync call was made", "\n".join(fx.calls()))
        subprocess.run(f"cd {fx.R} && git checkout -q . && git clean -fdq", shell=True)
        fx.write("scratch.npz", "x")      # gitignored, not allowlisted payload
        fx.clear_calls()
        expect("gitignored non-payload file refused", fx.stage(), 7, "neither git-tracked nor allowlisted payload: scratch.npz")
        check(not any(c.startswith("rsync") and "@" in c for c in fx.calls()), "  and nothing was rsynced")
    finally:
        fx.close()


def t_laptop():
    fx = Fx()
    try:
        staged(fx, LIK_LARGE_BYTES="100")        # models/r/a.pkl (200 B) is "large": existence-only from the laptop
        r = fx.script("verify_staged_tree.sh")
        expect("laptop check passes and prints EXPECT_MANIFEST_SHA", r, 0, "EXPECT_MANIFEST_SHA=" + fx.expect_sha())
        T = fx.T
        run = lambda: fx.script("verify_staged_tree.sh", LIK_LARGE_BYTES="100")
        orig = open(T + "/src/a.py").read()
        open(T + "/src/a.py", "a").write("# evil\n")
        expect("tampered file: STOP", run(), 1, STOP, "src/a.py", absent=(CANNOT,))
        open(T + "/src/a.py", "w").write(orig)
        open(T + "/extra.py", "w").write("x\n")
        expect("unlisted .py: STOP", run(), 1, STOP, "unlisted CODE", "extra.py")
        os.remove(T + "/extra.py")
        open(T + "/hook.pth", "w").write("x\n")
        expect("unlisted .pth: STOP", run(), 1, STOP, "hook.pth")
        os.remove(T + "/hook.pth")
        os.symlink("/etc/passwd", T + "/lnk")
        expect("symlink: STOP", run(), 1, STOP, "SYMLINK", "lnk")
        os.remove(T + "/lnk")
        open(T + "/models/r/a.pkl", "wb").write(b"d" * 200)
        expect("large file changed: existence only from the laptop (passes)", run(), 0, "matches the laptop manifest")
        os.remove(T + "/models/r/a.pkl")
        expect("large file missing: STOP", run(), 1, STOP, "large file missing")
        open(T + "/models/r/a.pkl", "wb").write(b"c" * 200)
        open(fx.S + "/ssh_down", "w").close()
        expect("ssh failure: CANNOT VERIFY, not STOP", run(), 75, CANNOT, absent=("STOP:",))
        os.remove(fx.S + "/ssh_down")
        expect("recovers once ssh is back", run(), 0, "matches the laptop manifest")
    finally:
        fx.close()


def t_job():
    fx = Fx()
    try:
        staged(fx, LIK_LARGE_BYTES="100")
        T = fx.T
        sha0 = fx.expect_sha()
        r = fx.job('lik_job_check %s; echo PYC=$PYTHONPYCACHEPREFIX' % T)
        expect("job check passes", r, 0, "staged tree verified", "PYC=" + fx.S + "/lik-pycache.")
        pyc = re.search(r"PYC=(\S+)", r[1]).group(1)
        check(not os.path.exists(pyc), "pycache dir removed at exit")
        r = fx.job('lik_job_check %s; lik_job_check %s; echo $PYTHONPYCACHEPREFIX; trap -p EXIT' % (T, T))
        pycs = glob.glob(fx.S + "/lik-pycache.*")
        check(r[0] == 0 and not pycs and r[1].count("_lik_exit") == 1,
              "called twice: one EXIT trap, both dirs removed", r[1] + str(pycs))
        r = fx.job("trap 'echo PREVTRAP' EXIT; lik_job_check %s" % T)
        expect("an existing EXIT trap still runs", r, 0, "PREVTRAP")
        # allowed without flagging
        for d in ("cpp/build", "__pycache__", "results/x", "logs", "LLSUB.5", "pkg.egg-info"):
            os.makedirs(f"{T}/{d}", exist_ok=True)
        for f in ("cpp/build/lib.so", "__pycache__/a.cpython-312.pyc", "results/x/r.json", "logs/l.log", "LLSUB.5/o.log.0",
                  "pkg.egg-info/PKG-INFO", "run.log-5", "slurm-5.out", "models/r/m__step5.pkl", "models/r/m__step5.arch.json",
                  "models/r/m__fk_surrogate.pt"):
            open(f"{T}/{f}", "w").write("x")
        os.symlink("/dev/null", T + "/results/x/link")
        expect("cluster-made files (cpp/build, results, LLSUB, egg-info, exported checkpoints) pass", fx.job("lik_job_check %s" % T), 0,
               "staged tree verified")
        # failures
        orig = open(T + "/src/a.py").read()
        open(T + "/src/a.py", "a").write("# evil\n")
        expect("tampered file: STOP", fx.job("lik_job_check %s" % T), 1, STOP, "src/a.py")
        open(T + "/src/a.py", "w").write(orig)
        open(T + "/models/r/a.pkl", "wb").write(b"d" * 200)
        expect("large file changed: the job hashes it too", fx.job("lik_job_check %s" % T), 1, STOP, "models/r/a.pkl")
        open(T + "/models/r/a.pkl", "wb").write(b"c" * 200)
        open(T + "/x.sh", "w").write("x\n")
        expect("unlisted .sh: STOP", fx.job("lik_job_check %s" % T), 1, STOP, "x.sh")
        os.remove(T + "/x.sh")
        os.symlink("a.py", T + "/src/b.py")
        expect("symlink (also named .py): STOP", fx.job("lik_job_check %s" % T), 1, STOP, "SYMLINK")
        os.remove(T + "/src/b.py")
        r = fx.job("lik_job_check %s" % T, EXPECT_MANIFEST_SHA="")
        expect("EXPECT_MANIFEST_SHA unset/empty: refused", r, 1, "EXPECT_MANIFEST_SHA is not set", STOP)
        r = fx.job("lik_job_check %s" % T, EXPECT_MANIFEST_SHA="0" * 64)
        expect("wrong EXPECT_MANIFEST_SHA: refused", r, 1, "the laptop anchored", STOP)
        # a manifest rewritten to match a tampered tree no longer matches the laptop's anchor
        open(T + "/src/a.py", "a").write("# evil\n")
        subprocess.run(f"cd {T} && grep -v ' src/a.py$' .staged-manifest.sha256 > m && sha256sum src/a.py >> m && mv m .staged-manifest.sha256", shell=True)
        expect("tree manifest rewritten to fit tampered code: refused by the anchor", fx.job("lik_job_check %s" % T, EXPECT_MANIFEST_SHA=sha0), 1, "the laptop anchored", STOP)
        # set -euo pipefail
        open(T + "/src/a.py", "w").write(orig)
        fx.stage(LIK_LARGE_BYTES="100")
        r = fx.job("set -euo pipefail; lik_job_check %s; echo SURVIVED" % T)
        expect("under set -euo pipefail: passes and continues", r, 0, "SURVIVED")
        open(T + "/src/a.py", "a").write("# evil\n")
        r = fx.job("set -euo pipefail; lik_job_check %s; echo SURVIVED" % T)
        expect("under set -euo pipefail: refuses with the message", r, 1, STOP, absent=("SURVIVED",))
    finally:
        fx.close()


def t_content_pass():
    fx = Fx()
    try:
        staged(fx)
        st = os.stat(fx.T + "/src/a.py")
        fx.write("src/a.py", "print('b')\n")          # same size ("a" -> "b")
        os.utime(fx.R + "/src/a.py", ns=(st.st_atime_ns, st.st_mtime_ns))   # same mtime
        fx.commit("B")
        dry = subprocess.run([REALRSYNC, "-an", "--out-format=%n", fx.R + "/src/", fx.T + "/src/"], capture_output=True, text=True).stdout
        check("a.py" not in dry, "precondition: rsync -a's quick check does NOT see the same-size same-mtime change", dry)
        res = fx.stage()
        expect("restage with a same-size same-mtime change completes", res, 0, "matches the laptop manifest")
        check("print('b')" in open(fx.T + "/src/a.py").read(), "the content pass shipped the changed file")
    finally:
        fx.close()


def t_submit():
    fx = Fx()
    try:
        staged(fx)
        sha = fx.expect_sha()
        cases = [("submit.sh", ["manifest_t.txt", "1"], {}), ("submit_bench.sh", ["manifest_t.txt", "1"], {}),
                 ("submit_train.sh", ["rn", "1", "01:00:00"], {"ROBOT": "panda"}),
                 ("submit_export.sh", ["rn", "panda"], {}),
                 ("submit_job.sh", ["cluster/smoke.sh", "-q", "debug-gpu"], {}),
                 ("submit_chart_train.sh", ["--smoke", "panda", "1"], {})]
        for script, args, env in cases:
            fx.clear_calls()
            r = fx.script(script, *args, **env)
            subs = [c for c in fx.calls() if c.startswith("LLsub")]
            check(r[0] == 0 and subs and all(f"EXPECT_MANIFEST_SHA={sha}" in c for c in subs),
                  f"{script}: LLsub environment carries EXPECT_MANIFEST_SHA", f"rc={r[0]}\n{r[1]}\n" + "\n".join(fx.calls()[-3:]))
        # chain_datasets: needs the variable, hands it to sbatch
        fx.clear_calls()
        r = fx.script("chain_datasets.sh", "panda", LEARNED_IK_ROOT=fx.H + "/learned-ik", EXPECT_MANIFEST_SHA=sha)
        sb = [c for c in fx.calls() if c.startswith("sbatch")]
        check(r[0] == 0 and sb and f"EXPECT_MANIFEST_SHA={sha}" in sb[0], "chain_datasets.sh: sbatch --export carries EXPECT_MANIFEST_SHA",
              r[1] + "\n".join(sb))
        r = fx.script("chain_datasets.sh", "panda", LEARNED_IK_ROOT=fx.H + "/learned-ik")
        expect("chain_datasets.sh refuses without EXPECT_MANIFEST_SHA", r, 1, "EXPECT_MANIFEST_SHA")
        # a tampered tree: no submission
        open(fx.T + "/src/a.py", "a").write("# evil\n")
        fx.clear_calls()
        r = fx.script("submit.sh", "manifest_t.txt", "1")
        expect("tampered tree: submit.sh refuses", r, 1, STOP)
        check(not any(c.startswith("LLsub") for c in fx.calls()), "  and LLsub was never called")
        open(fx.S + "/ssh_down", "w").close()
        r = fx.script("submit_job.sh", "cluster/smoke.sh")
        expect("ssh down: submit_job.sh says CANNOT VERIFY", r, 75, CANNOT, absent=("STOP:",))
        # every job script that runs repo code checks first
        want = "smoke build_cpp_job build_dataset_job cuda_graphs_contention_job cuda_graphs_probe_job datagen_rate_job " \
               "export_and_screen_job fk_surrogate_job orientation_coverage_job preflight_root run_items setup_supercloud " \
               "tests_slow_job train_chart train_chart_pair train_flow calibrate swap_staged_tree".split()
        miss = [f for f in want if "lik_job_check" not in open(f"{REPO}/cluster/{f}.sh").read()]
        check(not miss, "every job script that runs repo code calls lik_job_check", str(miss))
    finally:
        fx.close()


def t_swap():
    fx = Fx()
    try:
        staged(fx)                                            # live tree = commit A
        fx.write("src/new.py", "n=1\n")
        os.remove(fx.R + "/src/a.py")
        fx.commit("B")
        staged(fx, STAGE_SUBDIR="repo-next")
        fx.stage_next = fx.expect_sha("repo-next")
        os.makedirs(fx.T + "/cpp/build")
        open(fx.T + "/cpp/build/lib.so", "w").write("x")
        env = dict(STAGE_SUBDIR="repo-next", SUBMIT_CMD="bash")
        fx.clear_calls()
        r = fx.script("submit_job.sh", "cluster/swap_staged_tree.sh", **env)
        expect("swap via submit_job.sh (set -euo pipefail inside)", r, 0, "swap_staged_tree: OK")
        check(fx.expect_sha() == fx.stage_next and os.path.exists(fx.T + "/src/new.py") and not os.path.exists(fx.T + "/src/a.py"),
              "live tree carries the new manifest, the new file, and not the removed one")
        check(os.path.exists(fx.T + "/cpp/build/lib.so"), "cluster-made cpp/build survived the swap")
        r = fx.run(f"bash {fx.H}/learned-ik/repo-next/cluster/swap_staged_tree.sh", cwd=fx.S)
        expect("swap job without EXPECT_MANIFEST_SHA refuses", r, 1, "EXPECT_MANIFEST_SHA is not set")
        r = fx.script("verify_staged_tree.sh")
        expect("the laptop check finds the swapped live tree's reference by its manifest hash", r, 0, "matches the laptop manifest")
    finally:
        fx.close()


GROUPS = {"fast": [t_stage, t_laptop, t_job, t_content_pass, t_submit, t_swap]}


def main():
    _tiers.run(GROUPS)
    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
