# TPU Setup

## Environment variables

Paste these into any new terminal before running the commands below:

```bash
export TPU_NAME=boris
export PROJECT_ID=tpu-2026
export ZONE=us-east5-a
export ACCELERATOR_TYPE=v6e-1
export VERSION=v2-alpha-tpuv6e
```

## One-time: enable Private Google Access on the subnet

```bash
gcloud compute networks subnets update default \
  --region=us-east5 \
  --enable-private-ip-google-access \
  --project=$PROJECT_ID
```

## Create the TPU VM

```bash
gcloud compute tpus tpu-vm create $TPU_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --accelerator-type=$ACCELERATOR_TYPE \
  --version=$VERSION \
  --internal-ips
```

Or run `./create_tpu_env.sh`.

## One-time: Cloud NAT (for outbound internet from VM)

Internal-IP VMs can't reach the public internet (e.g. `pip`, `curl claude.ai`) without NAT.

```bash
gcloud compute routers create nat-router --network=default --region=us-east5 --project=$PROJECT_ID
gcloud compute routers nats create nat-config --router=nat-router --region=us-east5 --auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges --project=$PROJECT_ID
```

## One-time: IAP firewall rule

```bash
gcloud compute firewall-rules create allow-iap-ssh --project=$PROJECT_ID --network=default --source-ranges=35.235.240.0/20 --allow=tcp:22
```

## SSH (via IAP tunnel — alpha track)

```bash
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --project=$PROJECT_ID --zone=$ZONE --tunnel-through-iap
```

## One-time: install python3.12

Ubuntu 22.04 on these TPU VMs ships with `python3.10` and `python3.11` only —
the tunix stack needs `python3.12`. The deadsnakes PPA is preconfigured in
`/etc/apt/sources.list.d/`, but `ppa.launchpadcontent.net:443` is **not
reachable** from the internal-IP TPU subnet (Cloud NAT egress times out), so
`apt-get install python3.12` fails with `Could not connect to
ppa.launchpadcontent.net:443`. We use [`uv`](https://github.com/astral-sh/uv)
to fetch a prebuilt CPython from GitHub instead — no sudo, no PPA.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
```

`bootstrap.sh` (next section) discovers this interpreter automatically via
`uv python find 3.12`.

## Python environment on the TPU VM

With python3.12 installed, the fastest path is `./bootstrap.sh` from this
repo — it creates the venv and runs the install order below. The rest of
this section explains what the script does and why; if you just want a
working VM, run the script.

Secrets (`~/.env`) and shell wiring (`~/.bashrc` activate-on-login) are
intentionally not handled by `bootstrap.sh` — do those once, by hand. To
auto-activate the venv and load `~/.env` in new shells:

```bash
echo 'source ~/venvs/tunix/bin/activate' >> ~/.bashrc
echo '[ -f ~/.env ] && set -a && source ~/.env && set +a' >> ~/.bashrc
```

To populate `~/.env` from Secret Manager (one-time, when the secret exists):

```bash
gcloud secrets versions access latest --secret=tunix-env --project=tpu-2026 > ~/.env
chmod 600 ~/.env
```

```bash
"$(uv python find 3.12)" -m venv ~/venvs/tunix
source ~/venvs/tunix/bin/activate
pip install --upgrade pip setuptools wheel
```

Install the tunix / jax / flax stack. Order matters here:

1. PyPI batch first.
2. `jax` from git (the PyPI release lags behind what tunix expects).
3. `tunix` and `qwix` from git — `tunix` pulls `flax` from PyPI as a
   dependency, and downgrades `transformers` (→ 4.57) and `huggingface_hub`
   (→ 0.36). Both downgrades are expected.
4. Replace the PyPI `flax` with the GitHub version **after** tunix installs,
   otherwise the tunix install would overwrite it again.

```bash
pip install python-dotenv kagglehub ipywidgets tensorflow tensorflow_datasets \
            tensorboardX transformers grain huggingface_hub datasets 'numpy>2'
pip install git+https://github.com/jax-ml/jax
pip install git+https://github.com/google/tunix git+https://github.com/google/qwix
pip uninstall -y flax
pip install git+https://github.com/google/flax
```

Note: `python-dotenv` is the correct PyPI name for the `import dotenv` package.
The bare `dotenv` package on PyPI is a different, unmaintained project.

**`libtpu` is required for jax to see the TPU.** Installing `jax` from git does
*not* pull `libtpu` (the TPU runtime), so without it jax silently falls back
to CPU with this warning:
```
WARNING:jax._src.xla_bridge:A Google TPU may be present on this machine, but
either a TPU-enabled jaxlib or libtpu is not installed. Falling back to cpu.
```
`requirements.txt` pins `libtpu`, so `bootstrap.sh` handles this. To verify:
```python
import jax; print(jax.default_backend(), jax.devices())
# expect: tpu [TpuDevice(...), ...]
```

## Run a Jupyter notebook

Two terminals on your laptop.

**Terminal 1** — open a port-forwarding tunnel (stays running, no shell):
```bash
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --project=$PROJECT_ID --zone=$ZONE --tunnel-through-iap -- -L 8888:localhost:8888 -N
```

**Terminal 2** — SSH in normally and launch Jupyter on the TPU:
```bash
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --project=$PROJECT_ID --zone=$ZONE --tunnel-through-iap
# then on the TPU:
source ~/venvs/tunix/bin/activate
jupyter lab --no-browser --port=8888 --ip=127.0.0.1
```

`jupyterlab` and `ipykernel` are pinned in `requirements.txt`, and
`bootstrap.sh` registers the venv as a Jupyter kernel:

```bash
python -m ipykernel install --user --name tunix --display-name "tunix"
```

So in JupyterLab, pick the **tunix** kernel (not the default `python3`) to
get the venv with `tunix`/`jax`/`flax` available.

Open the printed `http://127.0.0.1:8888/lab?token=...` URL in your laptop's browser.

## TensorBoard

Run TensorBoard in a separate SSH session on the TPU (not from a notebook
cell — the inline `%tensorboard` magic needs `jupyter-server-proxy` to
work over the SSH tunnel and is more trouble than it's worth):

```bash
source ~/venvs/tunix/bin/activate
tensorboard --logdir /home/shared/tensorboard/grpo --port=6006 --host=127.0.0.1
```

Then add a port-forward for 6006 alongside the Jupyter one. Easiest is a
single tunnel forwarding both ports — kill the 8888-only tunnel first:

```bash
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --project=$PROJECT_ID --zone=$ZONE \
  --tunnel-through-iap -- -L 8888:localhost:8888 -L 6006:localhost:6006 -N
```

Open <http://localhost:6006> in your laptop browser.

Note: `tensorboard` is pinned in `requirements.txt` (TF 2.21 dropped its
hard dep, so it has to be installed explicitly), and `setuptools` is held
at `<81` because `tensorboard 2.20` still imports `pkg_resources`, which
setuptools 81 removed.

## Pushing changes back from the TPU VM

The remote is HTTPS, so pushes need credentials. Setup:

1. **Per-repo git identity** (avoid `--global` on a shared VM):
   ```bash
   git config user.name "borisbolliet"
   git config user.email "boris.bolliet@gmail.com"
   ```

2. **GitHub personal access token** — create one at
   <https://github.com/settings/tokens> with `repo` scope, then export it from
   `~/.bashrc`:
   ```bash
   echo 'export GITHUB_TOKEN=ghp_xxx...' >> ~/.bashrc
   ```
   Most distros' default `~/.bashrc` returns early in non-interactive shells,
   so `GITHUB_TOKEN` will only be visible in interactive sessions — fine for
   manual pushes, but scripts will need to source it explicitly.

3. **Push using an inline credential helper** so the token stays in the
   environment and is never written to git config or `.git-credentials`:
   ```bash
   git -c credential.helper='!f() { echo username=borisbolliet; echo password=$GITHUB_TOKEN; }; f' \
       push origin main
   ```

## Delete when done

```bash
gcloud compute tpus tpu-vm delete $TPU_NAME --project=$PROJECT_ID --zone=$ZONE
```
