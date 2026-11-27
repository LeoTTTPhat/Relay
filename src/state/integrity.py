"""Confidentiality, integrity, freshness and anti-replay for the shipped block.

A resumption block is not neutral data.  It carries the goal, tool results,
file paths, entity values and -- the part that makes it security-relevant --
the side-effect ledger with its idempotency keys.  Two attacks follow
directly from the mechanism this paper proposes:

  * **forge a record.**  Inject a ledger entry for an action that never
    happened and the target's dispatcher suppresses it: a *denial of the
    action*, silent by construction, because suppression is supposed to be
    silent.
  * **replay a stale block.**  Re-present an older block, whose ledger is
    missing the most recent actions, and the target re-executes them.

Both are integrity failures, not confidentiality failures, so authentication
is the primary requirement and encryption the secondary one.  ``seal``
therefore binds, in the AEAD's associated data, everything that determines
whether a block is the *right* block: task, epoch, source and destination
device identity, and a monotonic counter.  A block sealed for one epoch or
one destination does not verify anywhere else.

Key management is deliberately the boring option: a per-task symmetric key
established once between the user's devices (the trust model pervasive
deployments already use for device pairing -- the same place a phone and a
watch get their shared secret), from which the transfer key and the
idempotency-key HMAC secret are derived by HKDF with distinct labels.  We do
not claim protection against a *compromised endpoint*: a target running
attacker code holds the key and can do anything the legitimate target could.
What sealing buys is that the network, an intermediary relay, and any device
not holding the task key cannot forge, replay, downgrade or read a block.
"""
import hashlib
import hmac
import json
import os
import struct
from dataclasses import dataclass
from typing import Optional

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    _HAVE_AEAD = True
except Exception:                                   # pragma: no cover
    _HAVE_AEAD = False


class IntegrityError(Exception):
    """Verification failed: forged, replayed, misdirected or corrupt."""


# ---------------------------------------------------------------------------
# Key schedule
# ---------------------------------------------------------------------------
def _hkdf(secret: bytes, label: bytes, n: int = 32) -> bytes:
    """HKDF-SHA256 (extract+expand) with a fixed empty salt."""
    prk = hmac.new(b"\x00" * 32, secret, hashlib.sha256).digest()
    out, t, i = b"", b"", 1
    while len(out) < n:
        t = hmac.new(prk, t + label + bytes([i]), hashlib.sha256).digest()
        out += t
        i += 1
    return out[:n]


@dataclass
class TaskKeys:
    """Keys derived from one paired-device task secret."""
    transfer: bytes       # AEAD key for the block
    idem: bytes           # HMAC secret for deterministic idempotency keys
    kid: str              # key id, so a target knows which key to try

    @staticmethod
    def derive(task_secret: bytes) -> "TaskKeys":
        return TaskKeys(
            transfer=_hkdf(task_secret, b"relay/transfer/v1"),
            idem=_hkdf(task_secret, b"relay/idempotency/v1"),
            kid=hashlib.sha256(task_secret).hexdigest()[:12])

    def idem_secret(self) -> str:
        """The secret ``protocol.derive_key`` should use for this task.

        Deriving the idempotency secret from the *task* key rather than from a
        device key is what makes two racing resumptions agree on a key: they
        share the task, not the device.
        """
        return self.idem.hex()


# ---------------------------------------------------------------------------
# Sealed envelope
# ---------------------------------------------------------------------------
@dataclass
class Envelope:
    kid: str
    task_id: str
    epoch: int
    counter: int
    src: str
    dst: str
    nonce: bytes
    ct: bytes

    def to_wire(self) -> bytes:
        hdr = json.dumps({"kid": self.kid, "task": self.task_id,
                          "epoch": self.epoch, "ctr": self.counter,
                          "src": self.src, "dst": self.dst},
                         separators=(",", ":"), sort_keys=True).encode()
        return struct.pack(">I", len(hdr)) + hdr + self.nonce + self.ct

    @staticmethod
    def from_wire(b: bytes) -> "Envelope":
        (n,) = struct.unpack(">I", b[:4])
        hdr = json.loads(b[4:4 + n].decode())
        rest = b[4 + n:]
        return Envelope(kid=hdr["kid"], task_id=hdr["task"],
                        epoch=hdr["epoch"], counter=hdr["ctr"],
                        src=hdr["src"], dst=hdr["dst"],
                        nonce=rest[:12], ct=rest[12:])

    def aad(self) -> bytes:
        """Associated data: everything that must not be swappable."""
        return json.dumps({"kid": self.kid, "task": self.task_id,
                           "epoch": self.epoch, "ctr": self.counter,
                           "src": self.src, "dst": self.dst},
                          separators=(",", ":"), sort_keys=True).encode()


def seal(blob: bytes, keys: TaskKeys, task_id: str, epoch: int,
         counter: int, src: str, dst: str,
         nonce: Optional[bytes] = None) -> Envelope:
    """Encrypt-and-authenticate a serialised block for one (epoch, dst)."""
    if not _HAVE_AEAD:
        raise RuntimeError("AEAD backend unavailable")
    env = Envelope(kid=keys.kid, task_id=task_id, epoch=epoch, counter=counter,
                   src=src, dst=dst, nonce=nonce or os.urandom(12), ct=b"")
    env.ct = AESGCM(keys.transfer).encrypt(env.nonce, blob, env.aad())
    return env


class ReplayWindow:
    """Per-task freshness state held by a receiving device.

    A block is fresh iff its epoch is not older than the highest epoch seen
    and its counter is strictly greater than the highest counter seen for
    that epoch.  This is the standard anti-replay window, and it is what
    stops an attacker (or a flaky retransmit path) from re-presenting a block
    whose ledger predates actions that have since been executed.
    """

    def __init__(self):
        self.max_epoch: int = -1
        self.max_counter: dict = {}

    def check(self, epoch: int, counter: int) -> None:
        """Freshness test only; never mutates the window."""
        if epoch < self.max_epoch:
            raise IntegrityError(
                f"stale epoch {epoch} < {self.max_epoch}: replayed block")
        if counter <= self.max_counter.get(epoch, -1):
            raise IntegrityError(
                f"replayed counter {counter} for epoch {epoch}")

    def update(self, epoch: int, counter: int) -> None:
        self.max_epoch = max(self.max_epoch, epoch)
        self.max_counter[epoch] = counter

    def check_and_update(self, epoch: int, counter: int) -> None:
        self.check(epoch, counter)
        self.update(epoch, counter)


def unseal(wire: bytes, keys: TaskKeys, task_id: str, me: str,
           window: Optional[ReplayWindow] = None) -> bytes:
    """Verify and decrypt.  Raises ``IntegrityError`` on every rejection.

    Failing closed is the whole point: a block that does not verify is not
    "probably fine but unsigned", it is a block whose ledger cannot be
    trusted, and a ledger that cannot be trusted is worse than no ledger --
    it can suppress actions that never happened.
    """
    if not _HAVE_AEAD:
        raise RuntimeError("AEAD backend unavailable")
    try:
        env = Envelope.from_wire(wire)
    except Exception as exc:
        raise IntegrityError(f"malformed envelope: {exc}")
    if env.kid != keys.kid:
        raise IntegrityError("unknown key id")
    if env.task_id != task_id:
        raise IntegrityError("block sealed for a different task")
    if env.dst != me:
        raise IntegrityError(f"block sealed for {env.dst}, not {me}")
    if window is not None:
        window.check(env.epoch, env.counter)
    try:
        blob = AESGCM(keys.transfer).decrypt(env.nonce, env.ct, env.aad())
    except Exception:
        raise IntegrityError("authentication failed: forged or corrupt block")
    # Advance the anti-replay window only for an *authenticated* envelope.
    # (Revision fix: the window used to advance before decryption, so an
    # unauthenticated envelope carrying a huge epoch -- the header is in
    # cleartext -- would make every later legitimate block look stale.)
    if window is not None:
        window.update(env.epoch, env.counter)
    return blob
