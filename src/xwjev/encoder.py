"""Clue/answer text encoders.

OnnxEncoder is the solve-time default: onnxruntime + the `tokenizers` library import in under a
second, versus 15-20 s for torch + sentence-transformers on a Windows laptop. It reproduces the
sentence-transformers pipeline of the fine-tuned model exactly: transformer -> CLS pooling ->
L2 normalisation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MAX_SEQ = 48


class OnnxEncoder:
    def __init__(self, model_dir: Path, max_seq_length: int = MAX_SEQ, threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(max_seq_length)
        self.tok.enable_padding(pad_id=self.tok.token_to_id("[PAD]") or 0, pad_token="[PAD]")
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(model_dir / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.sess.get_inputs()}

    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            encs = self.tok.encode_batch(texts[i : i + batch_size])
            ids = np.array([e.ids for e in encs], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self.sess.run(None, feed)[0]  # (B, T, d)
            cls = hidden[:, 0, :].astype(np.float32)
            cls /= np.maximum(np.linalg.norm(cls, axis=1, keepdims=True), 1e-12)
            out.append(cls)
        return np.concatenate(out, axis=0) if out else np.zeros((0, 384), dtype=np.float32)


def export_onnx(model_dir: Path, opset: int = 17) -> Path:
    """Export the fine-tuned HF transformer inside a sentence-transformers model dir to model.onnx."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_dir = Path(model_dir)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir)).eval()
    enc = tok(["Tortilla chip dip (5)", "UFO passenger (5) padded longer"], padding=True, return_tensors="pt")
    names = [n for n in ("input_ids", "attention_mask", "token_type_ids") if n in enc]

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, *args):
            return self.m(**dict(zip(names, args))).last_hidden_state

    out = model_dir / "model.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            Wrapper(model),
            tuple(enc[n] for n in names),
            str(out),
            input_names=names,
            output_names=["last_hidden_state"],
            dynamic_axes={**{n: {0: "batch", 1: "seq"} for n in names}, "last_hidden_state": {0: "batch", 1: "seq"}},
            opset_version=opset,
            dynamo=False,
        )
    # sanity check against a fresh torch forward pass (the traced instance is not reused)
    model = AutoModel.from_pretrained(str(model_dir)).eval()
    with torch.inference_mode():
        ref = model(**enc).last_hidden_state[:, 0, :].numpy()
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    got = OnnxEncoder(model_dir).encode(["Tortilla chip dip (5)", "UFO passenger (5) padded longer"])
    err = float(np.abs(ref - got).max())
    print(f"exported {out} ({out.stat().st_size / 1e6:.0f} MB), max |diff| vs torch = {err:.2e}")
    return out
