"""Preserve K2's grouped RMS normalization in vLLM's Transformers backend.

vLLM 0.21 replaces any class whose name ends in RMSNorm with ordinary RMSNorm.
K2HorizonRMSNorm normalizes independent groups and must retain its original
forward implementation. Import this module in the model worker through
worker_extension_cls="reasoning_vllm_compat.K2WorkerExtension".
No installed package or checkpoint files are modified.
"""

from collections import Counter

from vllm.model_executor.models.transformers import base, utils


def install_k2_norm_preservation():
    current = base.replace_rms_norm_class
    if getattr(current, "_preserves_greekmmlu_k2_norm", False):
        return

    def replace_preserving_k2(rms_norm, hidden_size):
        if type(rms_norm).__name__ == "K2HorizonRMSNorm":
            if not hasattr(rms_norm, "n_groups") or int(rms_norm.n_groups) < 1:
                raise RuntimeError("Unexpected K2 RMSNorm implementation; refusing replacement")
            return rms_norm
        return current(rms_norm, hidden_size)

    replace_preserving_k2._preserves_greekmmlu_k2_norm = True
    base.replace_rms_norm_class = replace_preserving_k2
    utils.replace_rms_norm_class = replace_preserving_k2


# The extension module is imported by WorkerWrapperBase before model loading.
install_k2_norm_preservation()


class K2WorkerExtension:
    def greekmmlu_k2_norm_audit(self):
        """Expose a worker-side assertion through LLM.collective_rpc."""
        model = self.model_runner.model
        config = model.config
        if getattr(config, "model_type", "") != "k2_horizon":
            raise RuntimeError("K2 norm audit called on a different architecture")
        norms = [module for module in model.modules()
                 if type(module).__name__ == "K2HorizonRMSNorm"]
        expected = 2 * int(config.num_hidden_layers) + 1
        groups = int(config.layernorm_num_groups)
        if len(norms) != expected or any(int(module.n_groups) != groups for module in norms):
            raise RuntimeError(
                f"K2 grouped norm mismatch: expected {expected} layers with {groups} groups, "
                f"found {len(norms)} with {dict(Counter(int(x.n_groups) for x in norms))}"
            )
        return {"preserved_k2_norm_layers": len(norms), "groups": groups,
                "original_forward": True}
