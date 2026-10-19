import csv
import hashlib
import json
import math
import numpy as np
import torch
import time
from torch import nn
import os
import warnings


from Net import (
    BENCHMARK_G_INIT_METHOD_CHOICES,
    G_INIT_METHOD_CHOICES,
    GradDiffNet,
    INIT_METHOD_CHOICES,
    QGradNet,
    Y_Net,
    initialization_metadata,
)
from equation import (
    FAIRNESS_SPEC,
    REFERENCE_LABEL_BY_POLICY,
    REFERENCE_MIXTURE_POLICY,
    REFERENCE_POLICY_BY_LABEL,
    REFERENCE_POLICY_LABELS,
    build_feasible_ds_allocation,
    build_unprojected_ds_allocation,
    load_open_loop_allocation_plan,
    project_rate_to_inventory_torch,
    step_transition_torch,
)
from gradient_diagnostics import (
    GradientDiagnosticsRecorder,
    git_commit,
    group_gradient_norms,
    parse_interval_subset,
)
from input_scaling import (
    apply_state_input_scaling,
    compute_physical_DH,
    input_scale_vector_for_mode,
    reparameterization_metadata,
    transform_first_layer,
    transform_g_state_dict_between_reparameterizations,
    transform_h_output_layer,
    transform_h_state_dict_between_reparameterizations,
)
from sampling_backend import TrainingStateSampler

LOSS_MODE_CE_CONSISTENCY = "ce_consistency"
LOSS_MODE_CE_NO_CONSISTENCY = "ce_no_consistency"
LOSS_MODE_MARTINGALE_QGRAD = "martingale_qgrad"
LOSS_MODE_CHOICES = (
    LOSS_MODE_CE_CONSISTENCY,
    LOSS_MODE_CE_NO_CONSISTENCY,
    LOSS_MODE_MARTINGALE_QGRAD,
)


def assemble_hg_component_gradients(
    params,
    h_count,
    base_grads,
    consistency_grads,
    *,
    h_base_multiplier=1.0,
    g_base_multiplier=1.0,
    h_consistency_coef=0.0,
    g_consistency_coef=0.0,
):
    """Assign H/G parameter grads from separately computed loss components."""
    for param_idx, param in enumerate(params):
        base_grad = base_grads[param_idx]
        consistency_grad = consistency_grads[param_idx]
        multiplier = h_base_multiplier if param_idx < h_count else g_base_multiplier
        consistency_weight = h_consistency_coef if param_idx < h_count else g_consistency_coef
        assembled = None
        if base_grad is not None and multiplier != 0.0:
            assembled = base_grad.detach() * float(multiplier)
        if consistency_grad is not None and consistency_weight != 0.0:
            weighted = consistency_grad.detach() * float(consistency_weight)
            assembled = weighted if assembled is None else assembled + weighted
        param.grad = None if assembled is None else assembled.clone()


def _state_dict_sha256(state_dict):
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def tensor_sha256(tensor):
    digest = hashlib.sha256()
    tensor_cpu = tensor.detach().cpu().contiguous()
    digest.update(str(tuple(tensor_cpu.shape)).encode("utf-8"))
    digest.update(str(tensor_cpu.dtype).encode("utf-8"))
    digest.update(tensor_cpu.numpy().tobytes())
    return digest.hexdigest()


def tensor_sequence_sha256(tensors):
    digest = hashlib.sha256()
    for idx, tensor in enumerate(tensors):
        digest.update(str(idx).encode("utf-8"))
        digest.update(tensor_sha256(tensor).encode("utf-8"))
    return digest.hexdigest()


def json_sha256(payload):
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stateless_seed_from_components(*components):
    encoded = json.dumps(components, default=str, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def module_state_sha256(module):
    return _state_dict_sha256(module.state_dict())


def module_list_state_sha256(module_list):
    digest = hashlib.sha256()
    for idx, module in enumerate(module_list):
        digest.update(str(idx).encode("utf-8"))
        digest.update(module_state_sha256(module).encode("utf-8"))
    return digest.hexdigest()


class DSPL_solver(object):
    """
    DSPL Solver Class for solving Liberia Inventory Model HJB equation using deep splitting method.

    Attributes:
        config (object): Configuration object containing network and equation parameters.
        bsde (object): LiberiaHJB equation instance implementing the Liberia inventory model.
    """
        
    def __init__(self, config, bsde):

        # Configuration objects
        self.config = config
        self.net_config = config.net_config  # Neural Network Hyperparameters
        self.eqn_config = config.eqn_config  # PDE-related system parameters
        self.bsde = bsde  # PDE method instance

        # Device setup
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU

        # PDE dimensions and time parameters
        self.dim = self.eqn_config.dim  # Dimensionality of the PDE
        self.num_time_interval = self.net_config.num_time_interval  # Number of time intervals partitioning the time horizon
        self.delta_t = self.eqn_config.total_time / self.num_time_interval  # Length of each time partition
        self.sqrt_delta_t = np.sqrt(self.delta_t)  # Square root of time partition length
        self.T = self.eqn_config.total_time  # Total time horizon
        self.print_interval = self.net_config.print_interval

        # Neural Network architecture parameters
        self.num_neurons = self.net_config.num_neurons  # Neurons in each hidden layer
        self.num_layers = self.net_config.num_layers  # Number of hidden layers
        self.activation = self.net_config.activation  # Activation function for hidden layers
        self.slope = self.net_config.slope  # Alpha for Leaky ReLU or CELU activations
        self.init_method = getattr(self.net_config, "init_method", "xavier_uniform")
        self.h_init_method = getattr(self.net_config, "h_init_method", self.init_method)
        self.g_init_method = getattr(self.net_config, "g_init_method", self.init_method)
        if self.h_init_method not in INIT_METHOD_CHOICES:
            raise ValueError(f"h_init_method must be one of {INIT_METHOD_CHOICES}, got {self.h_init_method!r}.")
        if self.g_init_method not in G_INIT_METHOD_CHOICES:
            raise ValueError(f"g_init_method must be one of {G_INIT_METHOD_CHOICES}, got {self.g_init_method!r}.")
        self.g_is_benchmark_initialized = self.g_init_method in BENCHMARK_G_INIT_METHOD_CHOICES
        self.g_constructor_init_method = "xavier_uniform" if self.g_is_benchmark_initialized else self.g_init_method
        self.g_output_bias = bool(getattr(self.net_config, "g_output_bias", False))
        if self.g_is_benchmark_initialized and not self.g_output_bias:
            raise ValueError(
                f"g_init_method={self.g_init_method!r} requires g_output_bias=true so the "
                "benchmark-implied G prior can be written as a final-layer bias."
            )
        self.g_benchmark_final_weight_scale = float(
            getattr(self.net_config, "g_benchmark_final_weight_scale", 0.0)
        )
        if self.g_benchmark_final_weight_scale != 0.0:
            raise ValueError(
                "Only g_benchmark_final_weight_scale=0.0 is supported; "
                "nonzero scales would require additional RNG isolation metadata."
            )
        self.g_prior_epsilon_a = float(getattr(self.net_config, "g_prior_epsilon_a", 1e-6))
        self.g_prior_lower_bound = float(getattr(self.net_config, "g_prior_lower_bound", 1e-8))
        self.g_prior_upper_bound = float(getattr(self.net_config, "g_prior_upper_bound", 1e3))
        self.g_prior_cap_offset = float(getattr(self.net_config, "g_prior_cap_offset", 1e-6))
        self.g_prior_fluid_plan_path = getattr(self.net_config, "g_prior_fluid_plan_path", None)
        if self.g_prior_epsilon_a <= 0.0:
            raise ValueError("g_prior_epsilon_a must be positive.")
        if not (0.0 <= self.g_prior_lower_bound < self.g_prior_upper_bound):
            raise ValueError("Require 0 <= g_prior_lower_bound < g_prior_upper_bound.")
        if not (0.0 <= self.g_prior_cap_offset < 1.0):
            raise ValueError("g_prior_cap_offset must lie in [0, 1).")
        self.valid_size = self.net_config.valid_size
        self.batch_size = self.net_config.batch_size
        self.prev_init = self.net_config.prev_init
        self.prev_init_same_params_only = bool(
            getattr(self.net_config, "prev_init_same_params_only", False)
        )
        self.prev_init_param_tol = float(
            getattr(self.net_config, "prev_init_param_tol", 1e-8)
        )

        # Learning rate parameters
        self.gamma = self.net_config.gamma  # Backward-compatible default scheduler decay.
        self.inner_gamma = float(getattr(self.net_config, "inner_gamma", self.gamma))
        self.final_gamma = float(getattr(self.net_config, "final_gamma", self.gamma))
        self.inner_learning_rate = self.net_config.inner_learning_rate  # Initial learning rate for inner intervals
        self.final_learning_rate = self.net_config.final_learning_rate  # Initial learning rate for the final interval
        self.inner_milestones = self.net_config.inner_milestones  # Milestones for adjusting learning rate (inner intervals)
        self.final_milestones = self.net_config.final_milestones  # Milestones for adjusting learning rate (final interval)

        # Gradient clipping parameters
        self.grad_clip = bool(self.net_config.grad_clip)  # Enable/disable gradient clipping
        self.grad_norm = float(self.net_config.grad_norm)  # Maximum gradient norm for clipping
        self.h_grad_norm = float(getattr(self.net_config, "h_grad_norm", self.grad_norm))
        self.g_grad_norm = float(getattr(self.net_config, "g_grad_norm", self.grad_norm))
        self.grad_clip_rule = str(getattr(self.net_config, "grad_clip_rule", "joint"))
        self.final_grad_clip_rule = str(getattr(self.net_config, "final_grad_clip_rule", self.grad_clip_rule))
        self.inner_grad_clip_rule = str(getattr(self.net_config, "inner_grad_clip_rule", self.grad_clip_rule))
        for rule_name, rule_value in {
            "grad_clip_rule": self.grad_clip_rule,
            "final_grad_clip_rule": self.final_grad_clip_rule,
            "inner_grad_clip_rule": self.inner_grad_clip_rule,
        }.items():
            if rule_value not in {"joint", "separate_hg"}:
                raise ValueError(f"{rule_name} must be one of {{'joint', 'separate_hg'}}")

        self.loss_type = getattr(self.net_config, "loss_type", None)
        self.loss_mode = self._resolve_loss_mode()
        self.requires_qgrad = self.loss_mode == LOSS_MODE_MARTINGALE_QGRAD

        # Early stopping parameters
        self.early_stopping = self.net_config.early_stopping  # Enable/disable early stopping
        self.patience_iterations = int(self.net_config.patience)  # Optimizer-iteration patience.
        self.validation_every = int(getattr(self.net_config, "validation_every", 25))
        if self.validation_every <= 0:
            raise ValueError(f"validation_every must be positive, got {self.validation_every}.")
        self.patience_checks = int(math.ceil(self.patience_iterations / self.validation_every))
        self.patience = self.patience_checks
        self.min_delta = self.net_config.min_delta  # Minimum improvement for early stopping
        self.final_min_iterations = int(getattr(self.net_config, "final_min_iterations", 0))
        self.inner_min_iterations = int(getattr(self.net_config, "inner_min_iterations", 0))
        self.train_interval_subset = parse_interval_subset(
            getattr(self.net_config, "train_interval_subset", None)
        )
        self.init_checkpoint_train_subset_mode = str(
            getattr(self.net_config, "init_checkpoint_train_subset_mode", "same_interval")
        )
        if self.init_checkpoint_train_subset_mode not in {"same_interval", "replay_original_init"}:
            raise ValueError(
                "init_checkpoint_train_subset_mode must be one of "
                "{'same_interval', 'replay_original_init'}, got "
                f"{self.init_checkpoint_train_subset_mode!r}."
            )
        if self.train_interval_subset is not None:
            invalid_intervals = [
                interval
                for interval in self.train_interval_subset
                if interval < 0 or interval >= self.num_time_interval
            ]
            if invalid_intervals:
                raise ValueError(
                    "train_interval_subset contains intervals outside "
                    f"0..{self.num_time_interval - 1}: {invalid_intervals}"
                )

        self.component_seed_isolation = bool(
            getattr(self.net_config, "component_seed_isolation", False)
        )
        train_seed_value = getattr(self.net_config, "train_seed", None)
        base_component_seed = 0 if train_seed_value is None else int(train_seed_value)
        self.h_network_seed = int(
            getattr(self.net_config, "h_network_seed", base_component_seed * 1000 + 101)
        )
        self.g_network_seed = int(
            getattr(self.net_config, "g_network_seed", base_component_seed * 1000 + 202)
        )
        self.qgrad_network_seed = int(
            getattr(self.net_config, "qgrad_network_seed", base_component_seed * 1000 + 303)
        )
        self.save_initial_checkpoints = bool(
            getattr(self.net_config, "save_initial_checkpoints", True)
        )

        # For LiberiaHJB, data is loaded in the equation class
        # Move data to device and prepare for sampling
        self.bsde.mu = self.bsde.mu.to(self.device)
        self.bsde.Gamma = self.bsde.Gamma.to(self.device)
        self.bsde.p = self.bsde.p.to(self.device)
        self.bsde.c = self.bsde.c.to(self.device)
        self.bsde.c_tilde = self.bsde.c_tilde.to(self.device)
        self.bsde.u = self.bsde.u.to(self.device)
        self.bsde.r = self.bsde.r.to(self.device)
        self.bsde.d = self.bsde.d.to(self.device)
        self.bsde.distance = self.bsde.distance.to(self.device)
        self.bsde.exposure = self.bsde.exposure.to(self.device)
        self.bsde.theta = self.bsde.theta.to(self.device)
        self.bsde.h = self.bsde.h.to(self.device)
        self.bsde.R = self.bsde.R.to(self.device)
        self.bsde.Omega_1 = self.bsde.Omega_1.to(self.device)
        self.bsde.Omega_2 = self.bsde.Omega_2.to(self.device)
        for optional_tensor_name in (
            "eta",
            "eta_schedule",
            "eta_schedule_raw",
            "eta_constant_input",
            "reference_fluid_plan_quantity",
            "reference_fluid_rate_plan",
            "_inventory_scaled_global_scale",
            "_status_quo_rate_schedule",
        ):
            self._move_optional_bsde_tensor(optional_tensor_name)
        self.gamma_sqrt = self._build_gamma_sqrt_cache()
        
        # State dimension is K+2 (Z has K+1 dims, I has 1 dim)
        self.state_dim = self.dim + 1
        self.grad_diff_dim = self.dim
        self.input_mode = str(getattr(self.net_config, "input_mode", "raw"))
        self.q0_input_scale = float(getattr(self.net_config, "q0_input_scale", self.bsde.I0))
        self.h_output_scale = float(getattr(self.net_config, "h_output_scale", 1.0))
        if not np.isfinite(self.h_output_scale) or self.h_output_scale <= 0.0:
            raise ValueError(f"h_output_scale must be finite and positive, got {self.h_output_scale}.")
        self.state_input_scale_vector_np = input_scale_vector_for_mode(
            self.input_mode,
            state_dim=self.state_dim,
            q0=self.q0_input_scale,
        )
        self.state_input_scale_vector = torch.tensor(
            self.state_input_scale_vector_np,
            device=self.device,
            dtype=torch.float32,
        )
        self.checkpoint_reparam_load_mode = str(
            getattr(self.net_config, "checkpoint_reparam_load_mode", "guard")
        )
        if self.checkpoint_reparam_load_mode not in {"guard", "convert"}:
            raise ValueError("checkpoint_reparam_load_mode must be 'guard' or 'convert'.")
        self.state_support_logging = bool(getattr(self.net_config, "state_support_logging", False))
        self.state_support_sample_size = int(
            getattr(self.net_config, "state_support_sample_size", min(int(self.valid_size), 16384))
        )
        self.inventory_zero_mass = float(getattr(self.eqn_config, "inventory_zero_mass", 0.05))
        self.inventory_near_zero_mass = float(getattr(self.eqn_config, "inventory_near_zero_mass", 0.05))
        self.inventory_near_zero_fraction = float(
            getattr(self.eqn_config, "inventory_near_zero_fraction", 0.05)
        )
        self.sampling_mode = self._normalize_sampling_mode(
            getattr(self.net_config, "sampling_mode", "box")
        )
        self.rollout_pool_size = int(
            getattr(
                self.net_config,
                "rollout_pool_size",
                max(int(self.batch_size), int(self.valid_size)),
            )
        )
        self.sampling_seed = getattr(self.net_config, "sampling_seed", None)
        self.stateless_seed_schedule = bool(
            getattr(self.net_config, "stateless_seed_schedule", False)
        )
        self.seed_schedule_experiment = str(
            getattr(self.net_config, "seed_schedule_experiment", "default")
        )
        self.seed_schedule_key = str(
            getattr(
                self.net_config,
                "seed_schedule_key",
                self.sampling_seed if self.sampling_seed is not None else getattr(self.net_config, "train_seed", 0),
            )
        )
        self.proposal_sampler = getattr(self.net_config, "proposal_sampler", None)
        self.mix_rollout = float(
            getattr(
                self.net_config,
                "mix_rollout",
                getattr(self.net_config, "mixed_rollout_fraction", 0.5),
            )
        )
        self.mix_box = float(getattr(self.net_config, "mix_box", 0.5))
        self.mix_shortage = float(getattr(self.net_config, "mix_shortage", 0.0))
        self.shortage_q_frac = float(getattr(self.net_config, "shortage_q_frac", 0.3))
        self.shortage_z_frac = float(getattr(self.net_config, "shortage_z_frac", 0.15))
        self.consistency_coef = float(getattr(self.net_config, "consistency_coef", 0.0))
        self.training_consistency_coef = (
            self.consistency_coef
            if self.loss_mode == LOSS_MODE_CE_CONSISTENCY
            else 0.0
        )
        self.gradient_assembly_mode = str(
            getattr(self.net_config, "gradient_assembly_mode", "standard")
        )
        self.h_base_grad_multiplier = float(
            getattr(self.net_config, "h_base_grad_multiplier", 1.0)
        )
        self.g_base_grad_multiplier = float(
            getattr(self.net_config, "g_base_grad_multiplier", 1.0)
        )
        self.h_consistency_coef = float(
            getattr(self.net_config, "h_consistency_coef", self.consistency_coef)
        )
        self.g_consistency_coef = float(
            getattr(self.net_config, "g_consistency_coef", self.consistency_coef)
        )
        self.use_component_gradient_assembly = (
            self.gradient_assembly_mode == "hg_component"
            or not np.isclose(self.h_base_grad_multiplier, 1.0)
            or not np.isclose(self.g_base_grad_multiplier, 1.0)
            or not np.isclose(self.h_consistency_coef, self.consistency_coef)
            or not np.isclose(self.g_consistency_coef, self.consistency_coef)
        )
        if self.gradient_assembly_mode not in {"standard", "hg_component"}:
            raise ValueError("gradient_assembly_mode must be 'standard' or 'hg_component'.")
        if self.requires_qgrad and self.use_component_gradient_assembly:
            raise ValueError(
                "martingale_qgrad does not support hg_component gradient assembly; "
                "LOSS-00 requires the standard joint Adam path without gradient multipliers."
            )
        self.reference_mixture = getattr(
            self.eqn_config,
            "reference_mixture",
            None,
        )
        self.reference_mixture_counts = self._parse_reference_mixture_counts(
            getattr(self.eqn_config, "reference_mixture_counts", None)
        )
        self.uniform_z_radius_mult = float(getattr(self.net_config, "uniform_z_radius_mult", 1.0))
        self.uniform_q_radius_frac = float(getattr(self.net_config, "uniform_q_radius_frac", 0.10))
        self.total_inventory_buffer = float(getattr(self.net_config, "total_inventory_buffer", 20.0))
        self.boundary_eps_frac = float(getattr(self.net_config, "boundary_eps_frac", 0.05))
        q_low_values = getattr(self.net_config, "q_low_values", [0.0, 1.0, 5.0, 10.0, 25.0, 50.0])
        if isinstance(q_low_values, str):
            q_low_values = [float(part.strip()) for part in q_low_values.split(",") if part.strip()]
        self.q_low_values = q_low_values
        self.lognormal_delta_z = float(getattr(self.net_config, "lognormal_delta_z", 1e-2))
        self.lognormal_delta_q = float(getattr(self.net_config, "lognormal_delta_q", 1e-3))
        self.lognormal_r_min = int(getattr(self.net_config, "lognormal_r_min", 20))
        self.initial_state_mode = getattr(self.net_config, "initial_state_mode", "standard")
        self.initial_perturb_prob = float(
            getattr(self.net_config, "initial_perturb_prob", 0.30)
        )
        self.initial_q_perturb_frac = float(
            getattr(self.net_config, "initial_q_perturb_frac", 0.05)
        )
        self.initial_z_perturb_frac = float(
            getattr(self.net_config, "initial_z_perturb_frac", 0.0125)
        )
        self.initial_perturb_minus_prob = float(
            getattr(self.net_config, "initial_perturb_minus_prob", 0.20)
        )
        self.initial_perturb_plus_prob = float(
            getattr(self.net_config, "initial_perturb_plus_prob", 0.10)
        )
        self.initial_perturb_minus_frac = float(
            getattr(self.net_config, "initial_perturb_minus_frac", 0.10)
        )
        self.initial_perturb_plus_frac = float(
            getattr(self.net_config, "initial_perturb_plus_frac", 0.05)
        )
        self.initial_dirichlet_alpha = float(
            getattr(self.net_config, "initial_dirichlet_alpha", 1.0)
        )
        self.state_sampler = TrainingStateSampler(
            bsde=self.bsde,
            device=self.device,
            dim=self.dim,
            num_time_interval=self.num_time_interval,
            sampling_mode=self.sampling_mode,
            rollout_pool_size=self.rollout_pool_size,
            mix_rollout=self.mix_rollout,
            mix_box=self.mix_box,
            mix_shortage=self.mix_shortage,
            sampling_seed=self.sampling_seed,
            box_state_sampler=self._sample_box_states,
            shortage_state_sampler=self._sample_shortage_states,
            reference_transition=self._reference_transition,
            reference_label_sampler=self._sample_reference_policy_labels,
            proposal_sampler=self.proposal_sampler,
            uniform_z_radius_mult=self.uniform_z_radius_mult,
            uniform_q_radius_frac=self.uniform_q_radius_frac,
            total_inventory_buffer=self.total_inventory_buffer,
            q_low_values=self.q_low_values,
            boundary_eps_frac=self.boundary_eps_frac,
            lognormal_delta_z=self.lognormal_delta_z,
            lognormal_delta_q=self.lognormal_delta_q,
            lognormal_r_min=self.lognormal_r_min,
            initial_state_mode=self.initial_state_mode,
            initial_perturb_prob=self.initial_perturb_prob,
            initial_q_perturb_frac=self.initial_q_perturb_frac,
            initial_z_perturb_frac=self.initial_z_perturb_frac,
            initial_perturb_minus_prob=self.initial_perturb_minus_prob,
            initial_perturb_plus_prob=self.initial_perturb_plus_prob,
            initial_perturb_minus_frac=self.initial_perturb_minus_frac,
            initial_perturb_plus_frac=self.initial_perturb_plus_frac,
            initial_dirichlet_alpha=self.initial_dirichlet_alpha,
        )

    def _move_optional_bsde_tensor(self, attr_name):
        value = getattr(self.bsde, attr_name, None)
        if isinstance(value, torch.Tensor):
            setattr(self.bsde, attr_name, value.to(self.device))

    def _build_gamma_sqrt_cache(self):
        """Cache per-interval diffusion square roots used by the reference process."""
        factors = []
        for n in range(self.num_time_interval):
            Gamma_t = self.bsde.Gamma[n, :, :]
            try:
                L_chol = torch.linalg.cholesky(Gamma_t)
            except Exception:
                diag_gamma = torch.diag(Gamma_t)
                L_chol = torch.diag(torch.sqrt(torch.clamp(diag_gamma, min=0.0)))
            factors.append(L_chol)
        return torch.stack(factors, dim=0)

    @staticmethod
    def _normalize_sampling_mode(mode):
        if mode == "mixed":
            return "mixture"
        if mode in {"lognormal", "proposal"}:
            return "lognormal_proposal"
        return mode

    def _resolve_loss_mode(self):
        explicit = getattr(self.net_config, "loss_mode", None)
        legacy = getattr(self.net_config, "loss_type", None)
        if explicit is not None and str(explicit).strip():
            mode = str(explicit).strip()
            if mode not in LOSS_MODE_CHOICES:
                raise ValueError(
                    f"loss_mode must be one of {LOSS_MODE_CHOICES}, got {mode!r}."
                )
            if legacy is not None:
                legacy_int = int(legacy)
                if legacy_int == 1:
                    raise ValueError(
                        "loss_type=1 cannot be combined with loss_mode. "
                        "The legacy martingale loss is not available for the "
                        "current gradient-difference network."
                    )
                if legacy_int != 2:
                    raise ValueError(f"Unsupported legacy loss_type={legacy!r}.")
            return mode

        if legacy is None:
            return LOSS_MODE_CE_CONSISTENCY
        legacy_int = int(legacy)
        if legacy_int == 2:
            warnings.warn(
                "loss_type=2 is deprecated; use loss_mode='ce_consistency'.",
                DeprecationWarning,
                stacklevel=2,
            )
            return LOSS_MODE_CE_CONSISTENCY
        if legacy_int == 1:
            raise ValueError(
                "loss_type=1 assumes access to the full state gradient. "
                "The gradient-difference refactor exposes only D V, so use "
                "loss_mode='martingale_qgrad' after the LOSS-00 checks."
            )
        raise ValueError(f"Unsupported legacy loss_type={legacy!r}.")

    @staticmethod
    def _parse_reference_mixture_counts(value):
        if value is None or value == "":
            return None
        if isinstance(value, str):
            counts = {}
            for part in value.split(","):
                if not part.strip():
                    continue
                key, raw_count = part.split(":", 1)
                counts[key.strip()] = int(raw_count)
            return counts
        if isinstance(value, dict):
            return {str(key): int(count) for key, count in value.items()}
        raise ValueError("reference_mixture_counts must be a dict or R0:count,R1:count string.")

    def _default_reference_mixture_counts(self):
        mixture = str(self.reference_mixture or "").upper()
        if mixture == "R0":
            return {"R0": self.rollout_pool_size}
        if mixture == "R1":
            return {"R1": self.rollout_pool_size}
        if mixture == "R2":
            return {"R2": self.rollout_pool_size}
        if mixture == "R3":
            return {"R0": 8192, "R2": 8192}
        if mixture == "R4":
            return {"R0": 8192, "R1": 4096, "R2": 4096}
        policy = str(getattr(self.bsde, "policy", ""))
        label = REFERENCE_LABEL_BY_POLICY.get(policy)
        if label is None:
            return None
        for name, candidate_label in REFERENCE_POLICY_LABELS.items():
            if candidate_label == label:
                return {name: self.rollout_pool_size}
        return None

    def _sample_reference_policy_labels(self, rollout_pool_size):
        counts = self.reference_mixture_counts or self._default_reference_mixture_counts()
        if counts is None:
            return torch.full((int(rollout_pool_size),), -1, dtype=torch.int64, device=self.device)
        total = sum(int(count) for count in counts.values())
        if total != int(rollout_pool_size):
            raise ValueError(
                "reference path counts must sum to rollout_pool_size: "
                f"counts={counts}, total={total}, rollout_pool_size={rollout_pool_size}."
            )
        labels = []
        for name in ("R0", "R1", "R2"):
            count = int(counts.get(name, 0))
            if count < 0:
                raise ValueError(f"reference path count for {name} must be nonnegative.")
            labels.extend([REFERENCE_POLICY_LABELS[name]] * count)
        unknown = sorted(set(counts) - {"R0", "R1", "R2"})
        if unknown:
            raise ValueError(f"Unknown reference policy labels in counts: {unknown}.")
        return torch.tensor(labels, dtype=torch.int64, device=self.device)

    def reparameterization_metadata(self):
        return reparameterization_metadata(
            input_mode=self.input_mode,
            state_dim=self.state_dim,
            q0=self.q0_input_scale,
            h_output_scale=self.h_output_scale,
            extra={
                "loss_mode": self.loss_mode,
                "legacy_loss_type": self.loss_type,
                "training_consistency_coef": self.training_consistency_coef,
                "validation_every": self.validation_every,
                "patience_iterations": self.patience_iterations,
                "patience_checks": self.patience_checks,
                "checkpoint_reparam_load_mode": self.checkpoint_reparam_load_mode,
                "final_grad_clip_rule": self.final_grad_clip_rule,
                "inner_grad_clip_rule": self.inner_grad_clip_rule,
                "h_grad_norm": self.h_grad_norm,
                "g_grad_norm": self.g_grad_norm,
                "h_init_method": self.h_init_method,
                "g_init_method": self.g_init_method,
                "g_constructor_init_method": self.g_constructor_init_method,
                "g_output_bias": self.g_output_bias,
                "component_seed_isolation": self.component_seed_isolation,
                "h_network_seed": self.h_network_seed if self.component_seed_isolation else "",
                "g_network_seed": self.g_network_seed if self.component_seed_isolation else "",
                "qgrad_network_seed": self.qgrad_network_seed if self.component_seed_isolation else "",
                "save_initial_checkpoints": self.save_initial_checkpoints,
            },
        )

    def _network_input(self, physical_states):
        return apply_state_input_scaling(physical_states, self.state_input_scale_vector)

    def _h_physical_output(self, y_net, physical_states):
        return y_net(self._network_input(physical_states)) / self.h_output_scale

    def _g_physical_output(self, g_net, physical_states):
        return g_net(self._network_input(physical_states))

    def _qgrad_physical_output(self, qgrad_net, physical_states):
        if qgrad_net is None:
            return None
        return qgrad_net(self._network_input(physical_states)).squeeze(1)

    def _physical_gradient_difference(self, y_net, physical_states, *, create_graph):
        return compute_physical_DH(
            y_net,
            physical_states,
            grad_diff_dim=self.grad_diff_dim,
            scale_vector=self.state_input_scale_vector,
            h_output_scale=self.h_output_scale,
            create_graph=create_graph,
            retain_graph=create_graph,
        )

    def _apply_scratch_reparameterization(self, y_net_list, z_net_list, qgrad_net_list=None):
        if np.allclose(self.state_input_scale_vector_np, 1.0) and np.isclose(self.h_output_scale, 1.0):
            return
        for y_net in y_net_list:
            transform_first_layer(y_net, self.state_input_scale_vector_np)
            transform_h_output_layer(y_net, self.h_output_scale)
        for z_net in z_net_list:
            transform_first_layer(z_net, self.state_input_scale_vector_np)
        if qgrad_net_list is not None:
            for qgrad_net in qgrad_net_list:
                transform_first_layer(qgrad_net, self.state_input_scale_vector_np)

    def _representation_from_config(self, config):
        net = (config or {}).get("net_config", {})
        eqn = (config or {}).get("eqn_config", {})
        source_input_mode = str(net.get("input_mode", "raw"))
        source_q0 = float(net.get("q0_input_scale", eqn.get("I0", self.q0_input_scale)))
        source_h_scale = float(net.get("h_output_scale", 1.0))
        return {
            "input_mode": source_input_mode,
            "q0_input_scale": source_q0,
            "h_output_scale": source_h_scale,
            "input_scale_vector": input_scale_vector_for_mode(
                source_input_mode,
                state_dim=self.state_dim,
                q0=source_q0,
            ),
        }

    def _load_source_reparameterization(self, checkpoint_dir):
        run_dir = os.path.dirname(os.path.abspath(str(checkpoint_dir)))
        config_path = os.path.join(run_dir, "config_used.json")
        if not os.path.isfile(config_path):
            config_path = os.path.join(run_dir, "config_scenario.json")
        if not os.path.isfile(config_path):
            return None
        import json

        with open(config_path, encoding="utf-8") as handle:
            return self._representation_from_config(json.load(handle))

    def _maybe_convert_loaded_state(self, state, *, is_h, source_repr):
        if source_repr is None:
            return state
        source_input = source_repr["input_scale_vector"]
        source_h = float(source_repr.get("h_output_scale", 1.0))
        same_input = np.allclose(source_input, self.state_input_scale_vector_np, rtol=1e-12, atol=1e-12)
        same_h = np.isclose(source_h, self.h_output_scale, rtol=1e-12, atol=1e-12)
        if same_input and (not is_h or same_h):
            return state
        if self.checkpoint_reparam_load_mode != "convert":
            raise ValueError(
                "Source checkpoint representation does not match target run. "
                "Set checkpoint_reparam_load_mode=convert only when you intend "
                "to function-preservingly convert raw/scaled checkpoints."
            )
        if is_h:
            return transform_h_state_dict_between_reparameterizations(
                state,
                source_input_scale=source_input,
                target_input_scale=self.state_input_scale_vector_np,
                source_h_output_scale=source_h,
                target_h_output_scale=self.h_output_scale,
            )
        return transform_g_state_dict_between_reparameterizations(
            state,
            source_input_scale=source_input,
            target_input_scale=self.state_input_scale_vector_np,
        )

    def _sample_inventory_states(self, num_sample):
        """Mix exact-zero, near-zero, and interior states so the boundary is learned."""
        probs = torch.rand(num_sample, 1, device=self.device)
        I_n = torch.empty(num_sample, 1, device=self.device)

        zero_mass = min(max(self.inventory_zero_mass, 0.0), 1.0)
        near_mass = min(max(self.inventory_near_zero_mass, 0.0), max(1.0 - zero_mass, 0.0))
        near_cap = max(self.inventory_near_zero_fraction, 0.0) * self.bsde.I0

        zero_mask = probs <= zero_mass
        near_mask = (probs > zero_mass) & (probs <= zero_mass + near_mass)
        interior_mask = ~(zero_mask | near_mask)

        I_n[zero_mask] = 0.0
        if near_mask.any():
            I_n[near_mask] = torch.rand_like(I_n[near_mask]) * near_cap
        if interior_mask.any():
            I_n[interior_mask] = torch.rand_like(I_n[interior_mask]) * self.bsde.I0

        return I_n

    
    def sample_states(self, n, batch_size, mode=None, return_metadata=False):
        """Sample interval-start states Y(t_n) from the configured backend."""
        return self.state_sampler.sample_states(
            n,
            batch_size,
            mode=mode,
            return_metadata=return_metadata,
        )

    # Generate sample paths
    def sample(
        self,
        num_sample,
        n,
        mode=None,
        return_metadata=False,
        sample_purpose=None,
        iteration_idx=None,
    ):
        """
        Generate sample paths for LiberiaHJB with reflected Brownian motion.

        Args:
            num_sample (int): Number of samples to generate.
            n (int): Current time interval index.
            mode (str, optional): Override configured state-sampling backend.

        Returns:
            Tuple: (Y_n, Y_{n+1}, a_ref, Delta_L, W_increment)
            - Y_n: Current state (Z_n, I_n) where Z is (K+1) dim and I is 1 dim
            - Y_{n+1}: Next state (Z_{n+1}, I_{n+1})
            - a_ref: Reference allocation policy
            - Delta_L: Regulator increments
            - W_increment: Brownian motion increments
        """
        if (
            self.stateless_seed_schedule
            and sample_purpose is not None
            and iteration_idx is not None
        ):
            return self._sample_with_stateless_seed(
                num_sample,
                n,
                mode=mode,
                return_metadata=return_metadata,
                sample_purpose=sample_purpose,
                iteration_idx=iteration_idx,
            )
        return self._sample_once(num_sample, n, mode=mode, return_metadata=return_metadata)

    def _sample_with_stateless_seed(
        self,
        num_sample,
        n,
        *,
        mode=None,
        return_metadata=False,
        sample_purpose,
        iteration_idx,
    ):
        seed = stateless_seed_from_components(
            self.seed_schedule_experiment,
            self.seed_schedule_key,
            int(n),
            int(iteration_idx),
            str(sample_purpose),
        )
        cpu_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        sampler_state = self.state_sampler.generator.get_state()
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            self.state_sampler.generator.manual_seed(seed)
            result = self._sample_once(num_sample, n, mode=mode, return_metadata=return_metadata)
            if return_metadata:
                *values, metadata = result
                metadata = dict(metadata)
                metadata["stateless_seed"] = int(seed)
                metadata["sample_purpose"] = str(sample_purpose)
                metadata["iteration_idx"] = int(iteration_idx)
                return (*values, metadata)
            return result
        finally:
            self.state_sampler.generator.set_state(sampler_state)
            torch.random.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def _sample_once(self, num_sample, n, mode=None, return_metadata=False):
        Y_n, metadata = self.sample_states(n, num_sample, mode=mode, return_metadata=True)
        labels = metadata.get("reference_policy_labels")
        path_ids = metadata.get("path_ids")
        if labels is not None:
            labels = labels.to(device=self.device, dtype=torch.int64)
        if path_ids is not None:
            path_ids = path_ids.to(device=self.device, dtype=torch.int64)
        Y_n1, a_ref, Delta_L, W_increment = self._reference_transition(
            n,
            Y_n,
            reference_policy_labels=labels,
            path_ids=path_ids,
        )
        if return_metadata:
            metadata = dict(metadata)
            if labels is not None:
                metadata["reference_policy_labels"] = labels.detach().cpu()
            if path_ids is not None:
                metadata["path_ids"] = path_ids.detach().cpu()
            return Y_n, Y_n1, a_ref, Delta_L, W_increment, metadata
        return Y_n, Y_n1, a_ref, Delta_L, W_increment

    def _sample_liberia(self, num_sample, n):
        """Backward-compatible wrapper for the legacy box-sampling path."""
        return self.sample(num_sample, n, mode="box")

    def _sample_box_states(self, num_sample, n):
        """Sample Y(t_n) from the time-dependent box domain used in legacy training."""
        Z_max = self._compute_box_z_max(n)
        Z_n = torch.rand(num_sample, self.dim, device=self.device) * Z_max
        I_n = self._sample_inventory_states(num_sample)
        return torch.cat([Z_n, I_n], dim=1)

    def _sample_shortage_states(self, num_sample, n):
        """Sample low-inventory states to emphasize shortage-sensitive regions."""
        Z_max = self._compute_box_z_max(n)
        shortage_z_cap = max(self.shortage_z_frac, 0.0) * Z_max
        shortage_q_cap = max(self.shortage_q_frac, 0.0) * self.bsde.I0

        Z_n = torch.rand(num_sample, self.dim, device=self.device) * shortage_z_cap
        I_n = torch.rand(num_sample, 1, device=self.device) * shortage_q_cap
        return torch.cat([Z_n, I_n], dim=1)

    def _state_support_quantile_rows(self, states, *, interval, source):
        arr = states.detach().cpu().numpy().reshape(-1, states.shape[-1])
        z = arr[:, : self.dim]
        q = arr[:, self.dim]
        variables = [("Q", q)]
        for idx in range(self.dim):
            variables.append((f"Z_{idx}", z[:, idx]))
        if self.dim > 1:
            cha = z[:, 1:]
            variables.extend(
                [
                    ("aggregate_CHA_inventory", np.sum(cha, axis=1)),
                    ("minimum_CHA_inventory", np.min(cha, axis=1)),
                    ("maximum_CHA_inventory", np.max(cha, axis=1)),
                ]
            )
        else:
            variables.extend(
                [
                    ("aggregate_CHA_inventory", np.zeros_like(q)),
                    ("minimum_CHA_inventory", np.zeros_like(q)),
                    ("maximum_CHA_inventory", np.zeros_like(q)),
                ]
            )
        rows = []
        for name, values in variables:
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size == 0:
                summary = {key: np.nan for key in ("min", "p01", "p05", "p50", "p95", "p99", "max")}
                count = 0
            else:
                summary = {
                    "min": float(np.min(finite)),
                    "p01": float(np.percentile(finite, 1)),
                    "p05": float(np.percentile(finite, 5)),
                    "p50": float(np.percentile(finite, 50)),
                    "p95": float(np.percentile(finite, 95)),
                    "p99": float(np.percentile(finite, 99)),
                    "max": float(np.max(finite)),
                }
                count = int(finite.size)
            rows.append(
                {
                    "interval": int(interval),
                    "source": str(source),
                    "variable": name,
                    "count": count,
                    **summary,
                }
            )
        return rows

    def _append_state_support_rows(self, path, rows):
        if not rows:
            return
        write_header = not os.path.isfile(path) or os.path.getsize(path) == 0
        fieldnames = [
            "interval",
            "source",
            "variable",
            "count",
            "min",
            "p01",
            "p05",
            "p50",
            "p95",
            "p99",
            "max",
        ]
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})

    def _log_state_support_summary(self, path, interval, training_states):
        if not self.state_support_logging or path is None:
            return
        rows = self._state_support_quantile_rows(
            training_states,
            interval=interval,
            source="training_sampler",
        )
        try:
            reference_states = self.sample_states(
                interval,
                max(int(self.state_support_sample_size), 1),
                mode="rollout",
            )
            rows.extend(
                self._state_support_quantile_rows(
                    reference_states,
                    interval=interval,
                    source="reference_rollout",
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "interval": int(interval),
                    "source": "reference_rollout_error",
                    "variable": str(exc),
                    "count": 0,
                }
            )
        self._append_state_support_rows(path, rows)

    def _compute_box_z_max(self, n):
        mu_t = self.bsde.mu[n, :]
        remaining_time = (self.num_time_interval - n) * self.delta_t
        mu_max = mu_t.max().item()
        return max(mu_max * remaining_time * 2.0, 50.0)

    def _parameters_match_across_intervals(self, n, next_n):
        """Check time-varying scientific parameters at an initialization boundary."""
        mu_diff = torch.max(torch.abs(self.bsde.mu[n] - self.bsde.mu[next_n])).item()
        gamma_diff = torch.max(
            torch.abs(self.bsde.Gamma[n] - self.bsde.Gamma[next_n])
        ).item()
        eta_diff = torch.max(
            torch.abs(
                self.bsde.eta_at_fine_index(n)
                - self.bsde.eta_at_fine_index(next_n)
            )
        ).item()
        tol = self.prev_init_param_tol
        mu_matches = mu_diff <= tol
        gamma_matches = gamma_diff <= tol
        eta_matches = eta_diff <= tol
        return {
            "mu_matches": mu_matches,
            "gamma_matches": gamma_matches,
            "eta_matches": eta_matches,
            "mu_max_abs_diff": mu_diff,
            "gamma_max_abs_diff": gamma_diff,
            "eta_max_abs_diff": eta_diff,
            "tol": tol,
        }

    def _get_interval_init_decision(self, n):
        """
        Decide whether interval n should warm-start from n+1 or keep fresh init.

        Returns a dict that is also suitable for concise training logs/tests.
        """
        decision = {
            "reuse_prev_init": False,
            "reason": "prev_init_disabled",
            "mu_matches": True,
            "gamma_matches": True,
            "eta_matches": True,
            "mu_max_abs_diff": 0.0,
            "gamma_max_abs_diff": 0.0,
            "eta_max_abs_diff": 0.0,
            "tol": self.prev_init_param_tol,
        }
        if not self.prev_init:
            return decision

        if not self.prev_init_same_params_only:
            decision["reuse_prev_init"] = True
            decision["reason"] = "prev_init_unconditional"
            return decision

        match_info = self._parameters_match_across_intervals(n, n + 1)
        decision.update(match_info)
        if (
            match_info["mu_matches"]
            and match_info["gamma_matches"]
            and match_info["eta_matches"]
        ):
            decision["reuse_prev_init"] = True
            decision["reason"] = "params_match_within_tol"
        else:
            decision["reason"] = "params_changed_beyond_tol"
        return decision

    def _get_interval_training_schedule(self, init_decision):
        """
        Choose optimization budget for a non-terminal interval.

        Intervals that were freshly reinitialized because adjacent mu/Gamma changed
        should train like the final interval in terms of optimizer budget, while
        preserving the non-terminal loss target. These jump intervals also skip
        early stopping so the larger optimization budget is actually used.
        """
        use_final_budget = init_decision["reason"] == "params_changed_beyond_tol"
        if use_final_budget:
            return {
                "iterations": self.net_config.final_iterations,
                "learning_rate": self.final_learning_rate,
                "milestones": self.final_milestones,
                "gamma": self.final_gamma,
                "minimum_iterations": self.final_min_iterations,
                "budget_type": "final",
                "early_stopping": False,
            }
        return {
            "iterations": self.net_config.inner_iterations,
            "learning_rate": self.inner_learning_rate,
            "milestones": self.inner_milestones,
            "gamma": self.inner_gamma,
            "minimum_iterations": self.inner_min_iterations,
            "budget_type": "inner",
            "early_stopping": self.early_stopping,
        }

    def _reference_transition(
        self,
        n,
        Y_n,
        generator=None,
        reference_policy_labels=None,
        path_ids=None,
    ):
        """
        Advance a batch of interval-start states through the reference-process kernel.

        Args:
            n: Time interval index (0 to num_time_interval-1)
            Y_n: Current state batch with shape (num_sample, K+2)

        Returns:
            Tuple: (Y_{n+1}, a_ref, Delta_L, W_increment)
        """
        num_sample = Y_n.shape[0]
        mu_t = self.bsde.mu[n, :]
        Z_n = Y_n[:, :self.dim]
        I_n = Y_n[:, self.dim:self.dim + 1]

        del path_ids  # path ids are preserved in sampler metadata; transition only needs the label.

        # Generate reference policy (allocation rates).  Mixtures are path-level:
        # each sampled state carries the label assigned when its rollout path was
        # created, and the same label is used for action, transition, and F
        # correction through a_ref.
        if reference_policy_labels is None:
            if self.bsde.policy == REFERENCE_MIXTURE_POLICY:
                raise ValueError("reference_mixture transitions require reference_policy_labels.")
            a_ref = self.bsde.generate_u_sample(num_sample, n=n, Z_n=Z_n, I_n=I_n, mu_t=mu_t)
        else:
            labels = reference_policy_labels.to(device=self.device, dtype=torch.int64).reshape(-1)
            if labels.numel() != num_sample:
                raise ValueError(
                    "reference_policy_labels length does not match batch: "
                    f"{labels.numel()} vs {num_sample}."
                )
            a_ref = torch.zeros(num_sample, self.dim, device=self.device, dtype=Y_n.dtype)
            unresolved = labels < 0
            if bool(unresolved.any()):
                if self.bsde.policy == REFERENCE_MIXTURE_POLICY:
                    raise ValueError("reference_mixture received states without path-level labels.")
                idx = unresolved.nonzero(as_tuple=False).flatten()
                a_ref[idx] = self.bsde.generate_u_sample(
                    int(idx.numel()),
                    n=n,
                    Z_n=Z_n[idx],
                    I_n=I_n[idx],
                    mu_t=mu_t,
                )
            for label_value, policy_name in REFERENCE_POLICY_BY_LABEL.items():
                mask = labels == int(label_value)
                if not bool(mask.any()):
                    continue
                idx = mask.nonzero(as_tuple=False).flatten()
                a_ref[idx] = self.bsde.generate_u_sample_for_policy(
                    policy_name,
                    int(idx.numel()),
                    n=n,
                    Z_n=Z_n[idx],
                    I_n=I_n[idx],
                    mu_t=mu_t,
                )
            known_mask = torch.zeros_like(labels, dtype=torch.bool)
            for label_value in REFERENCE_POLICY_BY_LABEL:
                known_mask = known_mask | (labels == int(label_value))
            unknown = labels[(labels >= 0) & ~known_mask]
            if int(unknown.numel()) > 0:
                raise ValueError(f"Unknown reference policy labels: {torch.unique(unknown).detach().cpu().tolist()}.")

        eta_t = (
            self.bsde.eta_at_fine_index(n)
            if hasattr(self.bsde, "eta_at_fine_index")
            else self.bsde.eta
        )
        a_ref, _ = project_rate_to_inventory_torch(
            a_ref,
            I_n,
            self.delta_t,
            eta_rate=eta_t,
        )
        
        # Step 1: Generate Brownian Increments
        # Generate ξ ~ N(0, I_{K+1})
        if generator is None:
            xi = torch.randn(num_sample, self.dim, device=self.device)  # (num_sample, K+1)
        else:
            xi = torch.randn(
                (num_sample, self.dim),
                generator=generator,
                dtype=Y_n.dtype,
            ).to(self.device)
        
        # Compute diffusion term with the cached Γ^{1/2}(t_n).
        L_chol = self.gamma_sqrt[n]  # (K+1, K+1)
        
        # \Delta W_n = L_chol @ ξ^T * √(\Delta t), then transpose back
        W_increment = (xi @ L_chol.T) * self.sqrt_delta_t  # (num_sample, K+1)
        
        # Step 2: Euler Step
        # Retrieve control: a_n = a(t_n, Z(t_n), I(t_n))
        # (already have a_ref)
        
        # X_{n+1} = Z(t_n) + (a_n - μ(t_n)) \Delta t + \Delta W_n
        drift_term = (a_ref - mu_t.unsqueeze(0)) * self.delta_t  # (num_sample, K+1)
        X_1 = Z_n + drift_term + W_increment  # (num_sample, K+1)
        
        # Step 3: Reflection Step
        # Initialize regulator increments vector \Delta L(t_n) = 0
        Delta_L = torch.zeros(num_sample, self.dim, device=self.device)  # (num_sample, K+1)
        
        # For CHAs (k=1,2,...,K): \Delta L_k = max(-X_{n+1,k}, 0)
        # X[:, 1:] corresponds to CHAs (indices 1 to K)
        Delta_L_CHAs = torch.relu(-X_1[:, 1:])  # (num_sample, K)
        Delta_L[:, 1:] = Delta_L_CHAs
        
        # For HF (k=0): 
        # First calculate referral load: sum(p_k * \Delta L_k) for k=1,...,K
        # p is shape (K,), Delta_L_CHAs is (num_sample, K)
        p_travel = self.bsde.p.unsqueeze(0)  # (1, K)
        referral_load = torch.sum(p_travel * Delta_L_CHAs, dim=1)  # (num_sample,)
        
        # Calculate effective HF inventory: X_{n+1,0} - referral_load
        effective_HF_inventory = X_1[:, 0] - referral_load  # (num_sample,)
        
        # \Delta L_0 = max(-effective_HF_inventory, 0)
        Delta_L[:, 0] = torch.relu(-effective_HF_inventory)  # (num_sample,)
        
        # Step 4: State Update
        # Z_0(t_{n+1}) = X_{n+1,0} + \Delta L_0 - sum(p_k * \Delta L_k)
        Z_1 = torch.zeros_like(X_1)  # (num_sample, K+1)
        Z_1[:, 0] = X_1[:, 0] + Delta_L[:, 0] - referral_load
        
        # Z_k(t_{n+1}) = X_{n+1,k} + \Delta L_k for k=1,2,...,K
        Z_1[:, 1:] = X_1[:, 1:] + Delta_L[:, 1:]
        
        # I(t_{n+1}) = I(t_n) - (sum_{k=0}^K a_{n,k}) \Delta t
        I_1 = I_n - a_ref.sum(dim=1, keepdim=True) * self.delta_t  # (num_sample, 1)
        I_1 = torch.clamp(I_1, min=0.0)  # Ensure non-negative
        
        Y_1 = torch.cat([Z_1, I_1], dim=1)  # (num_sample, K+2)
        
        # Return Y_1 (state at interval end), a_ref, Delta_L, and W_increment.
        return Y_1, a_ref, Delta_L, W_increment
    
    def ce_residual(self, Y_n, Y_n1, H_n, H_n1, D_n, a_ref, L_increment, n=0, is_final=False):
        """Return the conditional-expectation residual using physical D V units."""
        I_n = Y_n[:, self.dim:self.dim + 1].squeeze(1)
        Z_n1 = Y_n1[:, :self.dim]
        mu_t = self.bsde.mu[n, :]
        F_val = self.bsde.F_function(n, I_n, D_n, mu_t, self.delta_t)
        a_ref_D = (a_ref * D_n).sum(dim=1)
        c_tilde_L = (self.bsde.c_tilde.unsqueeze(0) * L_increment).sum(dim=1)
        if is_final:
            target = self.bsde.g_tf(Z_n1).squeeze(1)
        else:
            target = H_n1.detach()
        return target - H_n + F_val * self.delta_t - a_ref_D * self.delta_t + c_tilde_L

    def reconstruct_grad_z(self, D_n, Q_n):
        """Reconstruct physical grad_z V from D V and scalar partial_q V."""
        if Q_n is None:
            raise ValueError("QGrad output is required to reconstruct grad_z V.")
        q = Q_n.reshape(-1, 1)
        return D_n + q.expand_as(D_n)

    def loss_components(
        self,
        Y_n,
        Y_n1,
        H_n,
        H_n1,
        D_n,
        a_ref,
        L_increment,
        W_increment=None,
        Q_n=None,
        n=0,
        is_final=False,
        loss_mode=None,
    ):
        mode = loss_mode or self.loss_mode
        if mode not in LOSS_MODE_CHOICES:
            raise ValueError(f"Unsupported loss_mode={mode!r}.")
        residual_ce = self.ce_residual(
            Y_n=Y_n,
            Y_n1=Y_n1,
            H_n=H_n,
            H_n1=H_n1,
            D_n=D_n,
            a_ref=a_ref,
            L_increment=L_increment,
            n=n,
            is_final=is_final,
        )
        ce_loss = torch.mean(residual_ce ** 2)
        zero = residual_ce.new_zeros(())
        martingale_term = residual_ce.new_zeros(residual_ce.shape)
        residual_native = residual_ce
        martingale_loss = zero
        if mode == LOSS_MODE_MARTINGALE_QGRAD:
            if W_increment is None:
                raise ValueError("W_increment is required for martingale_qgrad.")
            grad_z_hat = self.reconstruct_grad_z(D_n, Q_n)
            martingale_term = (grad_z_hat * W_increment).sum(dim=1)
            residual_native = residual_ce - martingale_term
            martingale_loss = torch.mean(residual_native ** 2)
        native_loss = martingale_loss if mode == LOSS_MODE_MARTINGALE_QGRAD else ce_loss
        return {
            "loss_mode": mode,
            "ce_residual": residual_ce,
            "ce_loss": ce_loss,
            "martingale_term": martingale_term,
            "martingale_loss": martingale_loss,
            "native_residual": residual_native,
            "native_loss": native_loss,
        }

    def loss_fn(
        self,
        Y_n,
        Y_n1,
        H_n,
        H_n1,
        D_n,
        a_ref,
        L_increment,
        W_increment=None,
        Q_n=None,
        n=0,
        is_final=False,
        loss_type=None,
        loss_mode=None,
        return_components=False,
    ):
        if loss_type is not None and int(loss_type) != 2:
            raise ValueError(
                "loss_type=1 is disabled after the gradient-difference refactor because "
                "it requires the full state gradient rather than D V."
            )
        components = self.loss_components(
            Y_n=Y_n,
            Y_n1=Y_n1,
            H_n=H_n,
            H_n1=H_n1,
            D_n=D_n,
            a_ref=a_ref,
            L_increment=L_increment,
            W_increment=W_increment,
            Q_n=Q_n,
            n=n,
            is_final=is_final,
            loss_mode=loss_mode,
        )
        return components if return_components else components["native_loss"]

    def consistency_loss(self, y_net, D_n, Y_n, create_graph, force=False):
        if not force and self.training_consistency_coef <= 0.0:
            return D_n.new_zeros(())

        grad_diff_H = self._physical_gradient_difference(
            y_net,
            Y_n,
            create_graph=create_graph,
        )
        return torch.mean((D_n - grad_diff_H) ** 2)
    
    def save_weights_as_txt(self, network, network_name, save_path_cppweights):
        # Iterate through each layer in the network
        for j, layer in enumerate(network.linearlayers):
            # Save weights
            if hasattr(layer, 'weight'):
                weight_filename = os.path.join(save_path_cppweights, f"{network_name}_w{j}.txt")
                np.savetxt(weight_filename, layer.weight.detach().cpu().numpy(), delimiter = ",")
            # Save biases if they are not None
            if hasattr(layer, 'bias') and layer.bias is not None:
                bias_filename = os.path.join(save_path_cppweights, f"{network_name}_b{j}.txt")
                np.savetxt(bias_filename, layer.bias.detach().cpu().numpy(), delimiter = ",")

    def _tensor_stats(self, tensor):
        values = tensor.detach().float().cpu().reshape(-1)
        if values.numel() == 0:
            return {"mean": np.nan, "std": np.nan, "norm": np.nan}
        return {
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item()),
            "norm": float(torch.linalg.vector_norm(values).item()),
        }

    def _parameter_statistics_rows(self, module_list, network_name):
        rows = []
        for interval_idx, module in enumerate(module_list):
            for layer_idx, layer in enumerate(module.linearlayers):
                if not isinstance(layer, nn.Linear):
                    continue
                weight_stats = self._tensor_stats(layer.weight)
                bias_stats = (
                    self._tensor_stats(layer.bias)
                    if layer.bias is not None
                    else {"mean": "", "std": "", "norm": ""}
                )
                rows.append(
                    {
                        "network": network_name,
                        "interval": int(interval_idx),
                        "layer": int(layer_idx),
                        "layer_role": (
                            "output" if layer_idx == len(module.linearlayers) - 1 else "hidden"
                        ),
                        "weight_shape": tuple(layer.weight.shape),
                        "weight_mean": weight_stats["mean"],
                        "weight_std": weight_stats["std"],
                        "weight_norm": weight_stats["norm"],
                        "bias_shape": tuple(layer.bias.shape) if layer.bias is not None else "",
                        "bias_mean": bias_stats["mean"],
                        "bias_std": bias_stats["std"],
                        "bias_norm": bias_stats["norm"],
                    }
                )
        return rows

    def _write_csv_rows(self, path, rows, fieldnames):
        if not rows:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})

    def _write_initial_parameter_statistics(self, run_dir, y_net_list, z_net_list, qgrad_net_list):
        rows = []
        rows.extend(self._parameter_statistics_rows(y_net_list, "H"))
        rows.extend(self._parameter_statistics_rows(z_net_list, "G"))
        if qgrad_net_list is not None:
            rows.extend(self._parameter_statistics_rows(qgrad_net_list, "QGrad"))
        fieldnames = [
            "network",
            "interval",
            "layer",
            "layer_role",
            "weight_shape",
            "weight_mean",
            "weight_std",
            "weight_norm",
            "bias_shape",
            "bias_mean",
            "bias_std",
            "bias_norm",
        ]
        path = os.path.join(run_dir, "initial_parameter_statistics.csv")
        self._write_csv_rows(path, rows, fieldnames)
        return path

    def _write_initial_checkpoints(self, run_dir, y_net_list, z_net_list, qgrad_net_list):
        checkpoint_dir = os.path.join(run_dir, "initial_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        if not self.save_initial_checkpoints:
            manifest = {
                "initial_checkpoint_dir": "initial_checkpoints",
                "num_intervals": int(len(y_net_list)),
                "written_counts": {"H": 0, "G": 0, "QGrad": 0},
                "overwrite_rule": "initial checkpoint files skipped by save_initial_checkpoints=false",
                "skipped": True,
            }
            manifest_path = os.path.join(checkpoint_dir, "manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=4, default=str)
            return checkpoint_dir, manifest

        written = {"H": 0, "G": 0, "QGrad": 0}

        def save_if_missing(module, filename, key):
            path = os.path.join(checkpoint_dir, filename)
            if os.path.exists(path):
                return path
            torch.save(module.state_dict(), path)
            written[key] += 1
            return path

        for interval_idx, module in enumerate(y_net_list):
            save_if_missing(module, f"y_network{interval_idx}.pth", "H")
        for interval_idx, module in enumerate(z_net_list):
            save_if_missing(module, f"z_network{interval_idx}.pth", "G")
        if qgrad_net_list is not None:
            for interval_idx, module in enumerate(qgrad_net_list):
                save_if_missing(module, f"qgrad_network{interval_idx}.pth", "QGrad")
        manifest = {
            "initial_checkpoint_dir": "initial_checkpoints",
            "num_intervals": int(len(y_net_list)),
            "written_counts": written,
            "overwrite_rule": "existing initial checkpoint files are preserved",
        }
        manifest_path = os.path.join(checkpoint_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=4, default=str)
        return checkpoint_dir, manifest

    def _reference_and_sample_hashes(self):
        rollout_pool = self.state_sampler.get_rollout_state_pool()
        rollout_metadata = self.state_sampler.get_rollout_metadata_pool()
        tensors = list(rollout_pool)
        labels = rollout_metadata.get("reference_policy_labels")
        path_ids = rollout_metadata.get("path_ids")
        if labels is not None:
            tensors.extend(labels)
        if path_ids is not None:
            tensors.extend(path_ids)
        reference_pool_hash = tensor_sequence_sha256(tensors)
        sample_bank_payload = {
            "sampling_mode": self.sampling_mode,
            "proposal_sampler": self.proposal_sampler,
            "rollout_pool_size": self.rollout_pool_size,
            "sampling_seed": self.sampling_seed,
            "stateless_seed_schedule": self.stateless_seed_schedule,
            "seed_schedule_experiment": self.seed_schedule_experiment,
            "seed_schedule_key": self.seed_schedule_key,
            "batch_size": self.batch_size,
            "valid_size": self.valid_size,
            "initial_state_mode": self.initial_state_mode,
            "initial_perturb_prob": self.initial_perturb_prob,
            "initial_q_perturb_frac": self.initial_q_perturb_frac,
            "initial_z_perturb_frac": self.initial_z_perturb_frac,
            "initial_state_summary": self.state_sampler.initial_state_summary,
            "sampler_generator_state_after_reference_pool": tensor_sha256(self.state_sampler.generator.get_state()),
        }
        eval_config = getattr(self.config, "eval_config", None)
        eval_payload = {
            "eval_config": vars(eval_config) if hasattr(eval_config, "__dict__") else eval_config,
            "primary_extraction": "E3",
        }
        return {
            "reference_pool_hash": reference_pool_hash,
            "sample_bank_hash": json_sha256(sample_bank_payload),
            "evaluation_bank_hash": json_sha256(eval_payload),
            "sample_bank_payload": sample_bank_payload,
            "evaluation_bank_payload": eval_payload,
        }

    def _scale_gamma_for_capped_proportional(self, mu, eta):
        """Solve sum min(gamma * mu, eta) dt <= I0 for the scaled-static prior."""
        mu_np = mu.detach().cpu().double().numpy()
        eta_np = eta.detach().cpu().double().numpy()
        capacity_at_one = float(np.minimum(mu_np, eta_np).sum() * self.delta_t)
        if capacity_at_one <= float(self.bsde.I0) + 1e-12:
            return 1.0
        lo = 0.0
        hi = 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            shipped = float(np.minimum(mid * mu_np, eta_np).sum() * self.delta_t)
            if shipped <= float(self.bsde.I0):
                lo = mid
            else:
                hi = mid
        return float(lo)

    def _inverse_prior_from_rates(self, intended_rates, *, source):
        device = self.device
        dtype = self.bsde.mu.dtype
        mu = self.bsde.mu.to(device=device, dtype=dtype)
        eta = self.bsde.eta_schedule.to(device=device, dtype=dtype)
        rates = torch.clamp(intended_rates.to(device=device, dtype=dtype), min=0.0)
        eps_a = float(self.g_prior_epsilon_a)
        eps_mu = 1e-8
        lower = float(self.g_prior_lower_bound)
        upper = float(self.g_prior_upper_bound)
        cap_tol = torch.maximum(
            torch.full_like(eta, 1e-8),
            1e-6 * torch.maximum(eta, torch.ones_like(eta)),
        )
        positive_mu = mu > eps_mu
        positive_eta = eta > eps_a
        zero = rates <= eps_a
        cap = positive_eta & (rates >= eta - cap_tol)
        interior = positive_mu & positive_eta & (~zero) & (~cap)

        prior = torch.full_like(rates, upper)
        threshold = torch.zeros_like(rates)
        threshold[positive_mu & positive_eta] = float(self.bsde.lambda_param) * torch.pow(
            mu[positive_mu & positive_eta] / torch.clamp(eta[positive_mu & positive_eta], min=eps_a),
            float(self.bsde.alpha),
        )
        if cap.any():
            prior[cap] = torch.clamp(
                threshold[cap] * (1.0 - float(self.g_prior_cap_offset)),
                min=lower,
                max=upper,
            )
        if interior.any():
            safe_rate = torch.clamp(rates[interior], min=eps_a)
            prior[interior] = float(self.bsde.lambda_param) * torch.pow(
                mu[interior] / safe_rate,
                float(self.bsde.alpha),
            )
        prior = torch.clamp(prior, min=lower, max=upper)
        regimes = torch.full_like(rates, 0, dtype=torch.int64)
        regimes[interior] = 1
        regimes[cap] = 2
        metadata = {
            "source": source,
            "epsilon_a": eps_a,
            "prior_lower_bound": lower,
            "prior_upper_bound": upper,
            "cap_offset_below_threshold": float(self.g_prior_cap_offset),
            "zero_fraction": float(zero.float().mean().item()),
            "interior_fraction": float(interior.float().mean().item()),
            "cap_fraction": float(cap.float().mean().item()),
            "prior_min": float(prior.min().item()),
            "prior_max": float(prior.max().item()),
            "intended_rate_min": float(rates.min().item()),
            "intended_rate_max": float(rates.max().item()),
        }
        return prior, regimes, threshold, metadata

    def _scaled_static_prior(self):
        mu = self.bsde.mu.to(device=self.device)
        eta = self.bsde.eta_schedule.to(device=self.device, dtype=mu.dtype)
        gamma = self._scale_gamma_for_capped_proportional(mu, eta)
        raw_rates = float(gamma) * mu
        intended_rates = torch.minimum(raw_rates, eta).clamp(min=0.0)
        prior, regimes, threshold, metadata = self._inverse_prior_from_rates(
            intended_rates,
            source="scaled_static_inverse",
        )
        cap_raw = raw_rates > eta + torch.maximum(
            torch.full_like(eta, 1e-8),
            1e-6 * torch.maximum(eta, torch.ones_like(eta)),
        )
        metadata.update(
            {
                "gamma": float(gamma),
                "gamma_rule": "bisection for sum_nk min(gamma * mu_nk, eta_nk) * dt <= I0",
                "raw_gamma_mu_cap_fraction": float(cap_raw.float().mean().item()),
                "total_intended_shipment": float((intended_rates.sum() * self.delta_t).item()),
                "I0": float(self.bsde.I0),
            }
        )
        return prior, intended_rates, regimes, threshold, metadata

    def _fluid_prior(self):
        if not self.g_prior_fluid_plan_path:
            raise ValueError("g_init_method='fluid_inverse' requires g_prior_fluid_plan_path.")
        plan_quantity, review_dt, resolved_path = load_open_loop_allocation_plan(
            self.g_prior_fluid_plan_path,
            self.dim,
            self.T,
            require_fairness_spec=float(self.bsde.lambda_param) > 0.0,
        )
        with np.load(resolved_path, allow_pickle=False) as data:
            def scalar(key, default=None):
                if key not in data:
                    if default is not None:
                        return default
                    raise ValueError(f"Fluid prior artifact {resolved_path} is missing {key!r}.")
                return np.asarray(data[key]).reshape(-1)[0]

            lambda_actual = float(scalar("lambda_param"))
            alpha_actual = float(scalar("alpha"))
            if not np.isclose(lambda_actual, float(self.bsde.lambda_param), rtol=0.0, atol=1e-12):
                raise ValueError(
                    f"Fluid prior lambda_param mismatch: artifact={lambda_actual}, "
                    f"run={float(self.bsde.lambda_param)}."
                )
            if not np.isclose(alpha_actual, float(self.bsde.alpha), rtol=0.0, atol=1e-12):
                raise ValueError(
                    f"Fluid prior alpha mismatch: artifact={alpha_actual}, run={float(self.bsde.alpha)}."
                )
            fairness_spec = str(scalar("fairness_spec"))
            if fairness_spec != FAIRNESS_SPEC:
                raise ValueError(
                    f"Fluid prior fairness_spec={fairness_spec!r}; expected {FAIRNESS_SPEC!r}."
                )
            if "chi_eta" in data and not np.isclose(float(scalar("chi_eta")), float(getattr(self.eqn_config, "chi_eta", 1.75)), atol=1e-12):
                raise ValueError("Fluid prior chi_eta metadata is incompatible with this run.")
            if "s_eta" in data and not np.isclose(float(scalar("s_eta")), float(getattr(self.eqn_config, "s_eta", 0.0)), atol=1e-12):
                raise ValueError("Fluid prior s_eta metadata is incompatible with this run.")

        review_rows = int(plan_quantity.shape[0])
        eta_review = self.bsde.eta_for_review_grid(review_rows).detach().cpu().numpy()
        cap_quantity = eta_review * float(review_dt)
        cap_tol = np.maximum(1e-5, 1e-6 * np.maximum(1.0, cap_quantity))
        if np.any(plan_quantity > cap_quantity + cap_tol):
            row, col = np.argwhere(plan_quantity > cap_quantity + cap_tol)[0]
            raise ValueError(
                "Fluid prior artifact violates active eta bound at "
                f"row={int(row)}, location={int(col)}."
            )

        rates = []
        for n in range(self.num_time_interval):
            current_time = float(n) * float(self.delta_t)
            review_idx = min(
                int(np.floor((current_time + 1e-12) / float(review_dt))),
                review_rows - 1,
            )
            rates.append(plan_quantity[review_idx] / float(review_dt))
        intended_rates = torch.tensor(np.stack(rates, axis=0), device=self.device, dtype=self.bsde.mu.dtype)
        prior, regimes, threshold, metadata = self._inverse_prior_from_rates(
            intended_rates,
            source="fluid_inverse",
        )
        metadata.update(
            {
                "fluid_plan_path": resolved_path,
                "fluid_plan_shape": list(plan_quantity.shape),
                "fluid_review_dt": float(review_dt),
                "fluid_total_shipment": float(plan_quantity.sum()),
                "I0": float(self.bsde.I0),
            }
        )
        return prior, intended_rates, regimes, threshold, metadata

    def _benchmark_g_prior(self):
        if self.g_init_method == "scaled_static_inverse":
            return self._scaled_static_prior()
        if self.g_init_method == "fluid_inverse":
            return self._fluid_prior()
        return None

    def _write_g_prior_rows(self, run_dir, prior, intended_rates, regimes, threshold):
        rows = []
        mu = self.bsde.mu.detach().cpu().numpy()
        eta = self.bsde.eta_schedule.detach().cpu().numpy()
        prior_np = prior.detach().cpu().numpy()
        intended_np = intended_rates.detach().cpu().numpy()
        regimes_np = regimes.detach().cpu().numpy()
        threshold_np = threshold.detach().cpu().numpy()
        regime_name = {0: "zero", 1: "interior", 2: "cap"}
        for n in range(prior_np.shape[0]):
            for k in range(prior_np.shape[1]):
                rows.append(
                    {
                        "interval": n,
                        "location": k,
                        "bias": float(prior_np[n, k]),
                        "intended_rate": float(intended_np[n, k]),
                        "mu": float(mu[n, k]),
                        "eta": float(eta[n, k]),
                        "cap_threshold": float(threshold_np[n, k]),
                        "regime": regime_name.get(int(regimes_np[n, k]), str(int(regimes_np[n, k]))),
                    }
                )
        path = os.path.join(run_dir, "initial_g_prior_bias.csv")
        self._write_csv_rows(
            path,
            rows,
            ["interval", "location", "bias", "intended_rate", "mu", "eta", "cap_threshold", "regime"],
        )
        return path

    def _apply_benchmark_g_initialization(self, z_net_list, run_dir):
        prior_payload = self._benchmark_g_prior()
        if prior_payload is None:
            return None
        prior, intended_rates, regimes, threshold, metadata = prior_payload
        with torch.no_grad():
            for interval_idx, g_net in enumerate(z_net_list):
                final_layer = g_net.linearlayers[-1]
                if final_layer.bias is None:
                    raise ValueError("Benchmark G initialization requires a final-layer bias.")
                final_layer.weight.zero_()
                final_layer.bias.copy_(prior[interval_idx].to(device=final_layer.bias.device, dtype=final_layer.bias.dtype))
        bias_path = self._write_g_prior_rows(run_dir, prior, intended_rates, regimes, threshold)
        metadata = dict(metadata)
        metadata.update(
            {
                "final_layer_weight_rule": "zero",
                "final_layer_weight_scale": float(self.g_benchmark_final_weight_scale),
                "final_layer_bias_file": os.path.basename(bias_path),
                "hidden_layer_init_method": "xavier_uniform",
                "benchmark_final_layer_only_change": True,
            }
        )
        self.g_benchmark_prior = {
            "prior": prior.detach().clone(),
            "intended_rates": intended_rates.detach().clone(),
            "regimes": regimes.detach().clone(),
            "metadata": metadata,
        }
        return metadata

    def _fixed_initial_policy_state_bank(self, dtype):
        q_values = torch.linspace(0.0, float(self.bsde.I0), steps=32, device=self.device, dtype=dtype)
        states = torch.zeros(q_values.numel(), self.state_dim, device=self.device, dtype=dtype)
        states[:, self.dim] = q_values
        return states

    def _action_regime_fractions(self, actions, eta):
        eta_b = eta.to(device=actions.device, dtype=actions.dtype).unsqueeze(0).expand_as(actions)
        denom = max(int(actions.numel()), 1)
        zero = actions <= 1e-8
        cap = actions >= eta_b - torch.maximum(
            torch.full_like(eta_b, 1e-8),
            1e-6 * torch.maximum(eta_b, torch.ones_like(eta_b)),
        )
        interior = (~zero) & (~cap)
        return {
            "zero_fraction": float(zero.float().sum().item() / denom),
            "interior_fraction": float(interior.float().sum().item() / denom),
            "cap_fraction": float(cap.float().sum().item() / denom),
        }

    def _deterministic_initial_rollout_summary(self, z_net_list, weeks=(1, 2, 4, 8)):
        max_week = max(weeks)
        fine_per_week = int(round(1.0 / float(self.delta_t)))
        max_interval = min(int(max_week * fine_per_week), self.num_time_interval)
        Z = torch.zeros(1, self.dim, device=self.device, dtype=self.bsde.mu.dtype)
        Q = torch.full((1,), float(self.bsde.I0), device=self.device, dtype=self.bsde.mu.dtype)
        cumulative = 0.0
        cumulative_hf = 0.0
        out = {}
        zero_dw = torch.zeros(1, self.dim, device=self.device, dtype=self.bsde.mu.dtype)
        for n in range(max_interval):
            state = torch.cat([Z, Q.reshape(1, 1)], dim=1)
            grad = self._g_physical_output(z_net_list[n], state)
            a = build_feasible_ds_allocation(
                grad,
                Q,
                self.bsde.mu[n],
                self.bsde.eta_at_fine_index(n).to(device=self.device, dtype=grad.dtype),
                self.bsde.lambda_param,
                self.bsde.alpha,
                self.delta_t,
            )
            quantity = a * float(self.delta_t)
            cumulative += float(quantity.sum().item())
            cumulative_hf += float(quantity[:, 0].sum().item())
            Z, Q, _ = step_transition_torch(
                Z,
                Q,
                a,
                zero_dw,
                self.delta_t,
                self.bsde.mu[n],
                self.bsde.p,
            )
            week = (n + 1) / fine_per_week
            if any(np.isclose(week, target) for target in weeks):
                key = int(round(week))
                out[f"week_{key}_cumulative_shipment"] = cumulative
                out[f"week_{key}_cumulative_hf_shipment"] = cumulative_hf
        return out

    def _write_initial_policy_sanity(self, run_dir, z_net_list):
        rows = []
        bank = self._fixed_initial_policy_state_bank(self.bsde.mu.dtype)
        benchmark = getattr(self, "g_benchmark_prior", None)
        weekly_totals = {}
        for n, g_net in enumerate(z_net_list):
            eta = self.bsde.eta_at_fine_index(n).to(device=self.device)
            with torch.no_grad():
                grad = self._g_physical_output(g_net, bank)
                actions = build_feasible_ds_allocation(
                    grad,
                    bank[:, self.dim],
                    self.bsde.mu[n],
                    eta.to(dtype=grad.dtype),
                    self.bsde.lambda_param,
                    self.bsde.alpha,
                    self.delta_t,
                )
                raw_actions = build_unprojected_ds_allocation(
                    grad,
                    bank[:, self.dim],
                    self.bsde.mu[n],
                    eta.to(dtype=grad.dtype),
                    self.bsde.lambda_param,
                    self.bsde.alpha,
                    self.delta_t,
                )
            fractions = self._action_regime_fractions(actions, eta)
            mean_rate = actions.mean(dim=0)
            mean_quantity = mean_rate * float(self.delta_t)
            week_idx = int(np.floor(n * float(self.delta_t))) + 1
            weekly_totals.setdefault(week_idx, 0.0)
            weekly_totals[week_idx] += float(mean_quantity.sum().item())
            intended_rmse = ""
            intended_max_abs = ""
            intended_regime_match = ""
            if benchmark is not None:
                intended = benchmark["intended_rates"][n].to(device=actions.device, dtype=actions.dtype)
                full_q_state = torch.zeros(1, self.state_dim, device=self.device, dtype=actions.dtype)
                full_q_state[:, self.dim] = float(self.bsde.I0)
                with torch.no_grad():
                    full_grad = self._g_physical_output(g_net, full_q_state)
                    full_action = build_feasible_ds_allocation(
                        full_grad,
                        full_q_state[:, self.dim],
                        self.bsde.mu[n],
                        eta.to(dtype=full_grad.dtype),
                        self.bsde.lambda_param,
                        self.bsde.alpha,
                        self.delta_t,
                    ).squeeze(0)
                diff = full_action - intended
                intended_rmse = float(torch.sqrt(torch.mean(diff ** 2)).item())
                intended_max_abs = float(torch.max(torch.abs(diff)).item())
                intended_regime = benchmark["regimes"][n].to(device=actions.device)
                actual_regime = torch.zeros_like(intended_regime)
                cap = full_action >= eta - torch.maximum(
                    torch.full_like(eta, 1e-8),
                    1e-6 * torch.maximum(eta, torch.ones_like(eta)),
                )
                interior = (full_action > 1e-8) & (~cap)
                actual_regime[interior] = 1
                actual_regime[cap] = 2
                intended_regime_match = float((actual_regime == intended_regime).float().mean().item())
            rows.append(
                {
                    "interval": n,
                    "week": week_idx,
                    "mean_tentative_shipment": float(mean_quantity.sum().item()),
                    "mean_hf_shipment": float(mean_quantity[0].item()),
                    "hf_share": float(mean_quantity[0].item() / max(float(mean_quantity.sum().item()), 1e-12)),
                    "raw_action_norm_mean": float(torch.linalg.vector_norm(raw_actions, dim=1).mean().item()),
                    **fractions,
                    "benchmark_action_rmse": intended_rmse,
                    "benchmark_action_max_abs_diff": intended_max_abs,
                    "benchmark_regime_match_fraction": intended_regime_match,
                }
            )
        rollout_summary = self._deterministic_initial_rollout_summary(z_net_list)
        for row in rows:
            row["initial_weekly_tentative_shipment"] = weekly_totals.get(row["week"], "")
            row.update(rollout_summary)
        fieldnames = [
            "interval",
            "week",
            "mean_tentative_shipment",
            "mean_hf_shipment",
            "hf_share",
            "raw_action_norm_mean",
            "zero_fraction",
            "interior_fraction",
            "cap_fraction",
            "initial_weekly_tentative_shipment",
            "week_1_cumulative_shipment",
            "week_1_cumulative_hf_shipment",
            "week_2_cumulative_shipment",
            "week_2_cumulative_hf_shipment",
            "week_4_cumulative_shipment",
            "week_4_cumulative_hf_shipment",
            "week_8_cumulative_shipment",
            "week_8_cumulative_hf_shipment",
            "benchmark_action_rmse",
            "benchmark_action_max_abs_diff",
            "benchmark_regime_match_fraction",
        ]
        path = os.path.join(run_dir, "initial_policy_sanity.csv")
        self._write_csv_rows(path, rows, fieldnames)
        return path
    
    def train(
        self,
        final_iterations,
        inner_iterations,
        save_path_logs,
        save_path_cppweights,
        resume_from_dir=None,
        training_history_path=None,
        interval_summary_path=None,
        state_support_summary_path=None,
        train_final_interval_only=False,
        init_checkpoint_dir=None,
        init_metadata=None,
    ):
        
        # Step 1: Initialize neural networks for all intervals.
        # Y_Net maps Y=(Z, I) to the scalar value. The second network maps Y to D V.
        run_dir_for_metadata = os.path.dirname(os.path.abspath(save_path_logs.rstrip(os.sep)))

        def _construct_component(seed, factory):
            if not self.component_seed_isolation:
                return factory()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed))
                return factory()

        def _build_h_networks():
            return nn.ModuleList([
                Y_Net(
                    self.state_dim,
                    self.num_neurons,
                    self.num_layers,
                    self.activation,
                    self.slope,
                    self.h_init_method,
                )
                for _ in range(self.num_time_interval)
            ])

        def _build_g_networks():
            return nn.ModuleList([
                GradDiffNet(
                    self.state_dim,
                    self.grad_diff_dim,
                    self.num_neurons,
                    self.num_layers,
                    self.activation,
                    self.slope,
                    self.g_constructor_init_method,
                    output_bias=self.g_output_bias,
                )
                for _ in range(self.num_time_interval)
            ])

        y_net_list = _construct_component(
            self.h_network_seed,
            _build_h_networks,
        ).to(self.device) # Network for value function approximation for each interval
        z_net_list = _construct_component(
            self.g_network_seed,
            _build_g_networks,
        ).to(self.device)  # Network for control-relevant gradient differences for each interval
        qgrad_net_list = None
        if self.requires_qgrad:
            def _build_qgrad_networks():
                return nn.ModuleList([
                    QGradNet(
                        self.state_dim,
                        self.num_neurons,
                        self.num_layers,
                        self.activation,
                        self.slope,
                        self.g_constructor_init_method,
                    )
                    for _ in range(self.num_time_interval)
                ])

            qgrad_net_list = _construct_component(
                self.qgrad_network_seed,
                _build_qgrad_networks,
            ).to(self.device)
        self._apply_scratch_reparameterization(y_net_list, z_net_list, qgrad_net_list)
        benchmark_init_metadata = self._apply_benchmark_g_initialization(
            z_net_list,
            run_dir_for_metadata,
        )
        initial_parameter_statistics_path = self._write_initial_parameter_statistics(
            run_dir_for_metadata,
            y_net_list,
            z_net_list,
            qgrad_net_list,
        )
        initial_checkpoint_dir, initial_checkpoint_manifest = self._write_initial_checkpoints(
            run_dir_for_metadata,
            y_net_list,
            z_net_list,
            qgrad_net_list,
        )
        initial_policy_sanity_path = self._write_initial_policy_sanity(
            run_dir_for_metadata,
            z_net_list,
        )
        bank_hashes = self._reference_and_sample_hashes()
        initial_network_metadata = {
            "loss_mode": self.loss_mode,
            "train_seed": getattr(self.net_config, "train_seed", None),
            "sampling_seed": self.sampling_seed,
            "stateless_seed_schedule": self.stateless_seed_schedule,
            "seed_schedule_experiment": self.seed_schedule_experiment,
            "seed_schedule_key": self.seed_schedule_key,
            "eval_seed": getattr(getattr(self.config, "eval_config", None), "eval_seed", None),
            "legacy_init_method": self.init_method,
            "h_init_method": self.h_init_method,
            "g_init_method": self.g_init_method,
            "g_constructor_init_method": self.g_constructor_init_method,
            "g_output_bias": self.g_output_bias,
            "g_benchmark_final_weight_scale": self.g_benchmark_final_weight_scale,
            "component_seed_isolation": self.component_seed_isolation,
            "h_network_seed": self.h_network_seed if self.component_seed_isolation else "",
            "g_network_seed": self.g_network_seed if self.component_seed_isolation else "",
            "qgrad_network_seed": self.qgrad_network_seed if self.component_seed_isolation else "",
            "save_initial_checkpoints": self.save_initial_checkpoints,
            "init_checkpoint_train_subset_mode": self.init_checkpoint_train_subset_mode,
            "h_init_metadata": initialization_metadata(self.h_init_method),
            "g_constructor_init_metadata": initialization_metadata(self.g_constructor_init_method),
            "g_init_metadata": benchmark_init_metadata or initialization_metadata(self.g_init_method),
            "g_benchmark_initialized": self.g_is_benchmark_initialized,
            "initial_H_hash": module_list_state_sha256(y_net_list),
            "initial_G_hash": module_list_state_sha256(z_net_list),
            "initial_QGrad_hash": (
                module_list_state_sha256(qgrad_net_list)
                if qgrad_net_list is not None
                else ""
            ),
            "reference_pool_hash": bank_hashes["reference_pool_hash"],
            "sample_bank_hash": bank_hashes["sample_bank_hash"],
            "evaluation_bank_hash": bank_hashes["evaluation_bank_hash"],
            "sample_bank_payload": bank_hashes["sample_bank_payload"],
            "evaluation_bank_payload": bank_hashes["evaluation_bank_payload"],
            "initial_parameter_statistics_file": os.path.basename(initial_parameter_statistics_path),
            "initial_checkpoint_dir": os.path.basename(initial_checkpoint_dir),
            "initial_checkpoint_manifest": initial_checkpoint_manifest,
            "initial_policy_sanity_file": os.path.basename(initial_policy_sanity_path),
            "initialization_fairness_scheme": (
                "H module list is initialized before G. H uses h_init_method and G uses "
                "g_init_method/g_constructor_init_method, so changing G initialization cannot "
                "alter H initial weights. Benchmark priors are deterministic tensor writes "
                "after scratch construction and consume no RNG."
            ),
        }
        self.initial_network_metadata = dict(initial_network_metadata)
        try:
            with open(os.path.join(run_dir_for_metadata, "network_initialization_metadata.json"), "w") as f:
                json.dump(initial_network_metadata, f, indent=4, default=str)
        except Exception as exc:
            print(f"Warning: failed to write network initialization metadata: {exc}")

        training_history = []
        history_len_at_last_write = 0  # for incremental CSV write
        start_time = time.time()
        timing_totals = {
            "validation_seconds": 0.0,
            "checkpoint_write_seconds": 0.0,
        }
        self.last_training_timing = dict(timing_totals)
        self.valid_size = self.net_config.valid_size
        self.batch_size = self.net_config.batch_size
        interval_summary_columns = [
            "interval",
            "loss_mode",
            "validation_every",
            "patience_iterations",
            "patience_checks",
            "budget_type",
            "iterations_budget",
            "learning_rate",
            "milestones",
            "scheduler_gamma",
            "minimum_iterations",
            "early_stopping_enabled",
            "best_iter",
            "best_learning_rate",
            "best_train_loss",
            "best_val_loss",
            "best_train_residual_loss",
            "best_val_residual_loss",
            "best_train_ce_residual_loss",
            "best_val_ce_residual_loss",
            "best_train_martingale_loss",
            "best_val_martingale_loss",
            "best_train_martingale_term_mean",
            "best_val_martingale_term_mean",
            "best_train_residual_martingale_cov",
            "best_val_residual_martingale_cov",
            "best_train_qgrad_mean",
            "best_val_qgrad_mean",
            "best_train_consistency_loss",
            "best_val_consistency_loss",
            "best_train_weighted_consistency_loss",
            "best_val_weighted_consistency_loss",
            "best_train_weighted_consistency_to_residual_ratio",
            "best_val_weighted_consistency_to_residual_ratio",
            "final_train_loss",
            "final_val_loss",
            "final_train_residual_loss",
            "final_val_residual_loss",
            "final_train_ce_residual_loss",
            "final_val_ce_residual_loss",
            "final_train_martingale_loss",
            "final_val_martingale_loss",
            "final_train_martingale_term_mean",
            "final_val_martingale_term_mean",
            "final_train_residual_martingale_cov",
            "final_val_residual_martingale_cov",
            "final_train_qgrad_mean",
            "final_val_qgrad_mean",
            "final_train_consistency_loss",
            "final_val_consistency_loss",
            "final_train_weighted_consistency_loss",
            "final_val_weighted_consistency_loss",
            "final_train_weighted_consistency_to_residual_ratio",
            "final_val_weighted_consistency_to_residual_ratio",
            "final_grad_norm",
            "grad_norm_mean",
            "grad_norm_max",
            "final_grad_clipped",
            "grad_clipping_frequency",
            "final_loss_finite",
            "nan_or_inf_indicator",
            "iterations_completed",
            "stop_iter",
            "early_stopped",
            "restored_best",
            "elapsed_time",
            "consistency_coef",
            "training_consistency_coef",
            "initial_H_hash",
            "initial_G_hash",
            "initial_QGrad_hash",
            "reference_pool_hash",
            "sample_bank_hash",
            "evaluation_bank_hash",
            "reuse_prev_init",
            "init_reason",
            "initialization_mode",
            "initialization_source",
            "source_interval",
            "source_block_index",
            "source_checkpoint_hash",
            "block_entry_interval",
            "source_lambda",
            "source_checkpoint_dir",
            "source_y_checkpoint",
            "source_z_checkpoint",
            "source_qgrad_checkpoint",
            "mu_max_abs_diff",
            "gamma_max_abs_diff",
            "eta_max_abs_diff",
        ]

        def _diagnostic_metadata():
            repo_root = os.path.dirname(os.path.abspath(__file__))
            return {
                "lambda_param": getattr(self.eqn_config, "lambda_param", None),
                "seed": getattr(self.net_config, "train_seed", None),
                "sampling_seed": getattr(self.net_config, "sampling_seed", None),
                "chi_eta": getattr(self.eqn_config, "chi_eta", None),
                "s_eta": getattr(self.eqn_config, "s_eta", None),
                "M": self.num_time_interval,
                "loss_mode": self.loss_mode,
                "validation_every": self.validation_every,
                "patience_iterations": self.patience_iterations,
                "patience_checks": self.patience_checks,
                "final_learning_rate": self.final_learning_rate,
                "inner_learning_rate": self.inner_learning_rate,
                "final_milestones": list(self.final_milestones),
                "inner_milestones": list(self.inner_milestones),
                "final_gamma": self.final_gamma,
                "inner_gamma": self.inner_gamma,
                "grad_clip": self.grad_clip,
                "grad_norm": self.grad_norm,
                "grad_clip_rule": self.grad_clip_rule,
                "final_grad_clip_rule": self.final_grad_clip_rule,
                "inner_grad_clip_rule": self.inner_grad_clip_rule,
                "h_grad_norm": self.h_grad_norm,
                "g_grad_norm": self.g_grad_norm,
                "clip_rule": (
                    "torch.nn.utils.clip_grad_norm_ over joint H+G parameters"
                    if self.grad_clip_rule == "joint"
                    else "torch.nn.utils.clip_grad_norm_ applied separately to H and G"
                ),
                "state_input_normalization": getattr(self.net_config, "state_normalization", "none"),
                "reparameterization": self.reparameterization_metadata(),
                "target_scaling": getattr(self.net_config, "target_scaling", "none"),
                "loss_reduction": {
                    "native_residual": "mean over batch of squared native residual for the selected loss_mode",
                    "ce_residual": "mean over batch of squared conditional-expectation residual",
                    "martingale_residual": "mean over batch of squared CE-minus-grad_z-dot-W residual when loss_mode=martingale_qgrad",
                    "consistency": "mean over batch and action dimensions of squared H/G gradient-difference mismatch",
                },
                "optimizer": "Adam, one optimizer per interval over joint H+G(+QGrad when enabled) parameters",
                "optimizer_count_per_interval": 1,
                "gradient_assembly": {
                    "mode": self.gradient_assembly_mode,
                    "use_component_gradient_assembly": self.use_component_gradient_assembly,
                    "h_base_grad_multiplier": self.h_base_grad_multiplier,
                    "g_base_grad_multiplier": self.g_base_grad_multiplier,
                    "h_consistency_coef": self.h_consistency_coef,
                    "g_consistency_coef": self.g_consistency_coef,
                    "baseline_consistency_coef": self.consistency_coef,
                    "training_consistency_coef": self.training_consistency_coef,
                },
                "mixed_precision": False,
                "gradient_accumulation_steps": 1,
                "scheduler": "MultiStepLR stepped once after each optimizer.step",
                "init_checkpoint_train_subset_mode": self.init_checkpoint_train_subset_mode,
                "git_commit": git_commit(repo_root),
            }

        gradient_diagnostics_enabled = bool(
            getattr(self.net_config, "gradient_diagnostics", False)
        )
        gradient_diagnostics_interval_subset = parse_interval_subset(
            getattr(self.net_config, "gradient_diagnostics_interval_subset", None)
        )
        diag_recorder = GradientDiagnosticsRecorder(
            enabled=gradient_diagnostics_enabled,
            output_dir=getattr(self.net_config, "gradient_diagnostics_output_dir", None),
            interval_subset=gradient_diagnostics_interval_subset,
            every=int(getattr(self.net_config, "gradient_diagnostics_every", 100)),
            component_every=int(
                getattr(
                    self.net_config,
                    "gradient_diagnostics_component_every",
                    getattr(self.net_config, "gradient_diagnostics_every", 100),
                )
            ),
            max_records_per_interval=getattr(
                self.net_config,
                "gradient_diagnostics_max_records_per_interval",
                None,
            ),
            metadata=_diagnostic_metadata(),
            append=bool(getattr(self.net_config, "gradient_diagnostics_append", False)),
        )
        if gradient_diagnostics_enabled:
            print(
                "gradient diagnostics enabled:",
                {
                    "output_dir": getattr(self.net_config, "gradient_diagnostics_output_dir", None),
                    "interval_subset": gradient_diagnostics_interval_subset,
                    "every": getattr(self.net_config, "gradient_diagnostics_every", 100),
                    "component_every": getattr(
                        self.net_config,
                        "gradient_diagnostics_component_every",
                        getattr(self.net_config, "gradient_diagnostics_every", 100),
                    ),
                },
            )
        if self.train_interval_subset is not None:
            print(
                "diagnostic-only train interval subset enabled:",
                {
                    "intervals": self.train_interval_subset,
                    "init_checkpoint_train_subset_mode": self.init_checkpoint_train_subset_mode,
                    "note": (
                        "Only these intervals will be optimized. This is for gradient-scale "
                        "audit smoke runs, not a complete deployable policy chain."
                    ),
                },
            )
        train_interval_subset_set = (
            None if self.train_interval_subset is None else set(self.train_interval_subset)
        )

        def _should_train_interval(interval_idx):
            if train_interval_subset_set is None:
                return True
            return int(interval_idx) in train_interval_subset_set

        def _record_history(
            interval_idx,
            iter_idx,
            loss_train,
            loss_valid,
            elapsed_time,
            best_val_loss="",
            best_iter="",
            final_val_loss="",
            stop_iter="",
            early_stopped="",
            restored_best="",
            train_residual_loss="",
            val_residual_loss="",
            train_consistency_loss="",
            val_consistency_loss="",
            train_weighted_consistency_loss="",
            val_weighted_consistency_loss="",
            train_weighted_consistency_to_residual_ratio="",
            val_weighted_consistency_to_residual_ratio="",
            grad_norm="",
            grad_clipped="",
            loss_finite="",
            learning_rate="",
            loss_mode="",
            validation_checked="",
            validation_check_index="",
            train_ce_residual_loss="",
            val_ce_residual_loss="",
            train_martingale_loss="",
            val_martingale_loss="",
            train_qgrad_mean="",
            val_qgrad_mean="",
            **_extra,
        ):
            row = [
                interval_idx,
                iter_idx,
                loss_train,
                loss_valid,
                elapsed_time,
                best_val_loss,
                best_iter,
                final_val_loss,
                stop_iter,
                early_stopped,
                restored_best,
                train_residual_loss,
                val_residual_loss,
                train_consistency_loss,
                val_consistency_loss,
                train_weighted_consistency_loss,
                val_weighted_consistency_loss,
                train_weighted_consistency_to_residual_ratio,
                val_weighted_consistency_to_residual_ratio,
                grad_norm,
                grad_clipped,
                loss_finite,
                loss_mode,
                validation_checked,
                validation_check_index,
                train_ce_residual_loss,
                val_ce_residual_loss,
                train_martingale_loss,
                val_martingale_loss,
                train_qgrad_mean,
                val_qgrad_mean,
            ]
            if training_history and training_history[-1][0] == interval_idx and training_history[-1][1] == iter_idx:
                training_history[-1] = row
            else:
                training_history.append(row)

        def _append_history_to_csv():
            nonlocal history_len_at_last_write
            if training_history_path is None or not training_history:
                return
            new_rows = training_history[history_len_at_last_write:]
            if not new_rows:
                return
            with open(training_history_path, "a") as f:
                for row in new_rows:
                    f.write(",".join(str(x) for x in row) + "\n")
            history_len_at_last_write = len(training_history)

        def _append_interval_summary_to_csv(row):
            if interval_summary_path is None:
                return
            write_header = not os.path.isfile(interval_summary_path) or os.path.getsize(interval_summary_path) == 0
            with open(interval_summary_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=interval_summary_columns)
                if write_header:
                    writer.writeheader()
                writer.writerow({column: row.get(column, "") for column in interval_summary_columns})

        def _record_interval_summary(
            interval_idx,
            schedule,
            init_decision,
            best_iter,
            best_train_loss,
            best_val_loss,
            final_train_loss,
            final_val_loss,
            stop_iter,
            early_stopped,
            restored_best,
            elapsed_time,
            best_loss_components=None,
            final_loss_components=None,
        ):
            init_decision = init_decision or {}
            best_loss_components = best_loss_components or {}
            final_loss_components = final_loss_components or {}
            row = {
                "interval": interval_idx,
                "loss_mode": self.loss_mode,
                "validation_every": self.validation_every,
                "patience_iterations": self.patience_iterations,
                "patience_checks": self.patience_checks,
                "budget_type": schedule.get("budget_type", ""),
                "iterations_budget": schedule.get("iterations", ""),
                "learning_rate": schedule.get("learning_rate", ""),
                "milestones": schedule.get("milestones", ""),
                "scheduler_gamma": schedule.get("gamma", ""),
                "minimum_iterations": schedule.get("minimum_iterations", ""),
                "early_stopping_enabled": schedule.get("early_stopping", ""),
                "best_iter": best_iter,
                "best_learning_rate": best_loss_components.get("learning_rate", ""),
                "best_train_loss": best_train_loss,
                "best_val_loss": best_val_loss,
                "best_train_residual_loss": best_loss_components.get("train_residual_loss", ""),
                "best_val_residual_loss": best_loss_components.get("val_residual_loss", ""),
                "best_train_ce_residual_loss": best_loss_components.get("train_ce_residual_loss", ""),
                "best_val_ce_residual_loss": best_loss_components.get("val_ce_residual_loss", ""),
                "best_train_martingale_loss": best_loss_components.get("train_martingale_loss", ""),
                "best_val_martingale_loss": best_loss_components.get("val_martingale_loss", ""),
                "best_train_martingale_term_mean": best_loss_components.get("train_martingale_term_mean", ""),
                "best_val_martingale_term_mean": best_loss_components.get("val_martingale_term_mean", ""),
                "best_train_residual_martingale_cov": best_loss_components.get("train_residual_martingale_cov", ""),
                "best_val_residual_martingale_cov": best_loss_components.get("val_residual_martingale_cov", ""),
                "best_train_qgrad_mean": best_loss_components.get("train_qgrad_mean", ""),
                "best_val_qgrad_mean": best_loss_components.get("val_qgrad_mean", ""),
                "best_train_consistency_loss": best_loss_components.get("train_consistency_loss", ""),
                "best_val_consistency_loss": best_loss_components.get("val_consistency_loss", ""),
                "best_train_weighted_consistency_loss": best_loss_components.get("train_weighted_consistency_loss", ""),
                "best_val_weighted_consistency_loss": best_loss_components.get("val_weighted_consistency_loss", ""),
                "best_train_weighted_consistency_to_residual_ratio": best_loss_components.get("train_weighted_consistency_to_residual_ratio", ""),
                "best_val_weighted_consistency_to_residual_ratio": best_loss_components.get("val_weighted_consistency_to_residual_ratio", ""),
                "final_train_loss": final_train_loss,
                "final_val_loss": final_val_loss,
                "final_train_residual_loss": final_loss_components.get("train_residual_loss", ""),
                "final_val_residual_loss": final_loss_components.get("val_residual_loss", ""),
                "final_train_ce_residual_loss": final_loss_components.get("train_ce_residual_loss", ""),
                "final_val_ce_residual_loss": final_loss_components.get("val_ce_residual_loss", ""),
                "final_train_martingale_loss": final_loss_components.get("train_martingale_loss", ""),
                "final_val_martingale_loss": final_loss_components.get("val_martingale_loss", ""),
                "final_train_martingale_term_mean": final_loss_components.get("train_martingale_term_mean", ""),
                "final_val_martingale_term_mean": final_loss_components.get("val_martingale_term_mean", ""),
                "final_train_residual_martingale_cov": final_loss_components.get("train_residual_martingale_cov", ""),
                "final_val_residual_martingale_cov": final_loss_components.get("val_residual_martingale_cov", ""),
                "final_train_qgrad_mean": final_loss_components.get("train_qgrad_mean", ""),
                "final_val_qgrad_mean": final_loss_components.get("val_qgrad_mean", ""),
                "final_train_consistency_loss": final_loss_components.get("train_consistency_loss", ""),
                "final_val_consistency_loss": final_loss_components.get("val_consistency_loss", ""),
                "final_train_weighted_consistency_loss": final_loss_components.get("train_weighted_consistency_loss", ""),
                "final_val_weighted_consistency_loss": final_loss_components.get("val_weighted_consistency_loss", ""),
                "final_train_weighted_consistency_to_residual_ratio": final_loss_components.get("train_weighted_consistency_to_residual_ratio", ""),
                "final_val_weighted_consistency_to_residual_ratio": final_loss_components.get("val_weighted_consistency_to_residual_ratio", ""),
                "final_grad_norm": final_loss_components.get("grad_norm", ""),
                "grad_norm_mean": final_loss_components.get("grad_norm_mean", ""),
                "grad_norm_max": final_loss_components.get("grad_norm_max", ""),
                "final_grad_clipped": final_loss_components.get("grad_clipped", ""),
                "grad_clipping_frequency": final_loss_components.get("grad_clipping_frequency", ""),
                "final_loss_finite": final_loss_components.get("loss_finite", ""),
                "nan_or_inf_indicator": final_loss_components.get("nan_or_inf_indicator", ""),
                "iterations_completed": (int(stop_iter) + 1) if stop_iter is not None else "",
                "stop_iter": stop_iter,
                "early_stopped": early_stopped,
                "restored_best": restored_best,
                "elapsed_time": elapsed_time,
                "consistency_coef": self.consistency_coef,
                "training_consistency_coef": self.training_consistency_coef,
                "initial_H_hash": initial_network_metadata.get("initial_H_hash", ""),
                "initial_G_hash": initial_network_metadata.get("initial_G_hash", ""),
                "initial_QGrad_hash": initial_network_metadata.get("initial_QGrad_hash", ""),
                "reference_pool_hash": initial_network_metadata.get("reference_pool_hash", ""),
                "sample_bank_hash": initial_network_metadata.get("sample_bank_hash", ""),
                "evaluation_bank_hash": initial_network_metadata.get("evaluation_bank_hash", ""),
                "reuse_prev_init": init_decision.get("reuse_prev_init", ""),
                "init_reason": init_decision.get("reason", ""),
                "initialization_mode": init_decision.get("initialization_mode", init_decision.get("reason", "")),
                "initialization_source": init_decision.get("initialization_source", ""),
                "source_interval": init_decision.get("source_interval", ""),
                "source_block_index": init_decision.get("source_block_index", ""),
                "source_checkpoint_hash": init_decision.get("source_checkpoint_hash", ""),
                "block_entry_interval": init_decision.get("block_entry_interval", ""),
                "source_lambda": init_decision.get("source_lambda", ""),
                "source_checkpoint_dir": init_decision.get("source_checkpoint_dir", ""),
                "source_y_checkpoint": init_decision.get("source_y_checkpoint", ""),
                "source_z_checkpoint": init_decision.get("source_z_checkpoint", ""),
                "source_qgrad_checkpoint": init_decision.get("source_qgrad_checkpoint", ""),
                "mu_max_abs_diff": init_decision.get("mu_max_abs_diff", ""),
                "gamma_max_abs_diff": init_decision.get("gamma_max_abs_diff", ""),
                "eta_max_abs_diff": init_decision.get("eta_max_abs_diff", ""),
            }
            _append_interval_summary_to_csv(row)

        def _clone_state_dict(module):
            return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}

        def _restore_best_checkpoint(interval_idx, best_y_state, best_z_state, best_qgrad_state=None):
            if best_y_state is None or best_z_state is None:
                return False
            if self.requires_qgrad and best_qgrad_state is None:
                return False
            y_net_list[interval_idx].load_state_dict(best_y_state)
            z_net_list[interval_idx].load_state_dict(best_z_state)
            if self.requires_qgrad:
                qgrad_net_list[interval_idx].load_state_dict(best_qgrad_state)
            return True

        def _atomic_torch_save(state_dict, path):
            tmp_path = f"{path}.tmp.{os.getpid()}"
            checkpoint_start = time.time()
            try:
                torch.save(state_dict, tmp_path)
                os.replace(tmp_path, path)
            finally:
                timing_totals["checkpoint_write_seconds"] += time.time() - checkpoint_start
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

        def _save_interval_checkpoint(interval_idx):
            _atomic_torch_save(
                y_net_list[interval_idx].state_dict(),
                save_path_logs + f"y_network{interval_idx}.pth",
            )
            _atomic_torch_save(
                z_net_list[interval_idx].state_dict(),
                save_path_logs + f"z_network{interval_idx}.pth",
            )
            if self.requires_qgrad:
                _atomic_torch_save(
                    qgrad_net_list[interval_idx].state_dict(),
                    save_path_logs + f"qgrad_network{interval_idx}.pth",
                )

        def _print_interval_summary(
            interval_idx,
            best_val_loss,
            best_iter,
            final_val_loss,
            stop_iter,
            early_stopped,
            restored_best,
            best_loss_components=None,
            final_loss_components=None,
        ):
            best_loss_components = best_loss_components or {}
            final_loss_components = final_loss_components or {}
            print(
                "interval summary:",
                {
                    "interval": interval_idx,
                    "loss_mode": self.loss_mode,
                    "best_val_loss": best_val_loss,
                    "best_val_residual_loss": best_loss_components.get("val_residual_loss", ""),
                    "best_val_ce_residual_loss": best_loss_components.get("val_ce_residual_loss", ""),
                    "best_val_martingale_loss": best_loss_components.get("val_martingale_loss", ""),
                    "best_val_consistency_loss": best_loss_components.get("val_consistency_loss", ""),
                    "best_val_weighted_consistency_loss": best_loss_components.get("val_weighted_consistency_loss", ""),
                    "best_iter": best_iter,
                    "final_val_loss": final_val_loss,
                    "final_val_residual_loss": final_loss_components.get("val_residual_loss", ""),
                    "final_val_ce_residual_loss": final_loss_components.get("val_ce_residual_loss", ""),
                    "final_val_martingale_loss": final_loss_components.get("val_martingale_loss", ""),
                    "final_val_consistency_loss": final_loss_components.get("val_consistency_loss", ""),
                    "final_val_weighted_consistency_loss": final_loss_components.get("val_weighted_consistency_loss", ""),
                    "stop_iter": stop_iter,
                    "early_stopped": early_stopped,
                    "restored_best": restored_best,
                },
            )

        def _loss_value(loss_tensor):
            return float(loss_tensor.detach().item())

        def _safe_ratio(numerator, denominator):
            denominator = float(denominator)
            if not np.isfinite(denominator) or abs(denominator) <= 1e-12:
                return np.nan
            return float(numerator) / denominator

        def _apply_optimizer_step(
            interval_idx,
            loss_tensor,
            optimizer,
            *,
            phase,
            iteration_idx,
            residual_loss_tensor,
            consistency_loss_tensor,
            schedule,
        ):
            optimizer.zero_grad()
            y_params = list(y_net_list[interval_idx].parameters())
            z_params = list(z_net_list[interval_idx].parameters())
            q_params = (
                list(qgrad_net_list[interval_idx].parameters())
                if qgrad_net_list is not None
                else []
            )
            params = y_params + z_params + q_params
            learning_rate_before_step = optimizer.param_groups[0]["lr"]
            diagnostic_step = diag_recorder.prepare_step(
                interval_idx=interval_idx,
                iteration_idx=iteration_idx,
                phase=phase,
                budget_type=schedule.get("budget_type", phase),
                learning_rate=learning_rate_before_step,
                optimizer_param_group_count=len(optimizer.param_groups),
                h_params=y_params,
                g_params=z_params + q_params,
                residual_loss=residual_loss_tensor,
                consistency_loss=consistency_loss_tensor,
                total_loss=loss_tensor,
                consistency_coef=self.training_consistency_coef,
            )

            if self.use_component_gradient_assembly:
                base_grads = torch.autograd.grad(
                    residual_loss_tensor,
                    params,
                    retain_graph=True,
                    allow_unused=True,
                )
                if (
                    isinstance(consistency_loss_tensor, torch.Tensor)
                    and consistency_loss_tensor.requires_grad
                    and (
                        self.h_consistency_coef != 0.0
                        or self.g_consistency_coef != 0.0
                    )
                ):
                    consistency_grads = torch.autograd.grad(
                        consistency_loss_tensor,
                        params,
                        retain_graph=False,
                        allow_unused=True,
                    )
                else:
                    consistency_grads = tuple(None for _ in params)

                assemble_hg_component_gradients(
                    params,
                    len(y_params),
                    base_grads,
                    consistency_grads,
                    h_base_multiplier=self.h_base_grad_multiplier,
                    g_base_multiplier=self.g_base_grad_multiplier,
                    h_consistency_coef=self.h_consistency_coef,
                    g_consistency_coef=self.g_consistency_coef,
                )
            else:
                loss_tensor.backward()
            pre_clip_norms = group_gradient_norms(y_params, z_params + q_params)
            grad_clipped = False
            active_clip_rule = self.final_grad_clip_rule if phase == "final" else self.inner_grad_clip_rule
            if self.grad_clip:
                if active_clip_rule == "separate_hg":
                    h_norm = torch.nn.utils.clip_grad_norm_(y_params, self.h_grad_norm)
                    g_norm = torch.nn.utils.clip_grad_norm_(z_params + q_params, self.g_grad_norm)
                    h_norm_value = float(h_norm.detach().cpu().item())
                    g_norm_value = float(g_norm.detach().cpu().item())
                    grad_clipped = bool(
                        (np.isfinite(h_norm_value) and h_norm_value > self.h_grad_norm)
                        or (np.isfinite(g_norm_value) and g_norm_value > self.g_grad_norm)
                    )
                else:
                    total_norm = torch.nn.utils.clip_grad_norm_(params, self.grad_norm)
                    grad_norm_value_from_clip = float(total_norm.detach().cpu().item())
                    grad_clipped = bool(
                        np.isfinite(grad_norm_value_from_clip)
                        and grad_norm_value_from_clip > self.grad_norm
                    )
            else:
                grad_clipped = False

            post_clip_norms = group_gradient_norms(y_params, z_params + q_params)
            optimizer.step()
            diag_recorder.record_step(
                prepared=diagnostic_step,
                h_params=y_params,
                g_params=z_params + q_params,
                pre_clip_norms=pre_clip_norms,
                post_clip_norms=post_clip_norms,
                grad_clip_enabled=self.grad_clip,
                grad_clip_rule=active_clip_rule,
                clip_norm=self.grad_norm,
                clipped_indicator=grad_clipped,
                clip_norm_h=self.h_grad_norm if active_clip_rule == "separate_hg" else None,
                clip_norm_g=self.g_grad_norm if active_clip_rule == "separate_hg" else None,
                clip_norm_total=self.grad_norm if active_clip_rule == "joint" else None,
            )
            grad_norm_value = pre_clip_norms["total"]
            return grad_norm_value, grad_clipped

        def _tensor_mean(value):
            if value is None or not isinstance(value, torch.Tensor):
                return np.nan
            return float(value.detach().float().mean().cpu().item())

        def _tensor_cov(a, b):
            if a is None or b is None or not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                return np.nan
            a_flat = a.detach().float().reshape(-1)
            b_flat = b.detach().float().reshape(-1)
            if a_flat.numel() == 0 or b_flat.numel() == 0 or a_flat.numel() != b_flat.numel():
                return np.nan
            return float(((a_flat - a_flat.mean()) * (b_flat - b_flat.mean())).mean().cpu().item())

        def _component_value(components, key):
            value = components.get(key) if components else None
            if isinstance(value, torch.Tensor) and value.ndim == 0:
                return _loss_value(value)
            return np.nan

        def _coerce_loss_components(value):
            if isinstance(value, dict):
                return value
            if isinstance(value, torch.Tensor):
                zero = value.new_zeros(())
                return {
                    "loss_mode": self.loss_mode,
                    "ce_residual": None,
                    "ce_loss": value,
                    "martingale_term": None,
                    "martingale_loss": zero,
                    "native_residual": None,
                    "native_loss": value,
                }
            raise TypeError(f"loss_fn returned unsupported type {type(value)!r}.")

        def _build_loss_components(
            train_components,
            val_components,
            train_consistency_loss,
            val_consistency_loss,
            train_qgrad=None,
            val_qgrad=None,
            grad_norm="",
            grad_clipped="",
            validation_checked=True,
            validation_check_index="",
        ):
            train_residual = _component_value(train_components, "native_loss")
            val_residual = _component_value(val_components, "native_loss")
            train_ce = _component_value(train_components, "ce_loss")
            val_ce = _component_value(val_components, "ce_loss")
            train_martingale = _component_value(train_components, "martingale_loss")
            val_martingale = _component_value(val_components, "martingale_loss")
            train_consistency = _loss_value(train_consistency_loss)
            val_consistency = _loss_value(val_consistency_loss) if val_consistency_loss is not None else np.nan
            train_weighted = self.training_consistency_coef * train_consistency
            val_weighted = self.training_consistency_coef * val_consistency
            return {
                "loss_mode": self.loss_mode,
                "validation_checked": bool(validation_checked),
                "validation_check_index": validation_check_index,
                "train_residual_loss": train_residual,
                "val_residual_loss": val_residual,
                "train_ce_residual_loss": train_ce,
                "val_ce_residual_loss": val_ce,
                "train_martingale_loss": train_martingale,
                "val_martingale_loss": val_martingale,
                "train_martingale_term_mean": _tensor_mean(train_components.get("martingale_term") if train_components else None),
                "val_martingale_term_mean": _tensor_mean(val_components.get("martingale_term") if val_components else None),
                "train_residual_martingale_cov": _tensor_cov(
                    train_components.get("ce_residual") if train_components else None,
                    train_components.get("martingale_term") if train_components else None,
                ),
                "val_residual_martingale_cov": _tensor_cov(
                    val_components.get("ce_residual") if val_components else None,
                    val_components.get("martingale_term") if val_components else None,
                ),
                "train_qgrad_mean": _tensor_mean(train_qgrad),
                "val_qgrad_mean": _tensor_mean(val_qgrad),
                "train_consistency_loss": train_consistency,
                "val_consistency_loss": val_consistency,
                "train_weighted_consistency_loss": train_weighted,
                "val_weighted_consistency_loss": val_weighted,
                "train_weighted_consistency_to_residual_ratio": _safe_ratio(train_weighted, train_residual),
                "val_weighted_consistency_to_residual_ratio": _safe_ratio(val_weighted, val_residual),
                "grad_norm": grad_norm,
                "grad_clipped": grad_clipped,
                "loss_finite": bool(
                    np.isfinite(train_residual)
                    and np.isfinite(train_consistency)
                    and (not validation_checked or (np.isfinite(val_residual) and np.isfinite(val_consistency)))
                ),
            }

        def _compute_validation_losses(
            interval_idx,
            is_final,
            y_net,
            z_net,
            qgrad_net,
            Y_n_valid,
            Y_n1_valid,
            a_ref_valid,
            L_inc_valid,
            W_inc_valid,
            next_y_net=None,
        ):
            validation_start = time.time()
            try:
                with torch.no_grad():
                    D_n_valid = self._g_physical_output(z_net, Y_n_valid)
                    Q_n_valid = self._qgrad_physical_output(qgrad_net, Y_n_valid)
                    H_n_valid = self._h_physical_output(y_net, Y_n_valid)
                    if is_final:
                        H_n1_valid = None
                    else:
                        H_n1_valid = self._h_physical_output(next_y_net, Y_n1_valid)
                    components_valid = self.loss_fn(
                        Y_n=Y_n_valid,
                        Y_n1=Y_n1_valid,
                        H_n=H_n_valid.squeeze(1),
                        H_n1=None if H_n1_valid is None else H_n1_valid.squeeze(1),
                        D_n=D_n_valid,
                        a_ref=a_ref_valid,
                        L_increment=L_inc_valid,
                        W_increment=W_inc_valid,
                        Q_n=Q_n_valid,
                        n=interval_idx,
                        is_final=is_final,
                        loss_type=2,
                        return_components=True,
                    )
                    components_valid = _coerce_loss_components(components_valid)
                consistency_loss_valid = self.consistency_loss(
                    y_net,
                    D_n_valid,
                    Y_n_valid,
                    create_graph=False,
                )
                return components_valid, consistency_loss_valid, Q_n_valid
            finally:
                timing_totals["validation_seconds"] += time.time() - validation_start

        def _should_run_validation(iteration_idx, iterations_budget):
            if iterations_budget <= 0:
                return False
            if iterations_budget <= self.validation_every:
                return True
            return (
                iteration_idx == 0
                or (iteration_idx + 1) % self.validation_every == 0
                or iteration_idx == iterations_budget - 1
            )

        replay_original_init_from_source = (
            init_checkpoint_dir is not None
            and self.train_interval_subset is not None
            and self.init_checkpoint_train_subset_mode == "replay_original_init"
        )
        replay_source_train_subset = set(self.train_interval_subset or [])
        replay_initial_states = {}
        if replay_original_init_from_source:
            for n in replay_source_train_subset:
                replay_initial_states[n] = {
                    "y": _clone_state_dict(y_net_list[n]),
                    "z": _clone_state_dict(z_net_list[n]),
                    "qgrad": (
                        _clone_state_dict(qgrad_net_list[n])
                        if qgrad_net_list is not None
                        else None
                    ),
                }

        source_init_info = {}
        source_loaded_count = 0
        if init_checkpoint_dir is not None:
            source_dir = os.path.abspath(str(init_checkpoint_dir))
            source_repr = self._load_source_reparameterization(source_dir)
            checkpoint_pairs = list((init_metadata or {}).get("checkpoint_pairs", []))
            metadata_by_interval = {}
            if init_metadata:
                for pair in checkpoint_pairs:
                    target_interval = int(pair.get("target_interval", pair.get("interval")))
                    metadata_by_interval[target_interval] = pair
            if not metadata_by_interval:
                metadata_by_interval = {
                    n: {
                        "interval": n,
                        "source_interval": n,
                        "target_interval": n,
                    }
                    for n in range(self.num_time_interval)
                }
            for n in sorted(metadata_by_interval):
                pair = metadata_by_interval.get(n, {})
                source_interval = int(pair.get("source_interval", n))
                target_interval = int(pair.get("target_interval", n))
                if target_interval != n:
                    raise RuntimeError(
                        "Checkpoint continuation metadata target interval mismatch: "
                        f"key={n}, target_interval={target_interval}."
                    )
                y_path = pair.get("y_checkpoint") or os.path.join(source_dir, f"y_network{source_interval}.pth")
                z_path = pair.get("z_checkpoint") or os.path.join(source_dir, f"z_network{source_interval}.pth")
                qgrad_path = pair.get("qgrad_checkpoint") or os.path.join(source_dir, f"qgrad_network{source_interval}.pth")
                required_paths = [y_path, z_path] + ([qgrad_path] if self.requires_qgrad else [])
                if not all(os.path.isfile(path) for path in required_paths):
                    raise FileNotFoundError(
                        f"Missing source checkpoint set for interval {n}: "
                        f"y={y_path}, z={z_path}, qgrad={qgrad_path if self.requires_qgrad else '<not required>'}"
                    )
                y_state = self._maybe_convert_loaded_state(
                    torch.load(y_path, map_location=self.device),
                    is_h=True,
                    source_repr=source_repr,
                )
                z_state = self._maybe_convert_loaded_state(
                    torch.load(z_path, map_location=self.device),
                    is_h=False,
                    source_repr=source_repr,
                )
                y_net_list[n].load_state_dict(y_state)
                z_net_list[n].load_state_dict(z_state)
                if self.requires_qgrad:
                    qgrad_state = self._maybe_convert_loaded_state(
                        torch.load(qgrad_path, map_location=self.device),
                        is_h=False,
                        source_repr=source_repr,
                    )
                    qgrad_net_list[n].load_state_dict(qgrad_state)
                source_loaded_count += 1
                if not (replay_original_init_from_source and n in replay_source_train_subset):
                    source_init_info[n] = {
                        "reuse_prev_init": False,
                        "reason": (
                            "external_checkpoint_block_entry"
                            if pair.get("initialization_source") == "coarse_block_entry"
                            else "external_checkpoint_same_interval"
                        ),
                        "initialization_mode": (init_metadata or {}).get("initialization_mode", "external_checkpoint"),
                        "initialization_source": pair.get("initialization_source", "same_interval"),
                        "source_interval": source_interval,
                        "source_block_index": pair.get("source_block_index", ""),
                        "source_checkpoint_hash": pair.get("source_checkpoint_hash", ""),
                        "block_entry_interval": pair.get("block_entry_interval", ""),
                        "source_lambda": (init_metadata or {}).get("source_lambda", ""),
                        "source_checkpoint_dir": source_dir,
                        "source_y_checkpoint": y_path,
                        "source_z_checkpoint": z_path,
                        "source_qgrad_checkpoint": qgrad_path if self.requires_qgrad else "",
                        "mu_matches": True,
                        "gamma_matches": True,
                        "mu_max_abs_diff": "",
                        "gamma_max_abs_diff": "",
                        "eta_max_abs_diff": "",
                        "tol": self.prev_init_param_tol,
                    }
            if replay_original_init_from_source:
                for n in sorted(replay_initial_states):
                    states = replay_initial_states[n]
                    y_net_list[n].load_state_dict(states["y"])
                    z_net_list[n].load_state_dict(states["z"])
                    if qgrad_net_list is not None:
                        qgrad_net_list[n].load_state_dict(states["qgrad"])
            print(
                "Initialized all intervals from source checkpoints:",
                {
                    "source_checkpoint_dir": source_dir,
                    "source_lambda": (init_metadata or {}).get("source_lambda", ""),
                    "target_lambda": (init_metadata or {}).get("target_lambda", ""),
                    "loaded_interval_count": source_loaded_count,
                    "same_interval_source_init_count": len(source_init_info),
                    "replay_original_init_interval_count": (
                        len(replay_source_train_subset)
                        if replay_original_init_from_source
                        else 0
                    ),
                    "train_subset_mode": self.init_checkpoint_train_subset_mode,
                },
            )
            if replay_original_init_from_source:
                print(
                    "Restored train_interval_subset intervals to original initialization:",
                    {
                        "intervals": sorted(replay_source_train_subset),
                        "target_chain": (
                            "Source checkpoints remain loaded for non-selected intervals; "
                            "selected intervals reuse newly trained n+1 inside contiguous "
                            "replay windows, otherwise the source n+1 checkpoint."
                        ),
                    },
                )

        # Resume: load existing checkpoints and record which intervals are already trained
        trained = set()
        if resume_from_dir and os.path.isdir(resume_from_dir):
            for n in range(self.num_time_interval):
                y_path = os.path.join(resume_from_dir, f"y_network{n}.pth")
                z_path = os.path.join(resume_from_dir, f"z_network{n}.pth")
                qgrad_path = os.path.join(resume_from_dir, f"qgrad_network{n}.pth")
                required_paths = [y_path, z_path] + ([qgrad_path] if self.requires_qgrad else [])
                existing_count = sum(os.path.isfile(path) for path in required_paths)
                if existing_count == len(required_paths):
                    try:
                        y_state = torch.load(y_path, map_location=self.device)
                        z_state = torch.load(z_path, map_location=self.device)
                        qgrad_state = (
                            torch.load(qgrad_path, map_location=self.device)
                            if self.requires_qgrad
                            else None
                        )
                    except Exception as exc:
                        raise RuntimeError(f"Failed to load checkpoint set for interval {n}: {exc}") from exc
                    y_net_list[n].load_state_dict(y_state)
                    z_net_list[n].load_state_dict(z_state)
                    if self.requires_qgrad:
                        qgrad_net_list[n].load_state_dict(qgrad_state)
                    trained.add(n)
                elif existing_count > 0:
                    raise FileNotFoundError(
                        f"Incomplete checkpoint set for interval {n} in {resume_from_dir}: "
                        f"required={required_paths}"
                    )
            if trained:
                print(f"Resuming: loaded {len(trained)} intervals from {resume_from_dir}")

        # Initialize optimizer and learning rate scheduler (for final interval; recreated per interval if needed)
        optimizer = torch.optim.Adam(
            list(y_net_list[self.num_time_interval - 1].parameters())
            + list(z_net_list[self.num_time_interval - 1].parameters())
            + (
                list(qgrad_net_list[self.num_time_interval - 1].parameters())
                if qgrad_net_list is not None
                else []
            ),
            lr=self.final_learning_rate,
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = self.final_milestones, gamma = self.final_gamma)
        
        # Generate validation data for the final interval
        # The terminal cost can be zero on part of the state space, so the final interval may have
        # weak supervision. Validation-based early stopping plus best-checkpoint restore helps avoid
        # overtraining toward an overly flat terminal solution.
        Y_n_valid, Y_n1_valid, a_ref_valid, L_inc_valid, W_inc_valid = self.sample(
            self.valid_size,
            self.num_time_interval - 1,
            sample_purpose="validation",
            iteration_idx=-1,
        )
        self._log_state_support_summary(
            state_support_summary_path,
            self.num_time_interval - 1,
            Y_n_valid,
        )

        # Step 2: Train the network for the final interval (skip if resuming and already trained)
        if (
            (self.num_time_interval - 1) not in trained
            and _should_train_interval(self.num_time_interval - 1)
        ):
            best_val_loss = float("inf")
            best_iter = None
            best_train_loss = None
            best_loss_components = {}
            best_y_state = None
            best_z_state = None
            best_qgrad_state = None
            early_stop_reference_loss = float("inf")
            patience_counter = 0
            patience = self.patience
            min_delta = self.min_delta
            stop_iter = None
            final_train_loss = None
            final_val_loss = None
            final_loss_components = {}
            grad_norm_values = []
            grad_clipped_count = 0
            loss_finite_all = True
            early_stopped = False
            apply_early_stopping_final = False
            validation_checks_completed = 0
            final_schedule = {
                "iterations": final_iterations,
                "learning_rate": self.final_learning_rate,
                "milestones": self.final_milestones,
                "gamma": self.final_gamma,
                "minimum_iterations": self.final_min_iterations,
                "budget_type": "final",
                "early_stopping": apply_early_stopping_final,
            }
            final_init_decision = source_init_info.get(
                self.num_time_interval - 1,
                {
                    "reuse_prev_init": False,
                    "reason": "terminal_interval",
                    "initialization_mode": "production_initialization",
                    "mu_max_abs_diff": "",
                    "gamma_max_abs_diff": "",
                    "eta_max_abs_diff": "",
                },
            )

            for i in range(final_iterations):
                y_net_list[self.num_time_interval - 1].train()
                z_net_list[self.num_time_interval - 1].train()
                if qgrad_net_list is not None:
                    qgrad_net_list[self.num_time_interval - 1].train()

                Y_n_train, Y_n1_train, a_ref_train, L_inc_train, W_inc_train = self.sample(
                    self.batch_size,
                    self.num_time_interval - 1,
                    sample_purpose="train",
                    iteration_idx=i,
                )
                D_n_train = self._g_physical_output(
                    z_net_list[self.num_time_interval - 1],
                    Y_n_train,
                )
                H_n_train = self._h_physical_output(
                    y_net_list[self.num_time_interval - 1],
                    Y_n_train,
                )
                Q_n_train = self._qgrad_physical_output(
                    None if qgrad_net_list is None else qgrad_net_list[self.num_time_interval - 1],
                    Y_n_train,
                )
                train_components = self.loss_fn(
                    Y_n=Y_n_train,
                    Y_n1=Y_n1_train,
                    H_n=H_n_train.squeeze(1),
                    H_n1=None,
                    D_n=D_n_train,
                    a_ref=a_ref_train,
                    L_increment=L_inc_train,
                    W_increment=W_inc_train,
                    Q_n=Q_n_train,
                    n=self.num_time_interval - 1,
                    is_final=True,
                    loss_type=2,
                    return_components=True,
                )
                train_components = _coerce_loss_components(train_components)
                residual_loss = train_components["native_loss"]
                consistency_loss = self.consistency_loss(
                    y_net_list[self.num_time_interval - 1],
                    D_n_train,
                    Y_n_train,
                    create_graph=self.training_consistency_coef > 0.0,
                )
                loss = residual_loss + self.training_consistency_coef * consistency_loss
                loss_train = loss.item()

                grad_norm_value, grad_clipped = _apply_optimizer_step(
                    self.num_time_interval - 1,
                    loss,
                    optimizer,
                    phase="final",
                    iteration_idx=i,
                    residual_loss_tensor=residual_loss,
                    consistency_loss_tensor=consistency_loss,
                    schedule=final_schedule,
                )
                learning_rate_now = optimizer.param_groups[0]["lr"]
                grad_norm_values.append(grad_norm_value)
                grad_clipped_count += int(bool(grad_clipped))
                y_net_list[self.num_time_interval - 1].eval()
                z_net_list[self.num_time_interval - 1].eval()
                if qgrad_net_list is not None:
                    qgrad_net_list[self.num_time_interval - 1].eval()
                validation_checked = _should_run_validation(i, final_iterations)
                validation_check_index = ""
                val_components = None
                consistency_loss_valid = None
                Q_n_valid = None
                loss_valid = ""
                if validation_checked:
                    validation_checks_completed += 1
                    validation_check_index = validation_checks_completed
                    val_components, consistency_loss_valid, Q_n_valid = _compute_validation_losses(
                        self.num_time_interval - 1,
                        True,
                        y_net_list[self.num_time_interval - 1],
                        z_net_list[self.num_time_interval - 1],
                        None if qgrad_net_list is None else qgrad_net_list[self.num_time_interval - 1],
                        Y_n_valid,
                        Y_n1_valid,
                        a_ref_valid,
                        L_inc_valid,
                        W_inc_valid,
                    )
                    loss_valid = (
                        val_components["native_loss"]
                        + self.training_consistency_coef * consistency_loss_valid
                    ).item()
                current_loss_components = _build_loss_components(
                    train_components,
                    val_components,
                    consistency_loss,
                    consistency_loss_valid,
                    train_qgrad=Q_n_train,
                    val_qgrad=Q_n_valid,
                    grad_norm=grad_norm_value,
                    grad_clipped=grad_clipped,
                    validation_checked=validation_checked,
                    validation_check_index=validation_check_index,
                )
                current_loss_components["learning_rate"] = learning_rate_now
                loss_finite_all = bool(loss_finite_all and current_loss_components["loss_finite"])

                if validation_checked and loss_valid < best_val_loss:
                    best_val_loss = loss_valid
                    best_iter = i
                    best_train_loss = loss_train
                    best_loss_components = dict(current_loss_components)
                    best_y_state = _clone_state_dict(y_net_list[self.num_time_interval - 1])
                    best_z_state = _clone_state_dict(z_net_list[self.num_time_interval - 1])
                    best_qgrad_state = (
                        _clone_state_dict(qgrad_net_list[self.num_time_interval - 1])
                        if qgrad_net_list is not None
                        else None
                    )

                stop_training = False
                if apply_early_stopping_final and validation_checked:
                    if loss_valid < early_stop_reference_loss - min_delta:
                        early_stop_reference_loss = loss_valid
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter > patience and (i + 1) >= self.final_min_iterations:
                        stop_training = True
                        early_stopped = True

                stop_iter = i
                final_train_loss = loss_train
                final_val_loss = loss_valid
                final_loss_components = dict(current_loss_components)

                if stop_training:
                    elapsed_time = time.time() - start_time
                    _record_history(
                        self.num_time_interval - 1,
                        i,
                        loss_train,
                        loss_valid,
                        elapsed_time,
                        best_val_loss=best_val_loss,
                        best_iter=best_iter,
                        **current_loss_components,
                    )
                    print(f"Early stopping at iteration {i}, time interval {self.num_time_interval - 1}")
                    break

                if i % self.print_interval == 0:
                    elapsed_time = time.time() - start_time
                    _record_history(
                        self.num_time_interval - 1,
                        i,
                        loss_train,
                        loss_valid,
                        elapsed_time,
                        best_val_loss=best_val_loss,
                        best_iter=best_iter,
                        **current_loss_components,
                    )
                    print(
                        "interval: ", self.num_time_interval - 1,
                        " iter: ", i,
                        " train_loss: ", loss_train,
                        " validation_loss: ", loss_valid,
                        " train_residual_loss: ", current_loss_components["train_residual_loss"],
                        " validation_residual_loss: ", current_loss_components["val_residual_loss"],
                        " train_consistency_loss: ", current_loss_components["train_consistency_loss"],
                        " validation_consistency_loss: ", current_loss_components["val_consistency_loss"],
                        " train_weighted_consistency_loss: ", current_loss_components["train_weighted_consistency_loss"],
                        " validation_weighted_consistency_loss: ", current_loss_components["val_weighted_consistency_loss"],
                        " elapsed_time: ", elapsed_time,
                    )

                # Always record the last executed iteration, even if it is not on a print boundary.
                if i == final_iterations - 1:
                    elapsed_time = time.time() - start_time
                    _record_history(
                        self.num_time_interval - 1,
                        i,
                        loss_train,
                        loss_valid,
                        elapsed_time,
                        best_val_loss=best_val_loss,
                        best_iter=best_iter,
                        **current_loss_components,
                    )

                # MultiStepLR follows optimizer.step(); early stopping skips the scheduler step.
                scheduler.step()

            if final_iterations > 0 and stop_iter is not None:
                final_loss_components.update(
                    {
                        "grad_norm_mean": float(np.mean(grad_norm_values)) if grad_norm_values else np.nan,
                        "grad_norm_max": float(np.max(grad_norm_values)) if grad_norm_values else np.nan,
                        "grad_clipping_frequency": float(grad_clipped_count / max(len(grad_norm_values), 1)),
                        "nan_or_inf_indicator": not bool(loss_finite_all),
                    }
                )
                restored_best = _restore_best_checkpoint(
                    self.num_time_interval - 1,
                    best_y_state,
                    best_z_state,
                    best_qgrad_state,
                )
                elapsed_time = time.time() - start_time
                _record_history(
                    self.num_time_interval - 1,
                    stop_iter,
                    final_train_loss,
                    final_val_loss,
                    elapsed_time,
                    best_val_loss=best_val_loss,
                    best_iter=best_iter,
                    final_val_loss=final_val_loss,
                    stop_iter=stop_iter,
                    early_stopped=early_stopped,
                    restored_best=restored_best,
                    **final_loss_components,
                )
                _save_interval_checkpoint(self.num_time_interval - 1)
                trained.add(self.num_time_interval - 1)
                _record_interval_summary(
                    self.num_time_interval - 1,
                    final_schedule,
                    final_init_decision,
                    best_iter,
                    best_train_loss,
                    best_val_loss,
                    final_train_loss,
                    final_val_loss,
                    stop_iter,
                    early_stopped,
                    restored_best,
                    elapsed_time,
                    best_loss_components=best_loss_components,
                    final_loss_components=final_loss_components,
                )
                _print_interval_summary(
                    self.num_time_interval - 1,
                    best_val_loss,
                    best_iter,
                    final_val_loss,
                    stop_iter,
                    early_stopped,
                    restored_best,
                    best_loss_components=best_loss_components,
                    final_loss_components=final_loss_components,
                )
            _append_history_to_csv()
        elif not _should_train_interval(self.num_time_interval - 1):
            print(
                "Skipping terminal interval due to train_interval_subset:",
                self.num_time_interval - 1,
            )

        if train_final_interval_only:
            print(
                "train_final_interval_only=True: stopping after terminal interval "
                f"{self.num_time_interval - 1}."
            )
            diag_recorder.finalize()
            self.last_training_timing = dict(timing_totals)
            return

        # Step 3: Train networks for the remaining intervals in reverse order
        for n in range(self.num_time_interval - 2, -1, -1):

            if not _should_train_interval(n):
                continue

            if n in trained:
                continue

            if self.train_interval_subset is not None and (n + 1) not in trained:
                next_interval_has_source_target = (
                    init_checkpoint_dir is not None
                    and not (
                        replay_original_init_from_source
                        and (n + 1) in replay_source_train_subset
                    )
                )
                if next_interval_has_source_target:
                    print(
                        "Diagnostic interval subset is training interval "
                        f"{n} using source checkpoint interval {n + 1} as the fixed target."
                    )
                else:
                    print(
                        "Warning: diagnostic interval subset is training interval "
                        f"{n} while interval {n + 1} is not trained/loaded. "
                        "This is acceptable only for gradient-scale smoke diagnostics."
                    )

            init_decision = source_init_info.get(n) or self._get_interval_init_decision(n)
            if init_decision["reuse_prev_init"]:
                y_net_list[n].load_state_dict(y_net_list[n+1].state_dict())
                z_net_list[n].load_state_dict(z_net_list[n+1].state_dict())
                if qgrad_net_list is not None:
                    qgrad_net_list[n].load_state_dict(qgrad_net_list[n + 1].state_dict())
            print(
                "interval init:",
                {
                    "interval": n,
                    "reuse_prev_init": init_decision["reuse_prev_init"],
                    "reason": init_decision["reason"],
                    "tol": init_decision["tol"],
                    "mu_max_abs_diff": init_decision["mu_max_abs_diff"],
                    "gamma_max_abs_diff": init_decision["gamma_max_abs_diff"],
                    "eta_max_abs_diff": init_decision["eta_max_abs_diff"],
                    "initialization_mode": init_decision.get("initialization_mode", init_decision["reason"]),
                    "source_lambda": init_decision.get("source_lambda", ""),
                    "source_checkpoint_dir": init_decision.get("source_checkpoint_dir", ""),
                },
            )

            schedule = self._get_interval_training_schedule(init_decision)
            print(
                "interval schedule:",
                {
                    "interval": n,
                    "budget_type": schedule["budget_type"],
                    "iterations": schedule["iterations"],
                    "learning_rate": schedule["learning_rate"],
                    "milestones": schedule["milestones"],
                    "gamma": schedule["gamma"],
                    "minimum_iterations": schedule["minimum_iterations"],
                    "early_stopping": schedule["early_stopping"],
                },
            )

            for param in y_net_list[n+1].parameters():
                param.requires_grad = False

            for param in z_net_list[n+1].parameters():
                param.requires_grad = False
            if qgrad_net_list is not None:
                for param in qgrad_net_list[n + 1].parameters():
                    param.requires_grad = False

            optimizer = torch.optim.Adam(
                                        list(y_net_list[n].parameters()) +
                                        list(z_net_list[n].parameters()) +
                                        (
                                            list(qgrad_net_list[n].parameters())
                                            if qgrad_net_list is not None
                                            else []
                                        ), lr=schedule["learning_rate"])

            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=schedule["milestones"],
                gamma=schedule["gamma"],
            )

            Y_n_valid, Y_n1_valid, a_ref_valid, L_inc_valid, W_inc_valid = self.sample(
                self.valid_size,
                n,
                sample_purpose="validation",
                iteration_idx=-1,
            )
            self._log_state_support_summary(state_support_summary_path, n, Y_n_valid)

            best_val_loss = float("inf")
            best_iter = None
            best_train_loss = None
            best_loss_components = {}
            best_y_state = None
            best_z_state = None
            best_qgrad_state = None
            early_stop_reference_loss = float("inf")
            patience_counter = 0
            patience = self.patience
            min_delta = self.min_delta
            stop_iter = None
            final_train_loss = None
            final_val_loss = None
            final_loss_components = {}
            grad_norm_values = []
            grad_clipped_count = 0
            loss_finite_all = True
            early_stopped = False
            validation_checks_completed = 0

            for i in range(schedule["iterations"]):
                y_net_list[n].train()
                z_net_list[n].train()
                if qgrad_net_list is not None:
                    qgrad_net_list[n].train()

                Y_n_train, Y_n1_train, a_ref_train, L_inc_train, W_inc_train = self.sample(
                    self.batch_size,
                    n,
                    sample_purpose="train",
                    iteration_idx=i,
                )
                D_n_train = self._g_physical_output(z_net_list[n], Y_n_train)
                H_n_train = self._h_physical_output(y_net_list[n], Y_n_train)
                H_n1_train = self._h_physical_output(y_net_list[n + 1], Y_n1_train)
                Q_n_train = self._qgrad_physical_output(
                    None if qgrad_net_list is None else qgrad_net_list[n],
                    Y_n_train,
                )
                train_components = self.loss_fn(
                    Y_n=Y_n_train,
                    Y_n1=Y_n1_train,
                    H_n=H_n_train.squeeze(1),
                    H_n1=H_n1_train.squeeze(1),
                    D_n=D_n_train,
                    a_ref=a_ref_train,
                    L_increment=L_inc_train,
                    W_increment=W_inc_train,
                    Q_n=Q_n_train,
                    n=n,
                    is_final=False,
                    loss_type=2,
                    return_components=True,
                )
                train_components = _coerce_loss_components(train_components)
                residual_loss = train_components["native_loss"]
                consistency_loss = self.consistency_loss(
                    y_net_list[n],
                    D_n_train,
                    Y_n_train,
                    create_graph=self.training_consistency_coef > 0.0,
                )
                loss = residual_loss + self.training_consistency_coef * consistency_loss
                loss_train = loss.item()

                grad_norm_value, grad_clipped = _apply_optimizer_step(
                    n,
                    loss,
                    optimizer,
                    phase="interior",
                    iteration_idx=i,
                    residual_loss_tensor=residual_loss,
                    consistency_loss_tensor=consistency_loss,
                    schedule=schedule,
                )
                learning_rate_now = optimizer.param_groups[0]["lr"]
                grad_norm_values.append(grad_norm_value)
                grad_clipped_count += int(bool(grad_clipped))

                y_net_list[n].eval()
                z_net_list[n].eval()
                if qgrad_net_list is not None:
                    qgrad_net_list[n].eval()
                validation_checked = _should_run_validation(i, schedule["iterations"])
                validation_check_index = ""
                val_components = None
                consistency_loss_valid = None
                Q_n_valid = None
                loss_valid = ""
                if validation_checked:
                    validation_checks_completed += 1
                    validation_check_index = validation_checks_completed
                    val_components, consistency_loss_valid, Q_n_valid = _compute_validation_losses(
                        n,
                        False,
                        y_net_list[n],
                        z_net_list[n],
                        None if qgrad_net_list is None else qgrad_net_list[n],
                        Y_n_valid,
                        Y_n1_valid,
                        a_ref_valid,
                        L_inc_valid,
                        W_inc_valid,
                        next_y_net=y_net_list[n + 1],
                    )
                    loss_valid = (
                        val_components["native_loss"]
                        + self.training_consistency_coef * consistency_loss_valid
                    ).item()
                current_loss_components = _build_loss_components(
                    train_components,
                    val_components,
                    consistency_loss,
                    consistency_loss_valid,
                    train_qgrad=Q_n_train,
                    val_qgrad=Q_n_valid,
                    grad_norm=grad_norm_value,
                    grad_clipped=grad_clipped,
                    validation_checked=validation_checked,
                    validation_check_index=validation_check_index,
                )
                current_loss_components["learning_rate"] = learning_rate_now
                loss_finite_all = bool(loss_finite_all and current_loss_components["loss_finite"])

                if validation_checked and loss_valid < best_val_loss:
                    best_val_loss = loss_valid
                    best_iter = i
                    best_train_loss = loss_train
                    best_loss_components = dict(current_loss_components)
                    best_y_state = _clone_state_dict(y_net_list[n])
                    best_z_state = _clone_state_dict(z_net_list[n])
                    best_qgrad_state = (
                        _clone_state_dict(qgrad_net_list[n])
                        if qgrad_net_list is not None
                        else None
                    )

                stop_training = False
                if schedule["early_stopping"] and validation_checked:
                    if loss_valid < early_stop_reference_loss - min_delta:
                        early_stop_reference_loss = loss_valid
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter > patience and (i + 1) >= int(schedule.get("minimum_iterations", 0)):
                        stop_training = True
                        early_stopped = True

                stop_iter = i
                final_train_loss = loss_train
                final_val_loss = loss_valid
                final_loss_components = dict(current_loss_components)

                if stop_training:
                    elapsed_time = time.time() - start_time
                    _record_history(
                        n,
                        i,
                        loss_train,
                        loss_valid,
                        elapsed_time,
                        best_val_loss=best_val_loss,
                        best_iter=best_iter,
                        **current_loss_components,
                    )
                    print(f"Early stopping at iteration {i}, time interval {n}")
                    break

                # Step MultiStepLR after the optimizer unless early stopping has ended the interval.
                scheduler.step()

                if i % self.print_interval == 0:
                    elapsed_time = time.time() - start_time
                    _record_history(
                        n,
                        i,
                        loss_train,
                        loss_valid,
                        elapsed_time,
                        best_val_loss=best_val_loss,
                        best_iter=best_iter,
                        **current_loss_components,
                    )
                    print(
                        "interval: ", n,
                        " iter: ", i,
                        " train_loss: ", loss_train,
                        " validation_loss: ", loss_valid,
                        " train_residual_loss: ", current_loss_components["train_residual_loss"],
                        " validation_residual_loss: ", current_loss_components["val_residual_loss"],
                        " train_consistency_loss: ", current_loss_components["train_consistency_loss"],
                        " validation_consistency_loss: ", current_loss_components["val_consistency_loss"],
                        " train_weighted_consistency_loss: ", current_loss_components["train_weighted_consistency_loss"],
                        " validation_weighted_consistency_loss: ", current_loss_components["val_weighted_consistency_loss"],
                        " elapsed_time: ", elapsed_time,
                    )

                # Always record the last executed iteration, even if it is not on a print boundary.
                if i == schedule["iterations"] - 1:
                    elapsed_time = time.time() - start_time
                    _record_history(
                        n,
                        i,
                        loss_train,
                        loss_valid,
                        elapsed_time,
                        best_val_loss=best_val_loss,
                        best_iter=best_iter,
                        **current_loss_components,
                    )

            if schedule["iterations"] > 0 and stop_iter is not None:
                final_loss_components.update(
                    {
                        "grad_norm_mean": float(np.mean(grad_norm_values)) if grad_norm_values else np.nan,
                        "grad_norm_max": float(np.max(grad_norm_values)) if grad_norm_values else np.nan,
                        "grad_clipping_frequency": float(grad_clipped_count / max(len(grad_norm_values), 1)),
                        "nan_or_inf_indicator": not bool(loss_finite_all),
                    }
                )
                restored_best = _restore_best_checkpoint(n, best_y_state, best_z_state, best_qgrad_state)
                elapsed_time = time.time() - start_time
                _record_history(
                    n,
                    stop_iter,
                    final_train_loss,
                    final_val_loss,
                    elapsed_time,
                    best_val_loss=best_val_loss,
                    best_iter=best_iter,
                    final_val_loss=final_val_loss,
                    stop_iter=stop_iter,
                    early_stopped=early_stopped,
                    restored_best=restored_best,
                    **final_loss_components,
                )
                _save_interval_checkpoint(n)
                trained.add(n)
                _record_interval_summary(
                    n,
                    schedule,
                    init_decision,
                    best_iter,
                    best_train_loss,
                    best_val_loss,
                    final_train_loss,
                    final_val_loss,
                    stop_iter,
                    early_stopped,
                    restored_best,
                    elapsed_time,
                    best_loss_components=best_loss_components,
                    final_loss_components=final_loss_components,
                )
                _print_interval_summary(
                    n,
                    best_val_loss,
                    best_iter,
                    final_val_loss,
                    stop_iter,
                    early_stopped,
                    restored_best,
                    best_loss_components=best_loss_components,
                    final_loss_components=final_loss_components,
                )
            _append_history_to_csv()

        # Step 4: Save the final weights for use in C++ simulation (optional)
        if save_path_cppweights:
            os.makedirs(save_path_cppweights, exist_ok=True)
            for n in range(self.num_time_interval):
                self.save_weights_as_txt(z_net_list[n], f"z_network{n}", save_path_cppweights)
        
        diag_recorder.finalize()
        self.last_training_timing = dict(timing_totals)
        return training_history
