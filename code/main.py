import json
import os
import argparse
import shutil
import csv
import time
from types import SimpleNamespace

import torch 
import numpy as np
from torch import nn
import pandas as pd

import equation as eqn
from checkpoint_continuation import (
    load_json as load_continuation_json,
    save_continuation_metadata,
    validate_checkpoint_continuation,
)
from solver import DSPL_solver
from Net import Y_Net
from Net import Z_Net
from Net import G_INIT_METHOD_CHOICES, INIT_METHOD_CHOICES

REFERENCE_POLICY_CHOICES = [
    "static_proportional",
    "inventory_scaled_proportional",
    "myopic",
    "status_quo",
    "fluid_optimal_rollout",
    "reference_mixture",
    "even",
]
PROPOSAL_SAMPLER_CHOICES = [
    "empirical",
    "uniform_tube",
    "mixture_uniform_lowq_boundary",
    "truncated_lognormal",
]
ACTIVATION_CHOICES = [
    "ReLU",
    "LeakyReLU",
    "GELU",
    "SELU",
    "SiLU",
    "CELU",
]
ETA_MODE_CHOICES = list(eqn.ETA_MODES)
SAMPLER_CONFIG_KEYS = [
    "sampling_mode",
    "proposal_sampler",
    "rollout_pool_size",
    "sampling_seed",
    "stateless_seed_schedule",
    "seed_schedule_experiment",
    "seed_schedule_key",
    "uniform_z_radius_mult",
    "uniform_q_radius_frac",
    "total_inventory_buffer",
    "q_low_values",
    "boundary_eps_frac",
    "mix_rollout",
    "mix_box",
    "mix_shortage",
    "shortage_q_frac",
    "shortage_z_frac",
    "lognormal_delta_z",
    "lognormal_delta_q",
    "lognormal_r_min",
    "initial_state_mode",
    "initial_perturb_prob",
    "initial_q_perturb_frac",
    "initial_z_perturb_frac",
    "initial_perturb_minus_prob",
    "initial_perturb_plus_prob",
    "initial_perturb_minus_frac",
    "initial_perturb_plus_frac",
    "initial_dirichlet_alpha",
    "loss_mode",
    "validation_every",
    "consistency_coef",
    "train_seed",
    "init_method",
    "h_init_method",
    "g_init_method",
    "g_output_bias",
    "g_prior_fluid_plan_path",
    "g_prior_epsilon_a",
    "g_prior_lower_bound",
    "g_prior_upper_bound",
    "g_prior_cap_offset",
    "train_final_interval_only",
    "prev_init",
    "prev_init_same_params_only",
    "grad_clip",
    "grad_norm",
    "h_grad_norm",
    "g_grad_norm",
    "grad_clip_rule",
    "final_grad_clip_rule",
    "inner_grad_clip_rule",
    "input_mode",
    "q0_input_scale",
    "h_output_scale",
    "checkpoint_reparam_load_mode",
    "state_support_logging",
    "state_support_sample_size",
    "gradient_diagnostics",
    "gradient_diagnostics_interval_subset",
    "gradient_diagnostics_every",
    "gradient_diagnostics_component_every",
    "gradient_diagnostics_max_records_per_interval",
    "train_interval_subset",
    "init_checkpoint_train_subset_mode",
    "component_seed_isolation",
    "h_network_seed",
    "g_network_seed",
    "qgrad_network_seed",
    "save_initial_checkpoints",
]

try:
    import munch
except ImportError:  # pragma: no cover - optional dependency
    munch = None

def to_config_namespace(obj):
    if munch is not None:
        return munch.munchify(obj)
    if isinstance(obj, dict):
        return SimpleNamespace(**{key: to_config_namespace(value) for key, value in obj.items()})
    if isinstance(obj, list):
        return [to_config_namespace(value) for value in obj]
    return obj


def parse_int_list_arg(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_bool_arg(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected boolean value such as 0/1 or true/false, got {value!r}."
    )


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type = str, help = "The path to load json file")
    parser.add_argument('--run_name', type = str, help = 'The name of numerical experiments')
    parser.add_argument('--output_dir', type = str, default = ".",
                       help = 'Base directory for all outputs')
    
    # Parameters for hyperparameter testing
    parser.add_argument('--num_time_interval', type = int, default = None,
                       help = 'Override num_time_interval in both eqn_config and net_config')
    parser.add_argument('--num_layers', type = int, default = None,
                       help = 'Override num_layers in net_config')
    parser.add_argument('--num_neurons', type = int, default = None,
                       help = 'Override num_neurons in net_config')
    parser.add_argument('--activation', type=str, default=None,
                       choices=ACTIVATION_CHOICES,
                       help='Override activation in net_config')
    parser.add_argument('--init_method', type=str, default=None,
                       choices=INIT_METHOD_CHOICES,
                       help='Override network weight initialization in net_config')
    parser.add_argument('--h_init_method', type=str, default=None,
                       choices=INIT_METHOD_CHOICES,
                       help='Override H-network initialization without changing G.')
    parser.add_argument('--g_init_method', type=str, default=None,
                       choices=G_INIT_METHOD_CHOICES,
                       help='Override G-network initialization without changing H.')
    parser.add_argument('--g_output_bias', type=parse_bool_arg, default=None,
                       help='Enable a final-layer G bias. Required for benchmark inverse G initializations.')
    parser.add_argument('--g_prior_fluid_plan_path', type=str, default=None,
                       help='Fluid allocation_plan.npz for g_init_method=fluid_inverse.')
    parser.add_argument('--g_prior_epsilon_a', type=float, default=None,
                       help='Action floor used when inverting benchmark shipments into G priors.')
    parser.add_argument('--g_prior_lower_bound', type=float, default=None,
                       help='Lower clipping bound for benchmark G priors.')
    parser.add_argument('--g_prior_upper_bound', type=float, default=None,
                       help='Upper clipping bound for benchmark G priors.')
    parser.add_argument('--g_prior_cap_offset', type=float, default=None,
                       help='Small offset below the eta-cap threshold for cap-regime benchmark priors.')
    parser.add_argument('--loss_mode', type=str, default=None,
                       choices=['ce_consistency', 'ce_no_consistency', 'martingale_qgrad'],
                       help='Semantic loss mode for LOSS-00/LOSS-01 ablations')
    parser.add_argument('--final_learning_rate', type=float, default=None,
                       help='Override final_learning_rate in net_config')
    parser.add_argument('--inner_learning_rate', type=float, default=None,
                       help='Override inner_learning_rate in net_config')
    parser.add_argument('--final_iterations', type=int, default=None,
                       help='Override final_iterations in net_config')
    parser.add_argument('--inner_iterations', type=int, default=None,
                       help='Override inner_iterations in net_config')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch_size in net_config')
    parser.add_argument('--valid_size', type=int, default=None,
                       help='Override valid_size in net_config')
    parser.add_argument('--validation_every', type=int, default=None,
                       help='Run full validation every N optimizer iterations')
    parser.add_argument('--final_min_iterations', type=int, default=None,
                       help='Minimum terminal-interval optimizer iterations before early stopping can fire')
    parser.add_argument('--inner_min_iterations', type=int, default=None,
                       help='Minimum inner-interval optimizer iterations before early stopping can fire')
    parser.add_argument('--final_milestones', type=str, default=None,
                       help='Comma-separated override for final_milestones, e.g. 1000,4000,12000')
    parser.add_argument('--inner_milestones', type=str, default=None,
                       help='Comma-separated override for inner_milestones, e.g. 1000,4000')
    parser.add_argument('--gamma', type=float, default=None,
                       help='Override learning-rate scheduler gamma in net_config')
    parser.add_argument('--inner_gamma', type=float, default=None,
                       help='Override inner-interval learning-rate scheduler gamma in net_config')
    parser.add_argument('--final_gamma', type=float, default=None,
                       help='Override terminal-interval learning-rate scheduler gamma in net_config')
    parser.add_argument('--grad_norm', type=float, default=None,
                       help='Override gradient clipping norm in net_config')
    parser.add_argument('--grad_clip', type=parse_bool_arg, default=None,
                       help='Override whether gradient clipping is enabled. Accepts 0/1 or true/false.')
    parser.add_argument('--grad_clip_rule', type=str, default=None,
                       choices=['joint', 'separate_hg'],
                       help='Gradient clipping rule: joint H+G norm or separate H/G clipping')
    parser.add_argument('--final_grad_clip_rule', type=str, default=None,
                       choices=['joint', 'separate_hg'],
                       help='Terminal-interval clipping rule; defaults to grad_clip_rule')
    parser.add_argument('--inner_grad_clip_rule', type=str, default=None,
                       choices=['joint', 'separate_hg'],
                       help='Interior-interval clipping rule; defaults to grad_clip_rule')
    parser.add_argument('--h_grad_norm', type=float, default=None,
                       help='H clip norm used when a clipping rule is separate_hg')
    parser.add_argument('--g_grad_norm', type=float, default=None,
                       help='G clip norm used when a clipping rule is separate_hg')
    parser.add_argument('--input_mode', type=str, default=None,
                       choices=['raw', 'q0'],
                       help='Function-preserving state-input representation')
    parser.add_argument('--q0_input_scale', type=float, default=None,
                       help='Q0 scalar used for input_mode=q0; defaults to eqn_config.I0')
    parser.add_argument('--h_output_scale', type=float, default=None,
                       help='Function-preserving H output scale s_H, with H_physical=H_hat/s_H')
    parser.add_argument('--checkpoint_reparam_load_mode', type=str, default=None,
                       choices=['guard', 'convert'],
                       help='Guard or function-preservingly convert mismatched raw/scaled source checkpoints')
    parser.add_argument('--state_support_logging', action='store_true',
                       help='Write compact training/reference state-support quantiles per interval')
    parser.add_argument('--state_support_sample_size', type=int, default=None,
                       help='Sample size for compact reference state-support summaries')
    parser.add_argument('--gradient_diagnostics', action='store_true',
                       help='Enable CLIP-00 gradient-scale diagnostics')
    parser.add_argument('--gradient_diagnostics_interval_subset', type=str, default=None,
                       help='Comma-separated intervals for diagnostics; default records all trained intervals')
    parser.add_argument('--gradient_diagnostics_every', type=int, default=None,
                       help='Record gradient diagnostics every N optimizer iterations')
    parser.add_argument('--gradient_diagnostics_component_every', type=int, default=None,
                       help='Record component-gradient diagnostics every N optimizer iterations')
    parser.add_argument('--gradient_diagnostics_max_records_per_interval', type=int, default=None,
                       help='Maximum diagnostic rows to record per interval')
    parser.add_argument('--train_interval_subset', type=str, default=None,
                       help='Comma-separated interval subset to train for diagnostic-only smoke runs')
    parser.add_argument('--component_seed_isolation', type=parse_bool_arg, default=None,
                       help='Initialize H, G, and optional QGrad networks from independent component seeds.')
    parser.add_argument('--h_network_seed', type=int, default=None,
                       help='Component seed for H networks when component_seed_isolation is enabled.')
    parser.add_argument('--g_network_seed', type=int, default=None,
                       help='Component seed for G networks when component_seed_isolation is enabled.')
    parser.add_argument('--qgrad_network_seed', type=int, default=None,
                       help='Component seed for QGrad networks when component_seed_isolation is enabled.')
    parser.add_argument('--save_initial_checkpoints', type=parse_bool_arg, default=None,
                       help='Write initial H/G checkpoint copies. Disable for large M runs to avoid duplicate storage.')
    parser.add_argument('--reference_policy', type=str, default=None,
                       choices=REFERENCE_POLICY_CHOICES,
                       help='Override eqn_config.policy for the reference rollout process')
    parser.add_argument('--reference_fluid_plan_path', type=str, default=None,
                       help='Optional allocation_plan.npz used only for reference_policy=fluid_optimal_rollout training rollouts')
    parser.add_argument('--reference_mixture', type=str, default=None,
                       choices=['R0', 'R1', 'R2', 'R3', 'R4'],
                       help='Path-level reference mixture label for reference_policy=reference_mixture')
    parser.add_argument('--reference_mixture_counts', type=str, default=None,
                       help='Exact path counts such as R0:8192,R2:8192; must sum to rollout_pool_size')
    parser.add_argument('--eta_mode', type=str, default=None,
                       choices=ETA_MODE_CHOICES,
                       help='Override eqn_config.eta_mode for shipment bounds')
    parser.add_argument('--eta_profile', type=str, default=None,
                       choices=ETA_MODE_CHOICES,
                       help='Alias for --eta_mode used in production experiment metadata')
    parser.add_argument('--chi_eta', type=float, default=None,
                       help='Override eqn_config.chi_eta for weekly time-varying shipment bounds')
    parser.add_argument('--s_eta', type=float, default=None,
                       help='Override eqn_config.s_eta for weekly time-varying shipment bounds')
    parser.add_argument('--lambda_param', type=float, default=None,
                       help='Override eqn_config.lambda_param and net_config.lambd_const')
    parser.add_argument('--alpha', type=float, default=None,
                       help='Override eqn_config.alpha')
    parser.add_argument('--proposal_sampler', type=str, default=None,
                       choices=PROPOSAL_SAMPLER_CHOICES,
                       help='Override proposal sampler for interval-start training states')
    parser.add_argument('--sampling_mode', type=str, default=None,
                       choices=['box', 'rollout', 'shortage', 'mixture', 'mixed', 'lognormal_proposal', 'lognormal'],
                       help='Override state sampling backend in net_config')
    parser.add_argument('--rollout_pool_size', type=int, default=None,
                       help='Override rollout_pool_size in net_config')
    parser.add_argument('--sampling_seed', type=int, default=None,
                       help='Override sampling_seed in net_config')
    parser.add_argument('--stateless_seed_schedule', type=parse_bool_arg, default=None,
                       help='Use interval/iteration/purpose-indexed RNG for training samples')
    parser.add_argument('--seed_schedule_experiment', type=str, default=None,
                       help='Experiment label included in stateless training-sample RNG hashes')
    parser.add_argument('--seed_schedule_key', type=str, default=None,
                       help='Pairing key included in stateless training-sample RNG hashes')
    parser.add_argument('--mix_rollout', type=float, default=None,
                       help='Override mix_rollout in net_config')
    parser.add_argument('--mix_box', type=float, default=None,
                       help='Override mix_box in net_config')
    parser.add_argument('--mix_shortage', type=float, default=None,
                       help='Override mix_shortage in net_config')
    parser.add_argument('--shortage_q_frac', type=float, default=None,
                       help='Override shortage_q_frac in net_config')
    parser.add_argument('--shortage_z_frac', type=float, default=None,
                       help='Override shortage_z_frac in net_config')
    parser.add_argument('--lognormal_delta_z', type=float, default=None,
                       help='Override lognormal_delta_z in net_config')
    parser.add_argument('--lognormal_delta_q', type=float, default=None,
                       help='Override lognormal_delta_q in net_config')
    parser.add_argument('--lognormal_r_min', type=int, default=None,
                       help='Override lognormal_r_min in net_config')
    parser.add_argument('--initial_state_mode', type=str, default=None,
                       choices=['standard', 'q_sensitivity_mixture', 'paper_70_30'],
                       help='Override initial-state distribution for rollout training sampler')
    parser.add_argument('--initial_perturb_prob', type=float, default=None,
                       help='Probability of paper initial perturbation paths')
    parser.add_argument('--initial_q_perturb_frac', type=float, default=None,
                       help='Half-width of paper Q perturbation as a fraction of Q0')
    parser.add_argument('--initial_z_perturb_frac', type=float, default=None,
                       help='Upper bound of paper local inventory perturbation as a fraction of Q0')
    parser.add_argument('--initial_perturb_minus_prob', type=float, default=None,
                       help='Probability of initial Q-to-local redistribution paths')
    parser.add_argument('--initial_perturb_plus_prob', type=float, default=None,
                       help='Probability of diagnostic initial Q-plus paths')
    parser.add_argument('--initial_perturb_minus_frac', type=float, default=None,
                       help='Max fraction of Q0 redistributed to local inventories initially')
    parser.add_argument('--initial_perturb_plus_frac', type=float, default=None,
                       help='Max fraction of Q0 added to diagnostic Q-plus initial states')
    parser.add_argument('--initial_dirichlet_alpha', type=float, default=None,
                       help='Symmetric Dirichlet alpha for initial local redistribution')
    parser.add_argument('--consistency_coef', type=float, default=None,
                       help='Override consistency_coef in net_config')
    parser.add_argument('--gradient_assembly_mode', type=str, default=None,
                       choices=['standard', 'hg_component'],
                       help='Assemble H/G gradients from base and consistency components')
    parser.add_argument('--h_base_grad_multiplier', type=float, default=None,
                       help='Multiplier for H base-residual gradient in hg_component mode')
    parser.add_argument('--g_base_grad_multiplier', type=float, default=None,
                       help='Multiplier for G base-residual gradient in hg_component mode')
    parser.add_argument('--h_consistency_coef', type=float, default=None,
                       help='H-side consistency gradient coefficient in hg_component mode')
    parser.add_argument('--g_consistency_coef', type=float, default=None,
                       help='G-side consistency gradient coefficient in hg_component mode')
    parser.add_argument('--prev_init_same_params_only', type=parse_bool_arg, default=None,
                       help='Override prev_init_same_params_only in net_config. Accepts 0/1 or true/false.')
    parser.add_argument('--uniform_z_radius_mult', type=float, default=None,
                       help='Override uniform_tube Z-radius multiplier in net_config')
    parser.add_argument('--uniform_q_radius_frac', type=float, default=None,
                       help='Override uniform_tube Q-radius fraction in net_config')
    parser.add_argument('--total_inventory_buffer', type=float, default=None,
                       help='Override physical total-inventory buffer for proposal projection')
    parser.add_argument('--train_seed', type=int, default=None,
                       help='Override train_seed in net_config and seed torch/numpy before training')
    parser.add_argument('--eval_seed', type=int, default=None,
                       help='Record evaluation seed in run-local config metadata')
    parser.add_argument('--eval_num_paths', type=int, default=None,
                       help='Record evaluation path count in run-local config metadata')
    parser.add_argument('--eval_ext_num_paths', type=int, default=None,
                       help='Record extended-diagnostic evaluation path count in run-local config metadata')
    parser.add_argument('--train_final_interval_only', action='store_true',
                       help='Train only the terminal interval N-1 and skip all earlier intervals')

    parser.add_argument('--resume', action='store_true',
                        help='Resume training from existing checkpoints in run_dir (use same run_name and output_dir)')
    parser.add_argument('--init_checkpoint_dir', type=str, default=None,
                        help='Optional source logs dir used to initialize every interval from matching H/G checkpoints')
    parser.add_argument('--init_checkpoint_source_lambda', type=float, default=None,
                        help='Expected lambda_param for --init_checkpoint_dir; mismatches fail loudly')
    parser.add_argument('--init_checkpoint_mode', type=str, default='lambda_homotopy',
                        help='Metadata label for source-checkpoint initialization')
    parser.add_argument('--init_checkpoint_train_subset_mode', type=str, default=None,
                        choices=['same_interval', 'replay_original_init'],
                        help=(
                            'When --init_checkpoint_dir and --train_interval_subset are both set, '
                            'same_interval initializes selected intervals from their matching source '
                            'checkpoints; replay_original_init keeps source checkpoints only as the '
                            'target chain and restores selected intervals to their original '
                            'production initialization.'
                        ))

    args = parser.parse_args()

    return args


def load_config(args, config_path=None):
    """Load config from JSON. If config_path is None, use args.config_path."""
    path = config_path or args.config_path
    with open(path) as json_data_file:
        config = json.load(json_data_file)

    if args.num_time_interval is not None:
        config['eqn_config']['num_time_interval'] = args.num_time_interval
        config['net_config']['num_time_interval'] = args.num_time_interval
        print(f"Overriding num_time_interval to {args.num_time_interval}")

    if args.num_layers is not None:
        config['net_config']['num_layers'] = args.num_layers
        print(f"Overriding num_layers to {args.num_layers}")

    if args.num_neurons is not None:
        config['net_config']['num_neurons'] = args.num_neurons
        print(f"Overriding num_neurons to {args.num_neurons}")

    if args.activation is not None:
        config['net_config']['activation'] = args.activation
        print(f"Overriding activation to {args.activation}")

    config['net_config'].setdefault('init_method', 'xavier_uniform')
    if args.init_method is not None:
        config['net_config']['init_method'] = args.init_method
        print(f"Overriding init_method to {args.init_method}")
    if args.h_init_method is not None:
        config['net_config']['h_init_method'] = args.h_init_method
        print(f"Overriding h_init_method to {args.h_init_method}")
    if args.g_init_method is not None:
        config['net_config']['g_init_method'] = args.g_init_method
        print(f"Overriding g_init_method to {args.g_init_method}")
    if args.g_output_bias is not None:
        config['net_config']['g_output_bias'] = args.g_output_bias
        print(f"Overriding g_output_bias to {args.g_output_bias}")
    for key in (
        'g_prior_fluid_plan_path',
        'g_prior_epsilon_a',
        'g_prior_lower_bound',
        'g_prior_upper_bound',
        'g_prior_cap_offset',
    ):
        value = getattr(args, key)
        if value is not None:
            config['net_config'][key] = value
            print(f"Overriding {key} to {value}")

    if args.loss_mode is not None:
        config['net_config']['loss_mode'] = args.loss_mode
        config['net_config'].pop('loss_type', None)
        print(f"Overriding loss_mode to {args.loss_mode}")
    elif 'loss_mode' not in config['net_config'] and config['net_config'].get('loss_type') == 2:
        config['net_config']['loss_mode'] = 'ce_consistency'

    if args.final_learning_rate is not None:
        config['net_config']['final_learning_rate'] = args.final_learning_rate
        print(f"Overriding final_learning_rate to {args.final_learning_rate}")

    if args.inner_learning_rate is not None:
        config['net_config']['inner_learning_rate'] = args.inner_learning_rate
        print(f"Overriding inner_learning_rate to {args.inner_learning_rate}")

    if args.final_iterations is not None:
        config['net_config']['final_iterations'] = args.final_iterations
        print(f"Overriding final_iterations to {args.final_iterations}")

    if args.inner_iterations is not None:
        config['net_config']['inner_iterations'] = args.inner_iterations
        print(f"Overriding inner_iterations to {args.inner_iterations}")

    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch_size must be positive.")
        config['net_config']['batch_size'] = args.batch_size
        print(f"Overriding batch_size to {args.batch_size}")

    if args.valid_size is not None:
        if args.valid_size <= 0:
            raise ValueError("--valid_size must be positive.")
        config['net_config']['valid_size'] = args.valid_size
        print(f"Overriding valid_size to {args.valid_size}")

    config['net_config'].setdefault('validation_every', 25)
    if args.validation_every is not None:
        if args.validation_every <= 0:
            raise ValueError("--validation_every must be positive.")
        config['net_config']['validation_every'] = args.validation_every
        print(f"Overriding validation_every to {args.validation_every}")

    if args.final_min_iterations is not None:
        config['net_config']['final_min_iterations'] = args.final_min_iterations
        print(f"Overriding final_min_iterations to {args.final_min_iterations}")

    if args.inner_min_iterations is not None:
        config['net_config']['inner_min_iterations'] = args.inner_min_iterations
        print(f"Overriding inner_min_iterations to {args.inner_min_iterations}")

    if args.final_milestones is not None:
        config['net_config']['final_milestones'] = parse_int_list_arg(args.final_milestones)
        print(f"Overriding final_milestones to {config['net_config']['final_milestones']}")

    if args.inner_milestones is not None:
        config['net_config']['inner_milestones'] = parse_int_list_arg(args.inner_milestones)
        print(f"Overriding inner_milestones to {config['net_config']['inner_milestones']}")

    if args.gamma is not None:
        config['net_config']['gamma'] = args.gamma
        print(f"Overriding gamma to {args.gamma}")

    if args.inner_gamma is not None:
        config['net_config']['inner_gamma'] = args.inner_gamma
        print(f"Overriding inner_gamma to {args.inner_gamma}")

    if args.final_gamma is not None:
        config['net_config']['final_gamma'] = args.final_gamma
        print(f"Overriding final_gamma to {args.final_gamma}")

    if args.grad_norm is not None:
        config['net_config']['grad_norm'] = args.grad_norm
        print(f"Overriding grad_norm to {args.grad_norm}")

    if args.grad_clip is not None:
        config['net_config']['grad_clip'] = args.grad_clip
        print(f"Overriding grad_clip to {args.grad_clip}")

    if args.grad_clip_rule is not None:
        config['net_config']['grad_clip_rule'] = args.grad_clip_rule
        print(f"Overriding grad_clip_rule to {args.grad_clip_rule}")

    if args.final_grad_clip_rule is not None:
        config['net_config']['final_grad_clip_rule'] = args.final_grad_clip_rule
        print(f"Overriding final_grad_clip_rule to {args.final_grad_clip_rule}")

    if args.inner_grad_clip_rule is not None:
        config['net_config']['inner_grad_clip_rule'] = args.inner_grad_clip_rule
        print(f"Overriding inner_grad_clip_rule to {args.inner_grad_clip_rule}")

    if args.h_grad_norm is not None:
        config['net_config']['h_grad_norm'] = args.h_grad_norm
        print(f"Overriding h_grad_norm to {args.h_grad_norm}")

    if args.g_grad_norm is not None:
        config['net_config']['g_grad_norm'] = args.g_grad_norm
        print(f"Overriding g_grad_norm to {args.g_grad_norm}")

    config['net_config'].setdefault('input_mode', 'raw')
    config['net_config'].setdefault('q0_input_scale', config['eqn_config'].get('I0', 142.0))
    config['net_config'].setdefault('h_output_scale', 1.0)
    config['net_config'].setdefault('checkpoint_reparam_load_mode', 'guard')
    if args.input_mode is not None:
        config['net_config']['input_mode'] = args.input_mode
        print(f"Overriding input_mode to {args.input_mode}")

    if args.q0_input_scale is not None:
        config['net_config']['q0_input_scale'] = args.q0_input_scale
        print(f"Overriding q0_input_scale to {args.q0_input_scale}")

    if args.h_output_scale is not None:
        config['net_config']['h_output_scale'] = args.h_output_scale
        print(f"Overriding h_output_scale to {args.h_output_scale}")

    if args.checkpoint_reparam_load_mode is not None:
        config['net_config']['checkpoint_reparam_load_mode'] = args.checkpoint_reparam_load_mode
        print(f"Overriding checkpoint_reparam_load_mode to {args.checkpoint_reparam_load_mode}")

    if args.state_support_logging:
        config['net_config']['state_support_logging'] = True
        print("Enabling state_support_logging")

    if args.state_support_sample_size is not None:
        if args.state_support_sample_size <= 0:
            raise ValueError("--state_support_sample_size must be positive.")
        config['net_config']['state_support_sample_size'] = args.state_support_sample_size
        print(f"Overriding state_support_sample_size to {args.state_support_sample_size}")

    if args.gradient_diagnostics:
        config['net_config']['gradient_diagnostics'] = True
        print("Enabling gradient diagnostics")

    if args.gradient_diagnostics_interval_subset is not None:
        config['net_config']['gradient_diagnostics_interval_subset'] = parse_int_list_arg(
            args.gradient_diagnostics_interval_subset
        )
        print(
            "Overriding gradient_diagnostics_interval_subset to "
            f"{config['net_config']['gradient_diagnostics_interval_subset']}"
        )

    if args.gradient_diagnostics_every is not None:
        if args.gradient_diagnostics_every <= 0:
            raise ValueError("--gradient_diagnostics_every must be positive.")
        config['net_config']['gradient_diagnostics_every'] = args.gradient_diagnostics_every
        print(f"Overriding gradient_diagnostics_every to {args.gradient_diagnostics_every}")

    if args.gradient_diagnostics_component_every is not None:
        if args.gradient_diagnostics_component_every <= 0:
            raise ValueError("--gradient_diagnostics_component_every must be positive.")
        config['net_config']['gradient_diagnostics_component_every'] = args.gradient_diagnostics_component_every
        print(
            "Overriding gradient_diagnostics_component_every to "
            f"{args.gradient_diagnostics_component_every}"
        )

    if args.gradient_diagnostics_max_records_per_interval is not None:
        if args.gradient_diagnostics_max_records_per_interval <= 0:
            raise ValueError("--gradient_diagnostics_max_records_per_interval must be positive.")
        config['net_config']['gradient_diagnostics_max_records_per_interval'] = (
            args.gradient_diagnostics_max_records_per_interval
        )
        print(
            "Overriding gradient_diagnostics_max_records_per_interval to "
            f"{args.gradient_diagnostics_max_records_per_interval}"
        )

    if args.train_interval_subset is not None:
        config['net_config']['train_interval_subset'] = parse_int_list_arg(args.train_interval_subset)
        print(f"Overriding train_interval_subset to {config['net_config']['train_interval_subset']}")

    if args.init_checkpoint_train_subset_mode is not None:
        config['net_config']['init_checkpoint_train_subset_mode'] = args.init_checkpoint_train_subset_mode
        print(
            "Overriding init_checkpoint_train_subset_mode to "
            f"{args.init_checkpoint_train_subset_mode}"
        )

    for key in (
        'component_seed_isolation',
        'h_network_seed',
        'g_network_seed',
        'qgrad_network_seed',
        'save_initial_checkpoints',
    ):
        value = getattr(args, key)
        if value is not None:
            config['net_config'][key] = value
            print(f"Overriding {key} to {value}")

    if args.train_final_interval_only:
        config['net_config']['train_final_interval_only'] = True
        print("Training final interval only")

    if args.reference_policy is not None:
        config['eqn_config']['policy'] = args.reference_policy
        print(f"Overriding reference_policy/eqn_config.policy to {args.reference_policy}")
    if args.reference_fluid_plan_path is not None:
        config['eqn_config']['reference_fluid_plan_path'] = args.reference_fluid_plan_path
        print(f"Recording reference_fluid_plan_path for training rollouts: {args.reference_fluid_plan_path}")
    if args.reference_mixture is not None:
        config['eqn_config']['reference_mixture'] = args.reference_mixture
        print(f"Recording reference_mixture: {args.reference_mixture}")
    if args.reference_mixture_counts is not None:
        config['eqn_config']['reference_mixture_counts'] = args.reference_mixture_counts
        print(f"Recording reference_mixture_counts: {args.reference_mixture_counts}")

    config['eqn_config'].setdefault('eta_mode', config['eqn_config'].get('eta_profile', 'constant_peak'))
    config['eqn_config'].setdefault('eta_profile', config['eqn_config']['eta_mode'])
    config['eqn_config'].setdefault('eta_schedule_file', 'eta_schedule.csv')
    requested_eta_mode = args.eta_mode if args.eta_mode is not None else args.eta_profile
    if requested_eta_mode is not None:
        config['eqn_config']['eta_mode'] = requested_eta_mode
        config['eqn_config']['eta_profile'] = requested_eta_mode
        print(f"Overriding eta_mode/eta_profile to {requested_eta_mode}")
    config['eqn_config'].setdefault('chi_eta', 1.75)
    config['eqn_config'].setdefault('s_eta', 0.0)
    if args.chi_eta is not None:
        config['eqn_config']['chi_eta'] = args.chi_eta
        print(f"Overriding chi_eta to {args.chi_eta}")
    if args.s_eta is not None:
        config['eqn_config']['s_eta'] = args.s_eta
        print(f"Overriding s_eta to {args.s_eta}")
    if args.lambda_param is not None:
        config['eqn_config']['lambda_param'] = args.lambda_param
        config['net_config']['lambd_const'] = args.lambda_param
        print(f"Overriding lambda_param/lambd_const to {args.lambda_param}")
    if args.alpha is not None:
        config['eqn_config']['alpha'] = args.alpha
        print(f"Overriding alpha to {args.alpha}")
    fairness_spec = config['eqn_config'].get('fairness_spec')
    if fairness_spec is not None and fairness_spec != eqn.FAIRNESS_SPEC:
        raise ValueError(
            f"Config fairness_spec={fairness_spec!r}; expected {eqn.FAIRNESS_SPEC!r}."
        )
    config['eqn_config']['fairness_spec'] = eqn.FAIRNESS_SPEC
    config['eqn_config']['fairness_includes_hf'] = eqn.FAIRNESS_INCLUDES_HF
    config['eqn_config']['fairness_period_h_factor'] = eqn.FAIRNESS_PERIOD_H_FACTOR

    if args.proposal_sampler is not None:
        config['net_config']['proposal_sampler'] = args.proposal_sampler
        print(f"Overriding proposal_sampler to {args.proposal_sampler}")

    if args.sampling_mode is not None:
        config['net_config']['sampling_mode'] = args.sampling_mode
        print(f"Overriding sampling_mode to {args.sampling_mode}")

    if args.rollout_pool_size is not None:
        config['net_config']['rollout_pool_size'] = args.rollout_pool_size
        print(f"Overriding rollout_pool_size to {args.rollout_pool_size}")

    if args.sampling_seed is not None:
        config['net_config']['sampling_seed'] = args.sampling_seed
        print(f"Overriding sampling_seed to {args.sampling_seed}")

    if args.stateless_seed_schedule is not None:
        config['net_config']['stateless_seed_schedule'] = args.stateless_seed_schedule
        print(f"Overriding stateless_seed_schedule to {args.stateless_seed_schedule}")

    if args.seed_schedule_experiment is not None:
        config['net_config']['seed_schedule_experiment'] = args.seed_schedule_experiment
        print(f"Overriding seed_schedule_experiment to {args.seed_schedule_experiment}")

    if args.seed_schedule_key is not None:
        config['net_config']['seed_schedule_key'] = args.seed_schedule_key
        print(f"Overriding seed_schedule_key to {args.seed_schedule_key}")

    if args.mix_rollout is not None:
        config['net_config']['mix_rollout'] = args.mix_rollout
        print(f"Overriding mix_rollout to {args.mix_rollout}")

    if args.mix_box is not None:
        config['net_config']['mix_box'] = args.mix_box
        print(f"Overriding mix_box to {args.mix_box}")

    if args.mix_shortage is not None:
        config['net_config']['mix_shortage'] = args.mix_shortage
        print(f"Overriding mix_shortage to {args.mix_shortage}")

    if args.shortage_q_frac is not None:
        config['net_config']['shortage_q_frac'] = args.shortage_q_frac
        print(f"Overriding shortage_q_frac to {args.shortage_q_frac}")

    if args.shortage_z_frac is not None:
        config['net_config']['shortage_z_frac'] = args.shortage_z_frac
        print(f"Overriding shortage_z_frac to {args.shortage_z_frac}")

    if args.lognormal_delta_z is not None:
        config['net_config']['lognormal_delta_z'] = args.lognormal_delta_z
        print(f"Overriding lognormal_delta_z to {args.lognormal_delta_z}")

    if args.lognormal_delta_q is not None:
        config['net_config']['lognormal_delta_q'] = args.lognormal_delta_q
        print(f"Overriding lognormal_delta_q to {args.lognormal_delta_q}")

    if args.lognormal_r_min is not None:
        config['net_config']['lognormal_r_min'] = args.lognormal_r_min
        print(f"Overriding lognormal_r_min to {args.lognormal_r_min}")

    if args.initial_state_mode is not None:
        config['net_config']['initial_state_mode'] = args.initial_state_mode
        print(f"Overriding initial_state_mode to {args.initial_state_mode}")

    if args.initial_perturb_prob is not None:
        config['net_config']['initial_perturb_prob'] = args.initial_perturb_prob
        print(f"Overriding initial_perturb_prob to {args.initial_perturb_prob}")

    if args.initial_q_perturb_frac is not None:
        config['net_config']['initial_q_perturb_frac'] = args.initial_q_perturb_frac
        print(f"Overriding initial_q_perturb_frac to {args.initial_q_perturb_frac}")

    if args.initial_z_perturb_frac is not None:
        config['net_config']['initial_z_perturb_frac'] = args.initial_z_perturb_frac
        print(f"Overriding initial_z_perturb_frac to {args.initial_z_perturb_frac}")

    if args.initial_perturb_minus_prob is not None:
        config['net_config']['initial_perturb_minus_prob'] = args.initial_perturb_minus_prob
        print(f"Overriding initial_perturb_minus_prob to {args.initial_perturb_minus_prob}")

    if args.initial_perturb_plus_prob is not None:
        config['net_config']['initial_perturb_plus_prob'] = args.initial_perturb_plus_prob
        print(f"Overriding initial_perturb_plus_prob to {args.initial_perturb_plus_prob}")

    if args.initial_perturb_minus_frac is not None:
        config['net_config']['initial_perturb_minus_frac'] = args.initial_perturb_minus_frac
        print(f"Overriding initial_perturb_minus_frac to {args.initial_perturb_minus_frac}")

    if args.initial_perturb_plus_frac is not None:
        config['net_config']['initial_perturb_plus_frac'] = args.initial_perturb_plus_frac
        print(f"Overriding initial_perturb_plus_frac to {args.initial_perturb_plus_frac}")

    if args.initial_dirichlet_alpha is not None:
        config['net_config']['initial_dirichlet_alpha'] = args.initial_dirichlet_alpha
        print(f"Overriding initial_dirichlet_alpha to {args.initial_dirichlet_alpha}")

    if args.consistency_coef is not None:
        config['net_config']['consistency_coef'] = args.consistency_coef
        print(f"Overriding consistency_coef to {args.consistency_coef}")

    for key in (
        'gradient_assembly_mode',
        'h_base_grad_multiplier',
        'g_base_grad_multiplier',
        'h_consistency_coef',
        'g_consistency_coef',
    ):
        value = getattr(args, key)
        if value is not None:
            config['net_config'][key] = value
            print(f"Overriding {key} to {value}")

    if args.prev_init_same_params_only is not None:
        config['net_config']['prev_init_same_params_only'] = args.prev_init_same_params_only
        print(f"Overriding prev_init_same_params_only to {args.prev_init_same_params_only}")

    if args.uniform_z_radius_mult is not None:
        config['net_config']['uniform_z_radius_mult'] = args.uniform_z_radius_mult
        print(f"Overriding uniform_z_radius_mult to {args.uniform_z_radius_mult}")

    if args.uniform_q_radius_frac is not None:
        config['net_config']['uniform_q_radius_frac'] = args.uniform_q_radius_frac
        print(f"Overriding uniform_q_radius_frac to {args.uniform_q_radius_frac}")

    if args.total_inventory_buffer is not None:
        config['net_config']['total_inventory_buffer'] = args.total_inventory_buffer
        print(f"Overriding total_inventory_buffer to {args.total_inventory_buffer}")

    if args.train_seed is not None:
        config['net_config']['train_seed'] = args.train_seed
        print(f"Overriding train_seed to {args.train_seed}")

    eval_config = dict(config.get("eval_config", {}))
    if args.eval_seed is not None:
        eval_config["eval_seed"] = args.eval_seed
        eval_config["fine_grid_seed"] = args.eval_seed
        eval_config["periodic_seed"] = args.eval_seed
        print(f"Recording eval_seed {args.eval_seed}")
    if args.eval_num_paths is not None:
        eval_config["eval_num_paths"] = args.eval_num_paths
        print(f"Recording eval_num_paths {args.eval_num_paths}")
    if args.eval_ext_num_paths is not None:
        eval_config["eval_ext_num_paths"] = args.eval_ext_num_paths
        print(f"Recording eval_ext_num_paths {args.eval_ext_num_paths}")
    if eval_config:
        config["eval_config"] = eval_config

    return config, to_config_namespace(config)

def prepare_directories(output_dir, run_name):

    run_dir = os.path.join(output_dir, run_name)
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok = True)
    # os.makedirs(os.path.join(run_dir, "cppweights"), exist_ok = True)
    return run_dir, logs_dir


def save_config_files(raw_config, run_dir):
    config_dir = raw_config["eqn_config"]["config_dir"]
    params_dir = os.path.join(run_dir, "params")
    os.makedirs(params_dir, exist_ok=True)

    param_files = [
        raw_config["eqn_config"].get("theta_file"),
        raw_config["eqn_config"].get("exposure_file"),
        raw_config["eqn_config"].get("mu_file"),
        raw_config["eqn_config"].get("gamma_file"),
        raw_config["eqn_config"].get("p_file"),
        raw_config["eqn_config"].get("distance_file"),
        raw_config["eqn_config"].get("u_file"),
        raw_config["eqn_config"].get("r_file"),
        raw_config["eqn_config"].get("c_file"),
        raw_config["eqn_config"].get("d_file"),
        raw_config["eqn_config"].get("h_file"),
        raw_config["eqn_config"].get("eta_file"),
    ]

    for filename in param_files:
        if not filename:
            continue
        src = os.path.join(config_dir, filename)
        dst = os.path.join(params_dir, filename)
        if os.path.exists(src):
            if os.path.realpath(src) == os.path.realpath(dst):
                continue
            shutil.copy2(src, dst)
        else:
            print(f"Warning: parameter file not found: {src}")


def assert_resume_fairness_compatible(config_path):
    with open(config_path, encoding="utf-8") as config_file:
        existing = json.load(config_file)
    eqn_config = existing.get("eqn_config", {})
    lambda_param = float(eqn_config.get("lambda_param", 0.0))
    fairness_spec = eqn_config.get("fairness_spec")
    if fairness_spec is not None and fairness_spec != eqn.FAIRNESS_SPEC:
        raise ValueError(
            f"Cannot resume {config_path}: fairness_spec={fairness_spec!r}; expected {eqn.FAIRNESS_SPEC!r}."
        )
    if lambda_param > 0.0 and fairness_spec is None:
        raise ValueError(
            f"Cannot resume lambda={lambda_param:g} checkpoint from {config_path}: missing fairness_spec metadata. "
            "Existing lambda-positive checkpoints from the old fairness specification are incompatible."
        )


def sampler_config_snapshot(raw_config):
    net_config = raw_config.get("net_config", {})
    eqn_config = raw_config.get("eqn_config", {})
    return {
        "reference_policy": eqn_config.get("policy"),
        "model_config": {
            "eta_mode": eqn_config.get("eta_mode", "constant_peak"),
            "eta_profile": eqn_config.get("eta_profile", eqn_config.get("eta_mode", "constant_peak")),
            "chi_eta": eqn_config.get("chi_eta", 1.75),
            "s_eta": eqn_config.get("s_eta", 0.0),
            "lambda_param": eqn_config.get("lambda_param", 0.0),
            "alpha": eqn_config.get("alpha", 0.5),
            **eqn.fairness_metadata(),
            "eta_schedule_file": eqn_config.get("eta_schedule_file", "eta_schedule.csv"),
            "reference_fluid_plan_path": eqn_config.get("reference_fluid_plan_path"),
            "reference_fluid_plan_file": eqn_config.get("reference_fluid_plan_file"),
        },
        "net_config": {
            key: net_config.get(key)
            for key in SAMPLER_CONFIG_KEYS
            if key in net_config
        },
        "eval_config": raw_config.get("eval_config", {}),
    }


def save_sampler_config_file(raw_config, run_dir):
    path = os.path.join(run_dir, "sampler_config_used.json")
    with open(path, "w") as config_file:
        json.dump(sampler_config_snapshot(raw_config), config_file, indent=4)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def save_eta_metadata(bsde, raw_config, run_dir):
    params_dir = os.path.join(run_dir, "params")
    os.makedirs(params_dir, exist_ok=True)
    schedule_file = raw_config.get("eqn_config", {}).get("eta_schedule_file", "eta_schedule.csv")
    schedule_path = os.path.join(params_dir, schedule_file)
    eta_raw = bsde.eta_schedule_raw.detach().cpu().numpy()
    np.savetxt(schedule_path, eta_raw, delimiter=",", fmt="%.12g")

    diagnostics = bsde.eta_diagnostics() if hasattr(bsde, "eta_diagnostics") else {
        "eta_mode": raw_config.get("eqn_config", {}).get("eta_mode", "constant_peak"),
        "eta_schedule_raw_shape": list(eta_raw.shape),
    }
    diagnostics["eta_schedule_file"] = os.path.join("params", schedule_file)
    diagnostics["fluid_solver_eta_note"] = (
        "Evaluation uses precomputed Gurobi fluid artifacts for the fluid benchmark; "
        "the Torch-free fluid solver uses the same time-indexed eta schedule."
    )
    diagnostics.update(eqn.fairness_metadata())
    with open(os.path.join(run_dir, "eta_diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=4, default=_json_default)
    print(f"Saved eta schedule to {schedule_path}")


def save_reference_fluid_metadata(bsde, run_dir):
    metadata = getattr(bsde, "reference_fluid_plan_metadata", None)
    if not metadata:
        return
    path = os.path.join(run_dir, "reference_fluid_rollout_diagnostics.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=4, default=_json_default)
    print(f"Saved reference fluid rollout diagnostics to {path}")


def apply_train_seed(raw_config):
    train_seed = raw_config.get("net_config", {}).get("train_seed")
    if train_seed is None:
        return
    seed = int(train_seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Applied train_seed {seed}")


def attach_run_local_training_config(raw_config, args, run_dir):
    """Attach run-directory-local fields after the output path is known."""
    net_config = raw_config.setdefault("net_config", {})
    if net_config.get("gradient_diagnostics"):
        net_config["gradient_diagnostics_output_dir"] = os.path.join(run_dir, "gradient_diagnostics")
        net_config["gradient_diagnostics_append"] = bool(args.resume)
    return raw_config, to_config_namespace(raw_config)


def _to_float_or_nan(value):
    try:
        number = float(value)
    except Exception:
        return np.nan
    return number if np.isfinite(number) else np.nan


def _truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def save_training_run_summary(run_dir, run_name, wall_clock_seconds, timing_summary=None):
    path = os.path.join(run_dir, f"{run_name}_training_interval_summary.csv")
    if not os.path.isfile(path):
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    residuals = np.asarray(
        [_to_float_or_nan(row.get("best_val_residual_loss", "")) for row in rows],
        dtype=float,
    )
    finite_residuals = residuals[np.isfinite(residuals)]
    if finite_residuals.size:
        median_residual = float(np.median(finite_residuals))
        p90_residual = float(np.percentile(finite_residuals, 90))
        max_residual = float(np.max(finite_residuals))
        outlier_threshold = max(1e3, 10.0 * max(median_residual, 1e-12))
        residual_outliers = int(np.sum(finite_residuals > outlier_threshold))
    else:
        median_residual = p90_residual = max_residual = np.nan
        residual_outliers = 0
    nonfinite_residuals = int(residuals.size - finite_residuals.size)
    clipping_values = np.asarray(
        [_to_float_or_nan(row.get("grad_clipping_frequency", "")) for row in rows],
        dtype=float,
    )
    if not np.isfinite(clipping_values).any():
        clipping_values = np.asarray(
            [1.0 if _truthy(row.get("final_grad_clipped", "")) else 0.0 for row in rows],
            dtype=float,
        )
    early_values = np.asarray([1.0 if _truthy(row.get("early_stopped", "")) else 0.0 for row in rows], dtype=float)
    nan_inf_count = sum(
        1
        for row in rows
        if _truthy(row.get("nan_or_inf_indicator", ""))
        or str(row.get("final_loss_finite", "")).strip().lower() == "false"
    )
    gpu_memory_mb = None
    if torch.cuda.is_available():
        gpu_memory_mb = float(torch.cuda.max_memory_allocated() / (1024.0 ** 2))
    timing_summary = dict(timing_summary or {})
    validation_seconds = float(timing_summary.get("validation_seconds", 0.0) or 0.0)
    checkpoint_write_seconds = float(timing_summary.get("checkpoint_write_seconds", 0.0) or 0.0)
    summary = {
        "run_name": run_name,
        "interval_count": len(rows),
        "median_interval_validation_base_residual": median_residual,
        "p90_interval_validation_base_residual": p90_residual,
        "max_interval_validation_base_residual": max_residual,
        "unstable_interval_count": int(nonfinite_residuals + residual_outliers + nan_inf_count),
        "nonfinite_residual_count": nonfinite_residuals,
        "residual_outlier_count": residual_outliers,
        "nan_or_inf_interval_count": int(nan_inf_count),
        "early_stopping_fraction": float(np.mean(early_values)) if early_values.size else np.nan,
        "mean_clipping_frequency": float(np.nanmean(clipping_values)) if clipping_values.size else np.nan,
        "wall_clock_seconds": float(wall_clock_seconds),
        "validation_seconds": validation_seconds,
        "validation_time_fraction": (
            validation_seconds / float(wall_clock_seconds)
            if float(wall_clock_seconds) > 0.0
            else np.nan
        ),
        "checkpoint_write_seconds": checkpoint_write_seconds,
        "checkpoint_write_time_fraction": (
            checkpoint_write_seconds / float(wall_clock_seconds)
            if float(wall_clock_seconds) > 0.0
            else np.nan
        ),
        "gpu_memory_max_allocated_mb": gpu_memory_mb,
    }
    with open(os.path.join(run_dir, "training_run_summary.json"), "w") as f:
        json.dump(summary, f, indent=4, default=_json_default)
    print(f"Saved training run summary to {os.path.join(run_dir, 'training_run_summary.json')}")


def prepare_checkpoint_continuation(args, raw_config, run_dir):
    metadata_path = os.path.join(run_dir, "checkpoint_continuation_metadata.json")
    if args.init_checkpoint_dir:
        metadata = validate_checkpoint_continuation(
            source_checkpoint_dir=args.init_checkpoint_dir,
            target_run_dir=run_dir,
            target_config=raw_config,
            expected_source_lambda=args.init_checkpoint_source_lambda,
            initialization_mode=args.init_checkpoint_mode,
        )
        train_subset_mode = raw_config.get("net_config", {}).get(
            "init_checkpoint_train_subset_mode",
            "same_interval",
        )
        metadata["init_checkpoint_train_subset_mode"] = train_subset_mode
        metadata["effective_train_subset_initialization"] = (
            "source_checkpoints_as_target_chain_with_original_init_for_train_subset"
            if train_subset_mode == "replay_original_init"
            else "source_same_interval_initialization"
        )
        save_continuation_metadata(metadata_path, metadata)
        print(
            "Validated checkpoint continuation:",
            {
                "source_lambda": metadata["source_lambda"],
                "target_lambda": metadata["target_lambda"],
                "source_checkpoint_dir": metadata["source_checkpoint_dir"],
                "init_checkpoint_train_subset_mode": train_subset_mode,
            },
        )
        return metadata

    if args.resume and os.path.isfile(metadata_path):
        metadata = load_continuation_json(metadata_path)
        print(
            "Resume detected existing checkpoint continuation metadata:",
            {
                "source_lambda": metadata.get("source_lambda"),
                "target_lambda": metadata.get("target_lambda"),
                "source_checkpoint_dir": metadata.get("source_checkpoint_dir"),
            },
        )
        return metadata

    return None


def main():

    print(torch.__version__)
    args = parse_arguments()
    training_wall_start = time.time()

    run_dir, logs_dir = prepare_directories(args.output_dir, args.run_name)
    config_path_used = os.path.join(run_dir, "config_used.json")

    if args.resume:
        if not os.path.isfile(config_path_used):
            raise FileNotFoundError(f"Resume requested but {config_path_used} not found. Run without --resume first.")
        assert_resume_fairness_compatible(config_path_used)
        raw_config, config = load_config(args, config_path=config_path_used)
        raw_config, config = attach_run_local_training_config(raw_config, args, run_dir)
        print(f"Resuming from {run_dir}; config loaded from config_used.json")
    else:
        raw_config, config = load_config(args)
        raw_config, config = attach_run_local_training_config(raw_config, args, run_dir)
        with open(config_path_used, "w") as config_file:
            json.dump(raw_config, config_file, indent=4)
        save_sampler_config_file(raw_config, run_dir)
        save_config_files(raw_config, run_dir)

    if args.resume:
        save_sampler_config_file(raw_config, run_dir)

    apply_train_seed(raw_config)

    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    save_eta_metadata(bsde, raw_config, run_dir)
    save_reference_fluid_metadata(bsde, run_dir)

    dpsl_solver = DSPL_solver(config, bsde)
    continuation_metadata = prepare_checkpoint_continuation(args, raw_config, run_dir)

    final_iterations = config.net_config.final_iterations
    inner_iterations = config.net_config.inner_iterations

    save_path_logs = logs_dir + "/"
    save_path_cppweights = None
    training_history_path = os.path.join(run_dir, f"{args.run_name}_training_history.csv")
    interval_summary_path = os.path.join(run_dir, f"{args.run_name}_training_interval_summary.csv")
    state_support_summary_path = os.path.join(run_dir, "state_support_summary.csv")
    if not args.resume:
        for path in (training_history_path, interval_summary_path, state_support_summary_path):
            if os.path.exists(path):
                os.remove(path)

    with open(os.path.join(run_dir, "reparameterization_metadata.json"), "w") as f:
        json.dump(dpsl_solver.reparameterization_metadata(), f, indent=4)

    resume_from_dir = logs_dir if args.resume else None
    dpsl_solver.train(final_iterations, inner_iterations, save_path_logs, save_path_cppweights,
                      resume_from_dir=resume_from_dir, training_history_path=training_history_path,
                      interval_summary_path=interval_summary_path,
                      state_support_summary_path=state_support_summary_path,
                      train_final_interval_only=bool(
                          getattr(config.net_config, "train_final_interval_only", False)
                      ),
                      init_checkpoint_dir=(
                          continuation_metadata.get("source_checkpoint_dir")
                          if continuation_metadata
                          else None
                      ),
                      init_metadata=continuation_metadata)
    save_training_run_summary(
        run_dir,
        args.run_name,
        time.time() - training_wall_start,
        timing_summary=getattr(dpsl_solver, "last_training_timing", {}),
    )

if __name__ == '__main__':
    main()
