import torch


class TrainingStateSampler:
    """Configurable state sampler for Deep Splitting training."""

    SUPPORTED_MODES = {"box", "rollout", "shortage", "mixture", "lognormal_proposal"}
    SUPPORTED_INITIAL_STATE_MODES = {"standard", "q_sensitivity_mixture", "paper_70_30"}
    SUPPORTED_PROPOSALS = {
        "empirical",
        "uniform_tube",
        "mixture_uniform_lowq_boundary",
        "truncated_lognormal",
    }

    def __init__(
        self,
        *,
        bsde,
        device,
        dim,
        num_time_interval,
        sampling_mode,
        rollout_pool_size,
        mix_rollout,
        mix_box,
        mix_shortage,
        sampling_seed,
        box_state_sampler,
        shortage_state_sampler,
        reference_transition,
        reference_label_sampler=None,
        proposal_sampler=None,
        uniform_z_radius_mult=1.0,
        uniform_q_radius_frac=0.10,
        total_inventory_buffer=20.0,
        q_low_values=None,
        boundary_eps_frac=0.05,
        lognormal_delta_z=1e-2,
        lognormal_delta_q=1e-3,
        lognormal_r_min=20,
        initial_state_mode="standard",
        initial_perturb_prob=0.30,
        initial_q_perturb_frac=0.05,
        initial_z_perturb_frac=0.0125,
        initial_perturb_minus_prob=0.20,
        initial_perturb_plus_prob=0.10,
        initial_perturb_minus_frac=0.10,
        initial_perturb_plus_frac=0.05,
        initial_dirichlet_alpha=1.0,
    ):
        self.bsde = bsde
        self.device = device
        self.dim = dim
        self.state_dim = dim + 1
        self.num_time_interval = num_time_interval
        self.sampling_mode = self._normalize_mode_name(sampling_mode)
        self.proposal_sampler = self._normalize_proposal_name(proposal_sampler)
        self.rollout_pool_size = int(rollout_pool_size)
        self.mix_rollout = float(mix_rollout)
        self.mix_box = float(mix_box)
        self.mix_shortage = float(mix_shortage)
        self.box_state_sampler = box_state_sampler
        self.shortage_state_sampler = shortage_state_sampler
        self.reference_transition = reference_transition
        self.reference_label_sampler = reference_label_sampler
        self.rollout_state_pool_by_time = None
        self.rollout_label_pool_by_time = None
        self.rollout_path_id_pool_by_time = None
        self.lognormal_proposal_params_by_time = None
        self.truncated_lognormal_params_by_time = None
        self.uniform_z_radius_mult = float(uniform_z_radius_mult)
        self.uniform_q_radius_frac = float(uniform_q_radius_frac)
        self.total_inventory_buffer = float(total_inventory_buffer)
        self.q_low_values = tuple(float(x) for x in (q_low_values or (0.0, 1.0, 5.0, 10.0, 25.0, 50.0)))
        self.boundary_eps_frac = float(boundary_eps_frac)
        self.lognormal_delta_z = float(lognormal_delta_z)
        self.lognormal_delta_q = float(lognormal_delta_q)
        self.lognormal_r_min = int(lognormal_r_min)
        self.initial_state_mode = str(initial_state_mode or "standard")
        self.initial_perturb_prob = float(initial_perturb_prob)
        self.initial_q_perturb_frac = float(initial_q_perturb_frac)
        self.initial_z_perturb_frac = float(initial_z_perturb_frac)
        self.initial_perturb_minus_prob = float(initial_perturb_minus_prob)
        self.initial_perturb_plus_prob = float(initial_perturb_plus_prob)
        self.initial_perturb_minus_frac = float(initial_perturb_minus_frac)
        self.initial_perturb_plus_frac = float(initial_perturb_plus_frac)
        self.initial_dirichlet_alpha = float(initial_dirichlet_alpha)
        self.initial_state_component_labels = None
        self.initial_state_summary = {}

        if self.sampling_mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported sampling_mode='{self.sampling_mode}'. "
                f"Expected one of {sorted(self.SUPPORTED_MODES)}."
            )
        if self.initial_state_mode not in self.SUPPORTED_INITIAL_STATE_MODES:
            raise ValueError(
                f"Unsupported initial_state_mode='{self.initial_state_mode}'. "
                f"Expected one of {sorted(self.SUPPORTED_INITIAL_STATE_MODES)}."
            )
        if self.proposal_sampler is not None and self.proposal_sampler not in self.SUPPORTED_PROPOSALS:
            raise ValueError(
                f"Unsupported proposal_sampler='{self.proposal_sampler}'. "
                f"Expected one of {sorted(self.SUPPORTED_PROPOSALS)}."
            )
        if self.rollout_pool_size <= 0:
            raise ValueError("rollout_pool_size must be positive.")
        if self.uniform_z_radius_mult < 0.0:
            raise ValueError("uniform_z_radius_mult must be nonnegative.")
        if self.uniform_q_radius_frac < 0.0:
            raise ValueError("uniform_q_radius_frac must be nonnegative.")
        if self.total_inventory_buffer < 0.0:
            raise ValueError("total_inventory_buffer must be nonnegative.")
        if self.boundary_eps_frac < 0.0:
            raise ValueError("boundary_eps_frac must be nonnegative.")
        for name, value in (
            ("mix_rollout", self.mix_rollout),
            ("mix_box", self.mix_box),
            ("mix_shortage", self.mix_shortage),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
        if self.lognormal_delta_z < 0.0:
            raise ValueError("lognormal_delta_z must be nonnegative.")
        if self.lognormal_delta_q < 0.0:
            raise ValueError("lognormal_delta_q must be nonnegative.")
        if self.lognormal_r_min <= 0:
            raise ValueError("lognormal_r_min must be positive.")
        for name, value in (
            ("initial_perturb_prob", self.initial_perturb_prob),
            ("initial_q_perturb_frac", self.initial_q_perturb_frac),
            ("initial_z_perturb_frac", self.initial_z_perturb_frac),
            ("initial_perturb_minus_prob", self.initial_perturb_minus_prob),
            ("initial_perturb_plus_prob", self.initial_perturb_plus_prob),
            ("initial_perturb_minus_frac", self.initial_perturb_minus_frac),
            ("initial_perturb_plus_frac", self.initial_perturb_plus_frac),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative.")
        if self.initial_perturb_prob > 1.0 + 1e-12:
            raise ValueError("initial_perturb_prob must lie in [0, 1].")
        if self.initial_perturb_minus_prob + self.initial_perturb_plus_prob > 1.0 + 1e-12:
            raise ValueError("initial perturbation probabilities must sum to at most 1.")
        if self.initial_dirichlet_alpha <= 0.0:
            raise ValueError("initial_dirichlet_alpha must be positive.")
        mix_sum = self.mix_rollout + self.mix_box + self.mix_shortage
        if abs(mix_sum - 1.0) > 1e-8:
            raise ValueError("mix_rollout + mix_box + mix_shortage must sum to 1.")

        self.generator = torch.Generator(device="cpu")
        if sampling_seed is not None:
            self.generator.manual_seed(int(sampling_seed))

    def sample_states(self, n, batch_size, mode=None, return_metadata=False):
        if mode is None and self.proposal_sampler is not None:
            states, metadata = self._sample_proposal_states(n, batch_size)
            if return_metadata:
                return states, metadata
            return states

        mode = self._normalize_mode_name(mode or self.sampling_mode)
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported sampling_mode='{mode}'. "
                f"Expected one of {sorted(self.SUPPORTED_MODES)}."
            )

        if mode == "box":
            states = self.box_state_sampler(batch_size, n)
            metadata = {
                "mode": "box",
                "box_count": int(batch_size),
                "rollout_count": 0,
                "shortage_count": 0,
            }
        elif mode == "rollout":
            states, metadata = self._sample_rollout_states(n, batch_size)
        elif mode == "shortage":
            states = self.shortage_state_sampler(batch_size, n)
            metadata = {
                "mode": "shortage",
                "box_count": 0,
                "rollout_count": 0,
                "shortage_count": int(batch_size),
            }
        elif mode == "lognormal_proposal":
            states, metadata = self._sample_lognormal_proposal_states(n, batch_size)
        else:
            states, metadata = self._sample_mixture_states(n, batch_size)

        if return_metadata:
            return states, metadata
        return states

    def get_rollout_state_pool(self, n=None):
        self._ensure_rollout_state_pool()
        if n is None:
            return self.rollout_state_pool_by_time
        return self.rollout_state_pool_by_time[n]

    def get_rollout_metadata_pool(self, n=None):
        self._ensure_rollout_state_pool()
        if n is None:
            return {
                "reference_policy_labels": self.rollout_label_pool_by_time,
                "path_ids": self.rollout_path_id_pool_by_time,
            }
        return {
            "reference_policy_labels": self.rollout_label_pool_by_time[n],
            "path_ids": self.rollout_path_id_pool_by_time[n],
        }

    def _sample_rollout_states(self, n, batch_size):
        self._ensure_rollout_state_pool()
        state_pool = self.rollout_state_pool_by_time[n]
        indices_cpu = self._randint(state_pool.shape[0], (batch_size,))
        indices = indices_cpu.to(device=state_pool.device)
        states = state_pool[indices].clone()
        metadata = {
            "mode": "rollout",
            "box_count": 0,
            "rollout_count": int(batch_size),
            "shortage_count": 0,
            "indices": indices_cpu.clone(),
        }
        if self.rollout_label_pool_by_time is not None:
            metadata["reference_policy_labels"] = self.rollout_label_pool_by_time[n][indices].detach().cpu()
        if self.rollout_path_id_pool_by_time is not None:
            metadata["path_ids"] = self.rollout_path_id_pool_by_time[n][indices].detach().cpu()
        return states, metadata

    def _sample_mixture_states(self, n, batch_size):
        rollout_count, box_count, shortage_count = self._split_batch_size(batch_size)

        states = []
        source_labels = []
        rollout_indices = None

        if rollout_count > 0:
            rollout_states, rollout_meta = self._sample_rollout_states(n, rollout_count)
            rollout_indices = rollout_meta["indices"]
            states.append(rollout_states)
            source_labels.append(
                torch.full((rollout_count,), 1, dtype=torch.int64, device=self.device)
            )
            rollout_reference_labels = rollout_meta.get("reference_policy_labels")
            rollout_path_ids = rollout_meta.get("path_ids")
        else:
            rollout_reference_labels = None
            rollout_path_ids = None

        if box_count > 0:
            box_states = self.box_state_sampler(box_count, n)
            states.append(box_states)
            source_labels.append(
                torch.full((box_count,), 0, dtype=torch.int64, device=self.device)
            )

        if shortage_count > 0:
            shortage_states = self.shortage_state_sampler(shortage_count, n)
            states.append(shortage_states)
            source_labels.append(
                torch.full((shortage_count,), 2, dtype=torch.int64, device=self.device)
            )

        combined_states = torch.cat(states, dim=0)
        combined_sources = torch.cat(source_labels, dim=0)

        permutation_cpu = self._randperm(batch_size)
        permutation = permutation_cpu.to(device=self.device)
        combined_states = combined_states[permutation]
        combined_sources = combined_sources[permutation]

        metadata = {
            "mode": "mixture",
            "box_count": box_count,
            "rollout_count": rollout_count,
            "shortage_count": shortage_count,
            "source_labels": combined_sources.detach().cpu(),
            "permutation": permutation_cpu,
        }
        if rollout_indices is not None:
            metadata["rollout_indices"] = rollout_indices
        if rollout_reference_labels is not None:
            default_labels = torch.full((box_count + shortage_count,), -1, dtype=torch.int64)
            combined_reference_labels = torch.cat([rollout_reference_labels, default_labels], dim=0)
            metadata["reference_policy_labels"] = combined_reference_labels[permutation_cpu]
        if rollout_path_ids is not None:
            default_path_ids = torch.full((box_count + shortage_count,), -1, dtype=torch.int64)
            combined_path_ids = torch.cat([rollout_path_ids, default_path_ids], dim=0)
            metadata["path_ids"] = combined_path_ids[permutation_cpu]

        return combined_states, metadata

    def _split_batch_size(self, batch_size):
        weights = (
            ("rollout", self.mix_rollout),
            ("box", self.mix_box),
            ("shortage", self.mix_shortage),
        )
        raw_counts = [batch_size * weight for _, weight in weights]
        counts = [int(count) for count in raw_counts]
        remainder = int(batch_size) - sum(counts)

        if remainder > 0:
            fractional_parts = [
                (raw_counts[idx] - counts[idx], idx) for idx in range(len(weights))
            ]
            fractional_parts.sort(key=lambda item: (-item[0], item[1]))
            for _, idx in fractional_parts[:remainder]:
                counts[idx] += 1

        return tuple(counts)

    def _ensure_rollout_state_pool(self):
        if self.rollout_state_pool_by_time is None:
            self.rollout_state_pool_by_time = self._build_rollout_state_pool()

    def _ensure_lognormal_proposal_params(self):
        if self.lognormal_proposal_params_by_time is None:
            self._ensure_rollout_state_pool()
            self.lognormal_proposal_params_by_time = self._build_lognormal_proposal_params()

    def _build_rollout_state_pool(self):
        state_pool_by_time = []
        label_pool_by_time = []
        path_id_pool_by_time = []
        current_states = self._sample_initial_rollout_states()
        current_labels = self._sample_initial_reference_labels()
        path_ids = torch.arange(self.rollout_pool_size, dtype=torch.int64, device=self.device)

        for n in range(self.num_time_interval):
            state_pool_by_time.append(current_states.clone())
            label_pool_by_time.append(current_labels.clone())
            path_id_pool_by_time.append(path_ids.clone())
            next_states, _, _, _ = self.reference_transition(
                n,
                current_states,
                generator=self.generator,
                reference_policy_labels=current_labels,
                path_ids=path_ids,
            )
            current_states = next_states

        self.rollout_label_pool_by_time = tuple(label_pool_by_time)
        self.rollout_path_id_pool_by_time = tuple(path_id_pool_by_time)
        return tuple(state_pool_by_time)

    def _sample_initial_reference_labels(self):
        if self.reference_label_sampler is None:
            return torch.zeros(self.rollout_pool_size, dtype=torch.int64, device=self.device)
        labels = self.reference_label_sampler(self.rollout_pool_size)
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.int64)
        labels = labels.to(device=self.device, dtype=torch.int64).reshape(-1)
        if labels.numel() != self.rollout_pool_size:
            raise ValueError(
                "reference_label_sampler returned "
                f"{labels.numel()} labels for rollout_pool_size={self.rollout_pool_size}."
            )
        return labels

    def _sample_initial_rollout_states(self):
        dtype = self.bsde.mu.dtype
        states = torch.zeros(
            self.rollout_pool_size,
            self.state_dim,
            device=self.device,
            dtype=dtype,
        )
        states[:, self.dim] = float(self.bsde.I0)
        labels = torch.zeros(self.rollout_pool_size, dtype=torch.int64, device="cpu")

        if self.initial_state_mode == "paper_70_30":
            draws = torch.rand(
                (self.rollout_pool_size,),
                generator=self.generator,
                device="cpu",
                dtype=dtype,
            )
            perturb_mask = draws < self.initial_perturb_prob
            perturb_count = int(perturb_mask.sum().item())
            if perturb_count:
                q0 = float(self.bsde.I0)
                b_q = (
                    torch.rand((perturb_count,), generator=self.generator, device="cpu", dtype=dtype)
                    * 2.0
                    - 1.0
                ) * (self.initial_q_perturb_frac * q0)
                z_vals = torch.rand(
                    (perturb_count, self.dim),
                    generator=self.generator,
                    device="cpu",
                    dtype=dtype,
                ) * (self.initial_z_perturb_frac * q0)
                states[perturb_mask.to(device=self.device), : self.dim] = z_vals.to(device=self.device)
                states[perturb_mask.to(device=self.device), self.dim] = q0 + b_q.to(device=self.device)
                labels[perturb_mask] = 3

        elif self.initial_state_mode == "q_sensitivity_mixture":
            draws = torch.rand(
                (self.rollout_pool_size,),
                generator=self.generator,
                device="cpu",
                dtype=dtype,
            )
            redistribute_mask = draws < self.initial_perturb_minus_prob
            q_plus_mask = (
                (draws >= self.initial_perturb_minus_prob)
                & (draws < self.initial_perturb_minus_prob + self.initial_perturb_plus_prob)
            )

            redistribute_count = int(redistribute_mask.sum().item())
            if redistribute_count:
                b_minus = torch.rand(
                    (redistribute_count,),
                    generator=self.generator,
                    device="cpu",
                    dtype=dtype,
                ) * (self.initial_perturb_minus_frac * float(self.bsde.I0))
                weights = self._sample_dirichlet_weights(redistribute_count, dtype)
                states[redistribute_mask.to(device=self.device), : self.dim] = (
                    b_minus.to(device=self.device).unsqueeze(1)
                    * weights.to(device=self.device)
                )
                states[redistribute_mask.to(device=self.device), self.dim] = (
                    float(self.bsde.I0) - b_minus.to(device=self.device)
                )
                labels[redistribute_mask] = 1

            q_plus_count = int(q_plus_mask.sum().item())
            if q_plus_count:
                b_plus = torch.rand(
                    (q_plus_count,),
                    generator=self.generator,
                    device="cpu",
                    dtype=dtype,
                ) * (self.initial_perturb_plus_frac * float(self.bsde.I0))
                states[q_plus_mask.to(device=self.device), : self.dim] = 0.0
                states[q_plus_mask.to(device=self.device), self.dim] = (
                    float(self.bsde.I0) + b_plus.to(device=self.device)
                )
                labels[q_plus_mask] = 2

        self.initial_state_component_labels = labels
        self.initial_state_summary = self._summarize_initial_states(states, labels)
        return states

    def _sample_dirichlet_weights(self, count, dtype):
        concentration = torch.full(
            (int(count), self.dim),
            float(self.initial_dirichlet_alpha),
            device="cpu",
            dtype=dtype,
        )
        try:
            gamma_draws = torch._standard_gamma(concentration, generator=self.generator)
        except TypeError:  # pragma: no cover - compatibility for older torch builds
            gamma_draws = torch._standard_gamma(concentration)
        gamma_draws = torch.clamp(gamma_draws, min=1e-12)
        return gamma_draws / gamma_draws.sum(dim=1, keepdim=True)

    def _summarize_initial_states(self, states, labels):
        states_cpu = states.detach().cpu().double()
        q = states_cpu[:, self.dim]
        local = states_cpu[:, : self.dim]
        total = states_cpu.sum(dim=1)
        count = max(int(states_cpu.shape[0]), 1)
        return {
            "initial_state_mode": self.initial_state_mode,
            "initial_q_mean": float(q.mean().item()),
            "initial_q_p05": float(torch.quantile(q, 0.05).item()),
            "initial_q_p50": float(torch.quantile(q, 0.50).item()),
            "initial_q_p95": float(torch.quantile(q, 0.95).item()),
            "initial_q_above_q0_share": float((q > float(self.bsde.I0) + 1e-10).double().mean().item()),
            "initial_standard_share": float((labels == 0).sum().item()) / float(count),
            "initial_redistribute_share": float((labels == 1).sum().item()) / float(count),
            "initial_q_plus_share": float((labels == 2).sum().item()) / float(count),
            "initial_paper_perturb_share": float((labels == 3).sum().item()) / float(count),
            "initial_local_inventory_mean_by_location": [
                float(value) for value in local.mean(dim=0).tolist()
            ],
            "initial_total_inventory_mean": float(total.mean().item()),
            "initial_total_inventory_p95": float(torch.quantile(total, 0.95).item()),
            "initial_perturb_minus_prob": self.initial_perturb_minus_prob,
            "initial_perturb_plus_prob": self.initial_perturb_plus_prob,
            "initial_perturb_minus_frac": self.initial_perturb_minus_frac,
            "initial_perturb_plus_frac": self.initial_perturb_plus_frac,
            "initial_perturb_prob": self.initial_perturb_prob,
            "initial_q_perturb_frac": self.initial_q_perturb_frac,
            "initial_z_perturb_frac": self.initial_z_perturb_frac,
            "initial_dirichlet_alpha": self.initial_dirichlet_alpha,
        }

    def _build_lognormal_proposal_params(self):
        thresholds = self._lognormal_fallback_thresholds()
        params_by_time = []

        for n in range(self.num_time_interval):
            state_pool = self.rollout_state_pool_by_time[n].detach().cpu()
            coord_params = []
            uniform_count = 0
            lognormal_count = 0

            for coord in range(self.state_dim):
                threshold = float(thresholds[coord].item())
                positive_values = state_pool[:, coord]
                positive_values = positive_values[positive_values > threshold]

                if positive_values.numel() < self.lognormal_r_min:
                    upper = max(threshold, 1e-12)
                    coord_params.append(("uniform", upper))
                    uniform_count += 1
                    continue

                log_values = torch.log(torch.clamp(positive_values.double(), min=1e-12))
                loc = float(log_values.mean().item())
                scale = float(log_values.std(unbiased=False).item())
                if not torch.isfinite(torch.tensor(scale)) or scale < 1e-6:
                    scale = 1e-6
                coord_params.append(("lognormal", loc, scale))
                lognormal_count += 1

            params_by_time.append(
                {
                    "coord_params": tuple(coord_params),
                    "uniform_count": uniform_count,
                    "lognormal_count": lognormal_count,
                }
            )

        return tuple(params_by_time)

    def _sample_proposal_states(self, n, batch_size):
        proposal = self.proposal_sampler
        if proposal == "empirical":
            states, metadata = self._sample_rollout_states(n, batch_size)
            metadata["mode"] = "empirical"
            metadata["proposal_sampler"] = "empirical"
            metadata["empirical_count"] = int(batch_size)
            return states, metadata
        if proposal == "uniform_tube":
            return self._sample_uniform_tube_states(n, batch_size)
        if proposal == "mixture_uniform_lowq_boundary":
            return self._sample_mixture_uniform_lowq_boundary_states(n, batch_size)
        if proposal == "truncated_lognormal":
            return self._sample_truncated_lognormal_states(n, batch_size)
        raise ValueError(f"Unsupported proposal_sampler='{proposal}'.")

    def _sample_uniform_tube_states(self, n, batch_size):
        states, _ = self._sample_rollout_states(n, batch_size)
        dim = self.dim
        dtype = states.dtype
        device = states.device

        radius_z = self._uniform_z_radius(n).to(device=device, dtype=dtype)
        radius_q = float(self.uniform_q_radius_frac) * float(self.bsde.I0)

        rand_z = torch.rand(
            (int(batch_size), dim),
            generator=self.generator,
            device="cpu",
            dtype=dtype,
        ).to(device=device)
        rand_q = torch.rand(
            (int(batch_size), 1),
            generator=self.generator,
            device="cpu",
            dtype=dtype,
        ).to(device=device)

        states = states.clone()
        q_before = states[:, dim].clone()
        states[:, :dim] = states[:, :dim] + (2.0 * rand_z - 1.0) * radius_z.unsqueeze(0)
        states[:, dim:dim + 1] = states[:, dim:dim + 1] + (2.0 * rand_q - 1.0) * radius_q
        states, cap_meta = self._enforce_physical_caps(states)

        metadata = {
            "mode": "uniform_tube",
            "proposal_sampler": "uniform_tube",
            "uniform_tube_count": int(batch_size),
            "q_changed_count": int(torch.sum(torch.abs(states[:, dim] - q_before) > 1e-10).item()),
            "total_cap_projection_count": int(cap_meta["total_cap_projection_count"]),
        }
        return states, metadata

    def _sample_mixture_uniform_lowq_boundary_states(self, n, batch_size):
        empirical_count, uniform_count, low_q_count, boundary_count = self._split_counts(
            int(batch_size), (0.60, 0.25, 0.10, 0.05)
        )
        parts = []
        labels = []
        projection_count = 0

        if empirical_count:
            empirical, _ = self._sample_rollout_states(n, empirical_count)
            empirical, meta = self._enforce_physical_caps(empirical)
            projection_count += int(meta["total_cap_projection_count"])
            parts.append(empirical)
            labels.append(torch.full((empirical_count,), 0, dtype=torch.int64, device=self.device))

        if uniform_count:
            uniform, uniform_meta = self._sample_uniform_tube_states(n, uniform_count)
            projection_count += int(uniform_meta.get("total_cap_projection_count", 0))
            parts.append(uniform)
            labels.append(torch.full((uniform_count,), 1, dtype=torch.int64, device=self.device))

        if low_q_count:
            low_q, _ = self._sample_rollout_states(n, low_q_count)
            low_q = low_q.clone()
            low_q[:, self.dim] = self._sample_low_q_values(low_q_count, low_q.dtype, low_q.device)
            low_q, meta = self._enforce_physical_caps(low_q)
            projection_count += int(meta["total_cap_projection_count"])
            parts.append(low_q)
            labels.append(torch.full((low_q_count,), 2, dtype=torch.int64, device=self.device))

        if boundary_count:
            boundary = self._sample_boundary_replacement_states(n, boundary_count)
            boundary, meta = self._enforce_physical_caps(boundary)
            projection_count += int(meta["total_cap_projection_count"])
            parts.append(boundary)
            labels.append(torch.full((boundary_count,), 3, dtype=torch.int64, device=self.device))

        states = torch.cat(parts, dim=0)
        source_labels = torch.cat(labels, dim=0)
        permutation_cpu = self._randperm(int(batch_size))
        permutation = permutation_cpu.to(device=self.device)
        states = states[permutation]
        source_labels = source_labels[permutation]

        metadata = {
            "mode": "mixture_uniform_lowq_boundary",
            "proposal_sampler": "mixture_uniform_lowq_boundary",
            "empirical_count": empirical_count,
            "uniform_tube_count": uniform_count,
            "low_q_replacement_count": low_q_count,
            "boundary_replacement_count": boundary_count,
            "total_cap_projection_count": projection_count,
            "source_labels": source_labels.detach().cpu(),
            "permutation": permutation_cpu,
        }
        return states, metadata

    def _sample_truncated_lognormal_states(self, n, batch_size):
        self._ensure_truncated_lognormal_params()
        params = self.truncated_lognormal_params_by_time[n]
        dtype = self.bsde.mu.detach().cpu().dtype
        samples = []
        tail_rejections = 0

        for param in params["coord_params"]:
            kind = param[0]
            if kind == "uniform":
                sample = torch.rand(
                    (int(batch_size),),
                    generator=self.generator,
                    device="cpu",
                    dtype=dtype,
                ) * float(param[1])
            else:
                _, loc, scale, cap = param
                sample = self._lognormal_cpu(int(batch_size), loc, scale, dtype)
                reject_mask = sample > float(cap)
                attempts = 0
                while bool(reject_mask.any()) and attempts < 50:
                    reject_count = int(reject_mask.sum().item())
                    tail_rejections += reject_count
                    sample[reject_mask] = self._lognormal_cpu(reject_count, loc, scale, dtype)
                    reject_mask = sample > float(cap)
                    attempts += 1
                if bool(reject_mask.any()):
                    tail_rejections += int(reject_mask.sum().item())
                    sample[reject_mask] = float(cap)
            samples.append(sample)

        states = torch.stack(samples, dim=1).to(device=self.device)
        states, cap_meta = self._enforce_physical_caps(states)
        metadata = {
            "mode": "truncated_lognormal",
            "proposal_sampler": "truncated_lognormal",
            "truncated_lognormal_count": int(batch_size),
            "tail_rejection_count": int(tail_rejections),
            "total_cap_projection_count": int(cap_meta["total_cap_projection_count"]),
        }
        return states, metadata

    def _ensure_truncated_lognormal_params(self):
        if self.truncated_lognormal_params_by_time is None:
            self._ensure_rollout_state_pool()
            self.truncated_lognormal_params_by_time = self._build_truncated_lognormal_params()

    def _build_truncated_lognormal_params(self):
        params_by_time = []
        for n in range(self.num_time_interval):
            state_pool = self.rollout_state_pool_by_time[n].detach().cpu()
            coord_params = []
            dim = state_pool.shape[1] - 1
            for coord in range(state_pool.shape[1]):
                values = state_pool[:, coord].double()
                positive = values[values > 1e-10]
                empirical_q99 = float(torch.quantile(values, 0.99).item()) if values.numel() else 0.0
                coord_cap = min(
                    max(empirical_q99, 1e-10),
                    float(self.bsde.I0) if coord == dim else self._physical_cap(),
                )
                if positive.numel() < 2 or coord_cap <= 1e-10:
                    coord_params.append(("uniform", coord_cap))
                    continue
                log_values = torch.log(torch.clamp(positive, min=1e-12))
                loc = float(log_values.mean().item())
                scale = max(float(log_values.std(unbiased=False).item()), 1e-6)
                coord_params.append(("lognormal", loc, scale, coord_cap))
            params_by_time.append({"coord_params": tuple(coord_params)})
        return tuple(params_by_time)

    def _lognormal_cpu(self, count, loc, scale, dtype):
        normal = torch.randn(
            (int(count),),
            generator=self.generator,
            device="cpu",
            dtype=dtype,
        )
        return torch.exp(normal * float(scale) + float(loc))

    def _eta_at_fine_index_cpu(self, n):
        if hasattr(self.bsde, "eta_at_fine_index"):
            return self.bsde.eta_at_fine_index(n).detach().cpu().double()
        return self.bsde.eta.detach().cpu().double()

    def _uniform_z_radius(self, n):
        review_length = float(
            getattr(
                self.bsde,
                "review_length",
                getattr(self.bsde, "raw_delta_t", self.bsde.delta_t),
            )
        )
        eta_dt = self._eta_at_fine_index_cpu(n) * review_length
        mu_t = self.bsde.mu[n, :].detach().cpu().double()
        radius = torch.maximum(torch.ones_like(eta_dt), torch.maximum(eta_dt, mu_t))
        return float(self.uniform_z_radius_mult) * radius

    def _boundary_eps(self, n):
        review_length = float(
            getattr(
                self.bsde,
                "review_length",
                getattr(self.bsde, "raw_delta_t", self.bsde.delta_t),
            )
        )
        eta_dt = self._eta_at_fine_index_cpu(n) * review_length
        mu_t = self.bsde.mu[n, :].detach().cpu().double()
        base = torch.maximum(torch.ones_like(eta_dt), torch.maximum(eta_dt, mu_t))
        return float(self.boundary_eps_frac) * base

    def _sample_low_q_values(self, count, dtype, device):
        values = torch.tensor(self.q_low_values, dtype=dtype, device="cpu")
        if values.numel() == 0:
            upper = min(float(self.bsde.I0), max(1.0, 0.1 * float(self.bsde.I0)))
            out = torch.rand((int(count),), generator=self.generator, device="cpu", dtype=dtype) * upper
        else:
            indices = self._randint(values.numel(), (int(count),))
            out = values[indices].clamp(0.0, float(self.bsde.I0))
        return out.to(device=device)

    def _sample_boundary_replacement_states(self, n, count):
        states, _ = self._sample_rollout_states(n, count)
        states = states.clone()
        eps = self._boundary_eps(n).to(device=states.device, dtype=states.dtype)
        for row in range(int(count)):
            n_coords = int(self._randint(self.dim, (1,)).item()) + 1
            coords_cpu = self._randperm(self.dim)[:n_coords]
            coords = coords_cpu.to(device=states.device)
            random_values = torch.rand(
                (n_coords,),
                generator=self.generator,
                device="cpu",
                dtype=states.dtype,
            ).to(device=states.device)
            states[row, coords] = random_values * eps[coords]
        return states

    def _enforce_physical_caps(self, states):
        capped = states.clone()
        dim = self.dim
        capped[:, :dim] = torch.clamp(capped[:, :dim], min=0.0)
        capped[:, dim] = torch.clamp(capped[:, dim], min=0.0, max=float(self.bsde.I0))

        total_cap = self._physical_cap()
        totals = capped.sum(dim=1)
        violation_mask = totals > total_cap + 1e-10
        violation_count = int(violation_mask.sum().item())
        if violation_count:
            factors = total_cap / torch.clamp(totals[violation_mask], min=1e-12)
            capped[violation_mask] = capped[violation_mask] * factors.unsqueeze(1)
            capped[:, dim] = torch.clamp(capped[:, dim], min=0.0, max=float(self.bsde.I0))
        return capped, {"total_cap_projection_count": violation_count}

    def _physical_cap(self):
        return float(self.bsde.I0) + float(self.total_inventory_buffer)

    @staticmethod
    def _split_counts(total, weights):
        raw = [float(total) * float(weight) for weight in weights]
        counts = [int(value) for value in raw]
        remainder = int(total) - sum(counts)
        order = sorted(
            ((raw[idx] - counts[idx], idx) for idx in range(len(weights))),
            key=lambda item: (-item[0], item[1]),
        )
        for _, idx in order[:remainder]:
            counts[idx] += 1
        return tuple(counts)

    def _sample_lognormal_proposal_states(self, n, batch_size):
        self._ensure_lognormal_proposal_params()
        params = self.lognormal_proposal_params_by_time[n]
        dtype = self.bsde.mu.detach().cpu().dtype
        samples = []

        for param in params["coord_params"]:
            if param[0] == "uniform":
                sample = torch.rand(
                    (batch_size,),
                    generator=self.generator,
                    device="cpu",
                    dtype=dtype,
                ) * float(param[1])
            else:
                _, loc, scale = param
                normal = torch.randn(
                    (batch_size,),
                    generator=self.generator,
                    device="cpu",
                    dtype=dtype,
                )
                sample = torch.exp(normal * float(scale) + float(loc))
            samples.append(sample)

        states = torch.stack(samples, dim=1).to(device=self.device)
        metadata = {
            "mode": "lognormal_proposal",
            "box_count": 0,
            "rollout_count": 0,
            "shortage_count": 0,
            "uniform_coord_count": params["uniform_count"],
            "lognormal_coord_count": params["lognormal_count"],
        }
        return states, metadata

    def _lognormal_fallback_thresholds(self):
        if hasattr(self.bsde, "eta_schedule"):
            eta = self.bsde.eta_schedule.detach().cpu().double().max(dim=0).values
        else:
            eta = self.bsde.eta.detach().cpu().double()
        delta_t = float(self.bsde.delta_t)
        z_eps = torch.clamp(self.lognormal_delta_z * eta * delta_t, min=1e-12)
        q_eps = max(self.lognormal_delta_q * delta_t * float(eta.sum().item()), 1e-12)
        return torch.cat([z_eps, torch.tensor([q_eps], dtype=torch.float64)])

    def _randint(self, high, size):
        return torch.randint(high, size, generator=self.generator, device="cpu")

    def _randperm(self, n):
        return torch.randperm(n, generator=self.generator, device="cpu")

    @staticmethod
    def _normalize_mode_name(mode):
        if mode == "mixed":
            return "mixture"
        if mode in {"lognormal", "proposal"}:
            return "lognormal_proposal"
        return mode

    @staticmethod
    def _normalize_proposal_name(proposal):
        if proposal in (None, ""):
            return None
        if proposal == "rollout":
            return "empirical"
        if proposal == "lognormal":
            return "truncated_lognormal"
        return proposal
