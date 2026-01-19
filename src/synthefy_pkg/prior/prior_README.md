## Core Configuration

- **`config_path`**: Optional path to YAML configuration file
  - *Suggested*: None (for programmatic config) or path to .yaml file
- **`n_jobs`**: Number of parallel jobs (intended only for SavePriorDataset access)
  - *Suggested*: 1 (default), 4-16 for parallel generation
- **`num_threads_per_generate`**: Number of threads per generation process
  - *Suggested*: 1 (default), 2-8 for CPU-intensive generation

## Dataset Generation Parameters

- **`run_id`**: Unique identifier for the current run
  - *Used in*: curriculum config file naming (curriculum_config_{run_id}.pkl)
  - *Suggested*: "default", "experiment_1", or descriptive name
- **`prior_dir`**: Directory path for storing prior datasets (None for on-the-fly generation)
  - *Used in*: genload.py for dataset persistence
  - *Suggested*: None (on-the-fly) or "/path/to/prior/data/"
- **`batch_size`**: Batch size for training (should be overridden by training config)
  - *Used in*: PriorDataset.__len__() and dataset generation loops
  - *Suggested*: 512 (default), 256-1024 depending on memory
- **`dataset_length`**: Total number of tables to generate (preferably multiple of workers)
  - *Used in*: PriorDataset.__len__() calculation with world_size division
  - *Suggested*: 5000 (default), 1024-10000 for different dataset sizes
- **`load_prior_start`**: Starting index for loading prior datasets
  - *Suggested*: 0 (default)
- **`delete_after_load`**: Whether to delete datasets after loading
  - *Suggested*: False (default), True for memory-constrained environments
- **`batch_size_per_gp`**: Batch size per generation process
  - *Used in*: genload.py for calculating num_gps and size_per_gp
  - *Related to*: batch_size (used together for parallel generation)
  - *Suggested*: 4 (default), 2-8 for parallel generation
- **`min_features/max_features`**: Range for number of features in generated datasets
  - *Used in*: dataset.py for feature count sampling
  - *Related to*: max_features must match num_correlates - 1 in dataset_config
  - *Suggested*: min_features=5, max_features=45-100 (must match dataset config)
- **`n_features`**: Default number of features
  - *Suggested*: 5 (default)
- **`max_classes`**: Maximum number of classes for classification tasks
  - *Suggested*: 10 (default), 2-20 for different classification complexity
- **`min_seq_len/max_seq_len`**: Range for sequence length in time series data
  - *Used in*: dataset.py for sequence length sampling
  - *Suggested*: min_seq_len=None, max_seq_len=768-1024 (must match dataset config)
- **`log_seq_len`**: Whether to use logarithmic scaling for sequence length
  - *Suggested*: False (default)
- **`seq_len_per_gp`**: Whether to set sequence length per generation process
  - *Suggested*: False (default)
- **`min_train_size/max_train_size`**: Range for training set size (as fraction or absolute)
  - *Used in*: curriculum learning for adaptive training size adjustment
  - *Suggested*: min_train_size=0.1-0.5, max_train_size=0.8-0.9
- **`replay_small`**: Whether to replay small datasets
  - *Suggested*: False (default)
- **`prior_type`**: Type of prior model ("mlp_scm", "tree_scm", "mix_scm", "dummy")
  - *Used in*: dataset.py._get_prior_class() and PriorDataset initialization
  - *Related to*: scm_mix_probs (when prior_type="mix_scm")
  - *Suggested*: "mix_scm" (default), "mlp_scm" for MLP-only, "tree_scm" for tree-only
- **`is_regression`**: Whether to generate regression datasets
  - *Suggested*: True (default), False for classification
- **`tabular_dataset_rate`**: Probability of creating tabular (non-time-series) datasets
  - *Used in*: dataset.py for dataset_is_tabular sampling
  - *Suggested*: 0.0 (default), 0.1-0.5 for mixed tabular/time-series
- **`dataset_has_timestamp_rate`**: Probability of adding timestamps to series datasets
  - *Suggested*: 1.0 (default), 0.5-1.0 for mixed timestamp usage

## Timestamp Configuration

- **`add_synthetic_timestamps`**: List of timestamp types to add ("minutely", "hourly", "daily", "monthly")
  - *Used in*: dataset.py.add_synthetic_timestamps() and single_mlp_generate.py
  - *Related to*: add_time_stamps_as_features, max_features, max_seq_len
  - *Suggested*: ["minutely", "hourly", "daily", "weekly", "monthly"] (default), [] for no timestamps
- **`add_time_stamps_as_features`**: Whether to include timestamps as features
  - *Used in*: dataset.py for conditional timestamp feature generation
  - *Suggested*: False (default), True for timestamp-derived features
- **`add_both_time_stamps_and_features`**: Whether to add both timestamps and derived features
  - *Suggested*: False (default)

## Curriculum Learning

- **`check_for_updates_freq`**: Frequency to check for curriculum updates (-1 to disable)
  - *Used in*: PriorDataset.__next__() for periodic config updates
  - *Related to*: check_for_curriculum_config
  - *Suggested*: -1 (disabled), 1000-5000 for active curriculum
- **`check_for_curriculum_config`**: Whether to check for curriculum configuration changes
  - *Used in*: PriorDataset._check_for_config_updates()
  - *Related to*: curriculum_config_path and run_id
  - *Suggested*: True (default), False for static configs

## Data Quality Parameters

- **`row_missing_prob`**: Probability of missing values per row
  - *Suggested*: 0.0 (default), 0.1-0.3 for realistic missing data
- **`column_has_missing_prob`**: Probability that a column contains missing values
  - *Suggested*: 0.0 (default), 0.1-0.5 for realistic missing data
- **`dataset_has_lag`**: Probability that dataset contains lag features
  - *Used in*: dataset.py for dataset_has_lag sampling
  - *Suggested*: 0.0-1.0 (default varies), 0.5-1.0 for time series with lags
- **`min_lag/max_lag`**: Range for lag values
  - *Used in*: dataset.py for lag value sampling
  - *Suggested*: min_lag=0-1, max_lag=0.1-0.2 (as fraction of sequence length)
- **`exclude_inputs`**: How to handle input exclusion ("allow", "exclude", "ensure")
  - *Suggested*: "allow" (default), "exclude" for strict separation, "ensure" for guaranteed exclusion
- **`univariate_prob`**: Probability of generating univariate datasets
  - *Suggested*: 0.0 (default), 0.1-0.3 for mixed univariate/multivariate
- **`num_flat_range`**: Range for number of flat features
  - *Suggested*: (7, 20) (default)
- **`flatten_prob`**: Probability of flattening features
  - *Suggested*: 0.0 (default), 0.1-0.5 for feature flattening
- **`respect_ancestry_for_lag`**: Whether to respect causal ancestry for lag features
  - *Suggested*: False (default), True for causal-aware lag generation
- **`use_input_as_target`**: Whether to use input features as targets
  - *Suggested*: False (default), True for input-target overlap

## SCM Prior Configuration - Fixed Parameters

- **`scm_use_layer_operator`**: Whether to use layer operators in SCM
  - *Used in*: config.get_scm_fixed_hp() method and mlp_scm.py LayerOperator initialization
  - *Related to*: All scm_kernel_*, scm_normalization_*, scm_functional_lag_* parameters (only used when True)
  - *Controls*: Whether LayerOperator instances are created for each MLP layer
  - *Suggested*: True (default), False for simpler models
- **`scm_mix_probs`**: Mix probabilities for MLP vs Tree SCM components
  - *Used in*: dataset.py.get_prior() when prior_type="mix_scm"
  - *Related to*: prior_type, scm_tree_model
  - *Suggested*: [0.7, 0.3] (default), [1.0, 0.0] for MLP-only, [0.0, 1.0] for tree-only
- **`scm_tree_model`**: Tree model type for TreeSCM ("xgboost")
  - *Used in*: config.get_scm_fixed_hp() method and tree_scm.py TreeSCM initialization
  - *Related to*: scm_mix_probs (only used when tree component is selected)
  - *Suggested*: "xgboost" (default), "lightgbm" for speed, "catboost" for categorical features
- **`scm_tree_depth_lambda`**: Lambda parameter for tree depth sampling
  - *Used in*: tree_scm.py for exponential distribution sampling of tree depth
  - *Related to*: scm_tree_n_estimators_lambda (both control tree complexity)
  - *Suggested*: 0.5 (default), 0.1-1.0 for different depth distributions
- **`scm_tree_n_estimators_lambda`**: Lambda parameter for number of estimators sampling
  - *Used in*: tree_scm.py for exponential distribution sampling of number of estimators
  - *Related to*: scm_tree_depth_lambda (both control tree complexity)
  - *Suggested*: 0.5 (default), 0.1-1.0 for different estimator distributions
- **`scm_balanced`**: Whether to balance classes in SCM
  - *Used in*: mlp_scm.py and tree_scm.py for class balancing in target generation
  - *Related to*: scm_multiclass_ordered_prob (affects multiclass problem structure)
  - *Suggested*: False (default), True for balanced datasets
- **`scm_multiclass_ordered_prob`**: Probability of ordered multiclass problems
  - *Used in*: mlp_scm.py and tree_scm.py for ordered vs unordered multiclass generation
  - *Related to*: scm_balanced (affects multiclass problem structure)
  - *Suggested*: 0.0 (default), 0.1-0.5 for ordered multiclass
- **`scm_cat_prob`**: Probability of categorical features
  - *Used in*: mlp_scm.py and tree_scm.py for categorical feature generation
  - *Related to*: scm_max_categories (affects categorical feature complexity)
  - *Suggested*: 0.2 (default), 0.1-0.5 for different categorical feature rates
- **`scm_max_categories`**: Maximum number of categories for categorical features
  - *Used in*: mlp_scm.py and tree_scm.py for categorical feature generation
  - *Related to*: scm_cat_prob (affects categorical feature complexity)
  - *Suggested*: float("inf") (default), 5-20 for limited categories
- **`scm_scale_by_max_features`**: Whether to scale by maximum features
  - *Used in*: mlp_scm.py for feature scaling in MLP initialization
  - *Related to*: max_features (affects scaling calculation)
  - *Suggested*: False (default), True for feature-scaled models
- **`scm_permute_features/labels`**: Whether to permute features/labels
  - *Used in*: mlp_scm.py for feature and label ordering in MLP inputs/outputs
  - *Related to*: Each other (affects both features and labels)
  - *Suggested*: True (default), False for deterministic ordering
- **`scm_override_activations`**: List of activation functions to override defaults
  - *Used in*: config.get_scm_sampled_hp() for activation override logic
  - *Related to*: scm_mlp_activations, scm_diverse_activation_names
  - *Suggested*: [] (default), ["identity", "relu", "tanh"] for specific activations
- **`scm_sigmoid_mixed_sampling_rate`**: Sampling rate for sigmoid mixed activations
  - *Used in*: mlp_scm.py for sigmoid activation sampling in mixed activation scenarios
  - *Related to*: scm_mlp_activations (affects sigmoid activation frequency)
  - *Suggested*: 7 (default), 1-10 for different sigmoid sampling rates
- **`scm_mixed_names`**: List of mixed function names for SCM generation
  - *Used in*: mlp_scm.py for mixed function selection in SCM generation
  - *Related to*: scm_sampling (affects available sampling methods)
  - *Suggested*: ["fourier", "wiener", "arima", "normal", "uniform"] (default)
- **`scm_diverse_activation_names`**: List of diverse activation function names
  - *Used in*: config.get_scm_sampled_hp() when scm_override_activations is set
  - *Related to*: scm_override_activations (used as fallback when override is empty)
  - *Suggested*: ["tanh", "relu", "elu", "silu", "gelu"] (default)

## SCM Layer Operator Parameters

- **`scm_normalization_type`**: Type of normalization ("none", "z_score", "min_max", "robust", "batch")
  - *Used in*: mlp_scm.py LayerOperator initialization and layer_operator.py.apply_normalization()
  - *Related to*: scm_normalization_minimum_magnitude, scm_normalization_maximum_magnitude, scm_normalization_apply_probability
  - *Suggested*: "none" (default), "z_score" for standardization, "min_max" for scaling
- **`scm_normalization_minimum_magnitude`**: Minimum magnitude for normalization
  - *Used in*: mlp_scm.py normalization_config creation for LayerOperator
  - *Related to*: scm_normalization_maximum_magnitude (used together for magnitude bounds)
  - *Suggested*: 0.0 (default), 0.1-1.0 for magnitude bounds
- **`scm_normalization_maximum_magnitude`**: Maximum magnitude for normalization
  - *Used in*: mlp_scm.py normalization_config creation for LayerOperator
  - *Related to*: scm_normalization_minimum_magnitude (used together for magnitude bounds)
  - *Suggested*: -1.0 (default), 1.0-20.0 for magnitude bounds
- **`scm_normalization_apply_probability`**: Probability of applying normalization
  - *Used in*: layer_operator.py.apply_normalization() for probabilistic normalization application
  - *Suggested*: 0.0 (default), 0.5-0.8 for probabilistic normalization

## SCM Kernel Parameters

- **`scm_kernel_direction`**: Direction of kernel application ("history", "future", "mixed")
  - *Used in*: layer_operator.py.SmoothingLagFilter initialization and kernel application logic
  - *Related to*: scm_kernel_size_min/max (affects kernel size calculation), scm_functional_lag_* (affects lag application)
  - *Suggested*: "history" (default), "future" for forward-looking, "mixed" for bidirectional
- **`scm_kernel_size_min/max`**: Distribution parameters for minimum/maximum kernel sizes
  - *Used in*: layer_operator.py.SmoothingLagFilter._init_filters() for random kernel size sampling
  - *Related to*: scm_kernel_direction (affects effective kernel size calculation)
  - *Suggested*: min=1, max=1 (default), min=3, max=5 for larger kernels
- **`scm_kernel_sigma`**: Distribution parameters for kernel sigma values
  - *Used in*: layer_operator.py.SmoothingLagFilter._create_gaussian_kernel() for Gaussian kernel creation
  - *Related to*: scm_kernel_type (only used when kernel_type="gaussian")
  - *Suggested*: 0.5 (default), 0.1-1.0 for different Gaussian widths
- **`scm_kernel_type`**: Distribution parameters for kernel type selection
  - *Used in*: layer_operator.py.SmoothingLagFilter._get_kernel() for kernel type dispatch
  - *Related to*: scm_kernel_sigma (when type="gaussian"), scm_kernel_size_min/max (affects all kernel types)
  - *Suggested*: "uniform" (default), "gaussian" for smoothing, "mixed" for variety, other options: "gaussian", "laplacian", "sobel", "recent_exponent", "mixed"

## SCM Functional Lag Parameters

- **`scm_functional_lag_min/max`**: Distribution parameters for functional lag ranges
  - *Used in*: layer_operator.py.SmoothingLagFilter for lag value sampling and application
  - *Related to*: scm_functional_lag_variance, scm_kernel_direction (affects lag direction)
  - *Suggested*: min=0.0, max=0.5 (default), min=0.1, max=0.3 for smaller lags
- **`scm_use_functional_lag`**: Distribution parameters for functional lag usage
  - *Used in*: layer_operator.py.SmoothingLagFilter initialization and lag application logic
  - *Related to*: scm_functional_lag_min/max, scm_functional_lag_variance
  - *Suggested*: [True, False] (default), [True] for always use lags
- **`scm_functional_lag_variance`**: Distribution parameters for functional lag variance
  - *Used in*: layer_operator.py.SmoothingLagFilter for random lag variation in kernel application
  - *Related to*: scm_functional_lag_min/max, scm_use_functional_lag
  - *Suggested*: 0.5 (default), 0.1-0.5 for different lag variation
- **`scm_layer_has_functional_lag_rate`**: Distribution parameters for layer functional lag rate
  - *Used in*: mlp_scm.py for determining which layers get functional lag operations
  - *Related to*: scm_use_layer_operator (must be True), scm_functional_lag_* parameters
  - *Suggested*: 0.5 (default), 0.3-0.7 for different layer lag rates
- **`scm_node_has_functional_lag_rate`**: Distribution parameters for node functional lag rate
  - *Used in*: mlp_scm.py for determining which nodes/features get functional lag operations
  - *Related to*: scm_use_layer_operator (must be True), scm_functional_lag_* parameters
  - *Suggested*: 0.5 (default), 0.3-0.7 for different node lag rates

## SCM Sampled Hyperparameters (Dynamic)

These parameters are used in config.get_scm_sampled_hp() and sampled during dataset generation:

### Architecture Parameters

- **`scm_multiclass_type`**: Distribution for multiclass type selection ("value", "rank")
  - *Used in*: mlp_scm.py and tree_scm.py for target variable encoding
  - *Related to*: max_classes (affects multiclass problem complexity)
  - *Suggested*: ["value", "rank"] (default), ["value"] for value-based, ["rank"] for rank-based
- **`scm_mlp_activations`**: Distribution for MLP activation function selection
  - *Used in*: mlp_scm.py for activation function assignment in MLP layers
  - *Related to*: scm_override_activations, scm_diverse_activation_names
  - *Suggested*: Default from TabICLConfig (mixed activations), override with specific activations
- **`scm_block_wise_dropout`**: Distribution for block-wise dropout usage
  - *Used in*: mlp_scm.py for weight initialization strategy (sparse vs dense)
  - *Related to*: scm_mlp_dropout_prob (alternative initialization method)
  - *Suggested*: [True, False] (default), [True] for sparse initialization, [False] for dense
- **`scm_mlp_dropout_prob`**: Distribution for MLP dropout probability
  - *Used in*: mlp_scm.py for dropout probability when block_wise_dropout=False
  - *Related to*: scm_block_wise_dropout (alternative to block-wise initialization)
  - *Suggested*: Beta distribution with min=0.1, max=5.0, scale=0.9 (default)

### Causal Structure Parameters

- **`scm_is_causal`**: Distribution for causal structure selection
  - *Used in*: mlp_scm.py for determining causal vs direct predictive mapping
  - *Related to*: scm_num_causes (only relevant when True), scm_y_is_effect, scm_in_clique
  - *Suggested*: [True] (default), [True, False] for mixed causal/non-causal
- **`scm_num_causes`**: Distribution for number of causal relationships
  - *Used in*: mlp_scm.py for number of initial root cause variables
  - *Related to*: scm_is_causal (only used when True), scm_hidden_dim (affects minimum size)
  - *Suggested*: Truncated normal log-scaled with min_mean=1, max_mean=12, lower_bound=1 (default)
- **`scm_y_is_effect`**: Distribution for whether Y is an effect variable
  - *Used in*: mlp_scm.py for target selection from MLP outputs (terminal vs intermediate)
  - *Related to*: scm_is_causal (only relevant when True)
  - *Suggested*: [True, False] (default), [True] for terminal effects, [False] for intermediate
- **`scm_in_clique`**: Distribution for clique membership
  - *Used in*: mlp_scm.py for contiguous vs scattered feature selection from MLP outputs
  - *Related to*: scm_is_causal (only relevant when True)
  - *Suggested*: [True, False] (default), [True] for contiguous features, [False] for scattered
- **`scm_sort_features`**: Distribution for feature sorting
  - *Used in*: mlp_scm.py for sorting features by original indices from MLP outputs
  - *Related to*: scm_is_causal (only relevant when True)
  - *Suggested*: [True, False] (default), [True] for sorted features, [False] for random order

### Network Architecture Parameters

- **`scm_num_layers`**: Distribution for number of SCM layers
  - *Used in*: mlp_scm.py for total MLP layers (must be >= 2)
  - *Related to*: scm_hidden_dim (affects minimum size calculation), dataset.py (affects single layer detection)
  - *Suggested*: Truncated normal log-scaled with min_mean=1, max_mean=6, lower_bound=2 (default)
- **`scm_hidden_dim`**: Distribution for hidden dimension size
  - *Used in*: mlp_scm.py for MLP layer dimensionality
  - *Related to*: scm_num_layers, scm_num_causes (minimum size constraints)
  - *Suggested*: Truncated normal log-scaled with min_mean=5, max_mean=130, lower_bound=4 (default)

### Initialization and Noise Parameters

- **`scm_init_std`**: Distribution for initialization standard deviation
  - *Used in*: mlp_scm.py for normal distribution weight initialization
  - *Related to*: scm_block_wise_dropout, scm_mlp_dropout_prob (affects scaling)
  - *Suggested*: Truncated normal log-scaled with min_mean=0.01, max_mean=10.0, lower_bound=0.0 (default)
- **`scm_noise_std`**: Distribution for noise standard deviation
  - *Used in*: mlp_scm.py for Gaussian noise added after each MLP layer
  - *Related to*: scm_pre_sample_noise_std (affects noise generation method)
  - *Suggested*: Truncated normal log-scaled with min_mean=0.0001, max_mean=0.3, lower_bound=0.0 (default)
- **`scm_noise_type`**: Distribution for noise type selection
  - *Used in*: mlp_scm.py for noise layer type ("gaussian", "ts", "mixed")
  - *Related to*: scm_noise_std, scm_mixed_noise_ratio
  - *Suggested*: ["gaussian"] (default), ["gaussian", "ts", "mixed"] for variety
- **`scm_mixed_noise_ratio`**: Distribution for mixed noise ratio
  - *Used in*: mlp_scm.py when noise_type="mixed" for ratio of different noise types
  - *Related to*: scm_noise_type (only used when "mixed")
  - *Suggested*: Uniform distribution with min=0.0, max=0.5 (default)

### Sampling Parameters

- **`scm_used_sampler`**: Distribution for sampler type selection
  - *Used in*: mlp_scm.py for XSampler type ("ts", "tabular", "real")
  - *Related to*: scm_sampling, scm_pre_sample_cause_stats
  - *Suggested*: ["ts"] (default), ["tabular"] for tabular data, ["real"] for real data
- **`scm_sampling`**: Distribution for sampling method selection
  - *Used in*: mlp_scm.py for XSampler sampling method ("normal", "mixed", etc.)
  - *Related to*: scm_used_sampler, scm_pre_sample_cause_stats
  - *Suggested*: ["mixed_both"] (default), ["normal"] for Gaussian, ["mixed_all"] for variety
- **`scm_ts_noise_sampling`**: Distribution for time series noise sampling
  - *Used in*: mlp_scm.py for time series noise generation when use_ts_noise=True
  - *Related to*: scm_noise_type (when "ts"), scm_sampling_mixed_names
  - *Suggested*: ["normal"] (default), ["mixed_simple", "mixed_all"] for variety
- **`scm_pre_sample_cause_stats`**: Distribution for pre-sampling cause statistics
  - *Used in*: mlp_scm.py for pre-sampling mean/std for cause variables
  - *Related to*: scm_sampling (when "normal"), scm_used_sampler
  - *Suggested*: [True, False] (default), [True] for pre-sampled stats, [False] for fixed stats
- **`scm_pre_sample_noise_std`**: Distribution for pre-sampling noise standard deviation
  - *Used in*: mlp_scm.py for per-dimension noise std sampling vs fixed std
  - *Related to*: scm_noise_std (affects noise generation method)
  - *Suggested*: [True, False] (default), [True] for per-dimension sampling, [False] for fixed std

### Counterfactual Parameters

- **`scm_counterfactual_type`**: Distribution for counterfactual type selection
  - *Used in*: dataset.py for counterfactual perturbation method
  - *Related to*: scm_counterfactual_num_changes, scm_counterfactual_num_samples
  - *Suggested*: ["random_normal", "random_uniform", "zero", "additive_norm"] (default)
- **`scm_counterfactual_num_changes`**: Distribution for number of counterfactual changes
  - *Used in*: dataset.py for number of features to modify in counterfactuals
  - *Related to*: scm_counterfactual_type, scm_counterfactual_num_samples
  - *Suggested*: Truncated normal log-scaled with min_mean=1, max_mean=10, lower_bound=1 (default)
- **`scm_counterfactual_num_samples`**: Distribution for number of counterfactual samples
  - *Used in*: dataset.py for number of counterfactual examples to generate
  - *Related to*: scm_counterfactual_type, scm_counterfactual_num_changes
  - *Suggested*: Truncated normal log-scaled with min_mean=0.1, max_mean=0.1, lower_bound=0 (default)

## System Configuration

- **`disable_print`**: Whether to disable print statements
  - *Suggested*: True (default), False for debugging
- **`prior_device`**: Device for prior model computation ("cpu", "cuda:X")
  - *Used in*: dataset generation and model loading
  - *Related to*: device (training device, should be different for parallel processing)
  - *Suggested*: "cpu" (default), "cuda:0" for GPU generation
- **`seed/np_seed/torch_seed`**: Random seeds for reproducibility (deprecated, use training config)
  - *Suggested*: 42 (default), any integer for reproducibility
- **`device`**: Training device (should be overridden by training config)
  - *Suggested*: "cuda:4" (default), "cuda:0" for single GPU, "cpu" for CPU training

## Key Methods

- **`get_scm_fixed_hp()`**: Returns fixed SCM hyperparameters dictionary
- **`get_scm_sampled_hp()`**: Returns sampled SCM hyperparameters dictionary with distribution configs
