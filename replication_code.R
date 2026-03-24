################################################################################
# Replication Code: "Hierarchy and War"
# van Beek, Lopate, Goodhart, Peterson, Edgerton, Xiong, Alam, Tiglay,
# Kent, & Braumoeller (2024), American Journal of Political Science
#
# This R implementation reconstructs the computational network model as
# described in the manuscript. The original was implemented in Julia/NetLogo.
# Parameter values not fixed in the paper are set to plausible defaults
# consistent with the model's described behavior.
#
# Model logic (per paper):
#   Each round has two stages:
#   (1) Hierarchy evaluation: states compare utility under each hierarchy vs.
#       anarchy and join the best option.
#   (2) Interstate bargaining: disputes arise probabilistically; states make
#       take-it-or-leave-it demands; rejection leads to war.
#
# Key equations (from paper, Section "Formal Model"):
#   Distance:  d(i,j) = sqrt(((r_i - r_j)*(1-v))^2 + ((s_i - s_j)*v)^2)
#   Demand:    x* maximizes U_i = p_war*(pi*p_victory - c_i) + (1-p_war)*x
#   Accept if: 1 - x <= p_victory_j * pi - (c_j + eps)
#   Tribute:   t_ih = (T_h - z_i) * sqrt((s_i - s_h)^2)
#   Join h* if: sum(U|h*) - t_ih* >= sum(U|h) - t_ih  for all h != h*
#              AND sum(U|h*) - t_ih* >= sum(U|anarchy)
################################################################################


# ==============================================================================
# 0. DEPENDENCIES
# ==============================================================================
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("dplyr",   quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("tidyr",   quietly = TRUE)) install.packages("tidyr")
if (!requireNamespace("patchwork", quietly = TRUE)) install.packages("patchwork")

library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)

set.seed(42)


# ==============================================================================
# 1. PARAMETERS
# ==============================================================================
# These are set to plausible defaults consistent with the paper's description.
# The paper lists exact values in its Online Appendix (not publicly available
# at time of writing). Adjust as needed.

params <- list(
  # System size
  n_states    = 10,    # number of states
  n_hierarchs = 2,     # number of potential hierarchs (randomly chosen)
  
  # Interest weights
  # v: weight on governance vs. intrinsic interests (0 = all intrinsic, 1 = all governance)
  v_baseline  = 0.3,   # baseline: mix of intrinsic and governance conflict
  v_high      = 0.8,   # high governance conflict (screening shock)
  
  # Bargaining / war parameters
  pi          = 1.0,   # value of the disputed issue (constant)
  p_victory   = 0.5,   # equal probability of victory (symmetric)
  c_baseline  = 0.3,   # baseline cost of war
  c_high      = 0.6,   # high systemic war cost (post-major-war shock)
  sigma_base  = 0.15,  # baseline std dev of uncertainty in war costs
  sigma_high  = 0.30,  # high uncertainty (post-major-war shock)
  
  # Hierarchy parameters
  T_h_base    = 0.5,   # baseline tribute (max, for most dissimilar state)
  # T_h scales with n_states per paper; we implement this as T_h * (n/200)
  u_baseline  = 0.2,   # hierarchy uncertainty-reduction factor (baseline)
  u_high      = 0.6,   # high uncertainty reduction (uncertainty shock)
  ch_baseline = 0.1,   # hierarchy war-cost increase for nonmembers (baseline)
  ch_high     = 0.4,   # high war-cost increase (war-cost shock)
  
  # Simulation settings
  n_turns_burnin    = 50,   # turns to reach baseline equilibrium
  n_turns_post      = 50,   # turns after manipulation
  n_iterations      = 200,  # Monte Carlo iterations
  max_disputes_per_dyad = 1 # max disputes per dyad per turn (simplification)
)


# ==============================================================================
# 2. STATE INITIALIZATION
# ==============================================================================
# Each state has:
#   r_i: intrinsic interest position ~ Uniform(0,1)
#   s_i: governance interest position ~ Uniform(0,1)
#   z_i: strategic value ~ Uniform(0, 0.2) [non-negative, bounded]
#   power_i: ~ Uniform(0,1); top n_hierarchs become hierarchs

initialize_states <- function(n_states, n_hierarchs) {
  states <- data.frame(
    id       = 1:n_states,
    r        = runif(n_states),          # intrinsic interest
    s        = runif(n_states),          # governance interest
    z        = runif(n_states, 0, 0.2),  # strategic value
    power    = runif(n_states),
    hierarch = FALSE,
    hier_id  = NA_integer_             # which hierarchy this state belongs to (NA = anarchy)
  )
  
  # Designate the top n_hierarchs states as hierarchs
  top_idx <- order(states$power, decreasing = TRUE)[1:n_hierarchs]
  states$hierarch[top_idx] <- TRUE
  states$hier_id[top_idx]  <- top_idx  # each hierarch leads its own hierarchy
  
  states
}


# ==============================================================================
# 3. DISTANCE FUNCTION
# ==============================================================================
# d(i,j) = sqrt( ((r_i - r_j)*(1-v))^2 + ((s_i - s_j)*v)^2 )
# This is the Euclidean distance in interest space, weighted by v.

interest_distance <- function(r_i, s_i, r_j, s_j, v) {
  sqrt(((r_i - r_j) * (1 - v))^2 + ((s_i - s_j) * v)^2)
}


# ==============================================================================
# 4. HIERARCHY MEMBERSHIP FUNCTIONS
# ==============================================================================

# Tribute that hierarch h demands of state i:
#   t_ih = (T_h - z_i) * sqrt((s_i - s_h)^2)
# T_h scales with n_states so hierarchies are more valuable in larger systems.
# We clamp tribute to >= 0.

compute_tribute <- function(s_i, z_i, s_h, T_h_base, n_states) {
  T_h <- T_h_base * (n_states / 200)  # scale with system size
  t   <- (T_h - z_i) * sqrt((s_i - s_h)^2)
  max(t, 0)
}

# Hierarchy reduces uncertainty: eps ~ N(0, sigma*(1-u))
# Hierarchy increases opponent's war cost: c_j -> c_j * (1 + ch)
# These are applied during the bargaining stage.


# ==============================================================================
# 5. BARGAINING MODEL
# ==============================================================================
# Take-it-or-leave-it demand by i:
#   i maximizes U_i = p_war*(pi*p_vic - c_i) + (1-p_war)*x
#   subject to j accepting: 1 - x <= p_vic_j * pi - (c_j + eps)
#
# j accepts iff 1 - x <= pi*p_vic_j - (c_j + eps)
# => j accepts iff x >= 1 - pi*p_vic_j + c_j + eps
# => j's acceptance threshold: x_thresh = 1 - pi*p_vic_j + c_j + eps
#
# i wants the largest x j will accept, so optimal demand: x* = 1 - pi*p_vic_j + c_j + eps
# But i doesn't know eps, so i must account for the distribution.
#
# War occurs if j rejects: 1 - x* <= pi*p_vic_j - (c_j + eps)
#   => eps < -(c_j)  (given our formulation)
#
# Probability of war: P(eps < -c_j) where eps ~ N(0, sigma_eff)
# = pnorm(-c_j, mean=0, sd=sigma_eff)
#
# i's optimal demand maximizes:
#   U_i = p_war*(pi*p_vic_i - c_i) + (1-p_war)*x
# Since x is declining in p_war (higher demand -> higher rejection probability),
# i balances these. The canonical result (Fearon 1995): i demands x* such that
#   j is just indifferent => p_war = pnorm(-c_j, sd=sigma_eff)
#
# Expected utility for i from a single dyadic interaction:
#   U_i = p_war*(pi*p_vic_i - c_i) + (1-p_war)*x*
# where x* = pi*p_vic_i - c_i  (demand that equates expected war vs peace payoff)
# We use the standard result: x* = pi*(1 - p_vic_j) + c_j + E[eps | accept]
# For simplicity and tractability, we compute:
#   p_war = pnorm(-c_j_eff, mean=0, sd=sigma_eff)   [prob j rejects]
#   U_i   = p_war*(pi*p_vic - c_i) + (1-p_war)*(pi*(1-p_vic) + c_j_eff)

bargaining_outcome <- function(c_i, c_j_eff, sigma_eff,
                               pi = 1.0, p_vic = 0.5) {
  # p_war: probability j rejects i's demand (war occurs)
  p_war <- pnorm(-c_j_eff, mean = 0, sd = sigma_eff)
  p_war <- max(min(p_war, 1), 0)  # clamp
  
  # Expected utility for i
  U_i <- p_war * (pi * p_vic - c_i) + (1 - p_war) * (pi * (1 - p_vic) + c_j_eff)
  
  list(p_war = p_war, U_i = U_i, war = rbinom(1, 1, p_war) == 1)
}


# ==============================================================================
# 6. HIERARCHY EVALUATION STAGE (Algorithm 2)
# ==============================================================================
# Each state evaluates:
#   - Expected utility under anarchy
#   - Expected utility as a member of each available hierarchy (net of tribute)
# Joins the hierarchy with highest net utility, if it exceeds anarchy utility.

evaluate_hierarchies <- function(states, hierarch_ids,
                                 v, c_base, sigma_base,
                                 u, ch, T_h_base, pi, p_vic, n_states) {
  
  n  <- nrow(states)
  new_hier_id <- rep(NA_integer_, n)
  
  for (i in seq_len(n)) {
    if (states$hierarch[i]) {
      # Hierarchs stay in their own hierarchy
      new_hier_id[i] <- states$id[i]
      next
    }
    
    # --- Compute aggregate utility under anarchy ---
    U_anarchy <- compute_aggregate_utility(
      focal_id   = i,
      states     = states,
      membership = rep(NA, n),  # all in anarchy for this counterfactual
      v          = v,
      c_base     = c_base,
      sigma_base = sigma_base,
      u          = 0,    # no hierarchy benefits
      ch         = 0,
      pi         = pi,
      p_vic      = p_vic,
      n_states   = n_states
    )
    
    best_util <- U_anarchy
    best_hier <- NA_integer_
    
    # --- Evaluate each available hierarchy ---
    for (h_id in hierarch_ids) {
      h_idx <- which(states$id == h_id)
      s_h   <- states$s[h_idx]
      z_i   <- states$z[i]
      s_i   <- states$s[i]
      
      tribute <- compute_tribute(s_i, z_i, s_h, T_h_base, n_states)
      
      # Hypothetical membership vector: i joins hierarchy h
      hyp_membership        <- states$hier_id
      hyp_membership[i]     <- h_id
      
      U_h <- compute_aggregate_utility(
        focal_id   = i,
        states     = states,
        membership = hyp_membership,
        v          = v,
        c_base     = c_base,
        sigma_base = sigma_base,
        u          = u,
        ch         = ch,
        pi         = pi,
        p_vic      = p_vic,
        n_states   = n_states
      ) - tribute
      
      if (U_h > best_util) {
        best_util <- U_h
        best_hier <- h_id
      }
    }
    
    new_hier_id[i] <- best_hier
  }
  
  states$hier_id <- new_hier_id
  states
}


# ==============================================================================
# 7. AGGREGATE UTILITY COMPUTATION
# ==============================================================================
# For state i, compute sum of expected utilities across all potential
# dyadic interactions, given the membership configuration.

compute_aggregate_utility <- function(focal_id, states, membership,
                                      v, c_base, sigma_base,
                                      u, ch, pi, p_vic, n_states) {
  i       <- focal_id
  n       <- nrow(states)
  total_U <- 0
  
  r_i <- states$r[i]
  s_i <- states$s[i]
  m_i <- membership[i]  # hierarchy of focal state
  
  for (j in seq_len(n)) {
    if (j == i) next
    
    r_j <- states$r[j]
    s_j <- states$s[j]
    m_j <- membership[j]
    
    # Probability a dispute arises (proportional to interest distance)
    dist_ij <- interest_distance(r_i, s_i, r_j, s_j, v)
    # Scale to (0,1): max possible distance on unit square is sqrt(2) ≈ 1.414
    p_dispute <- dist_ij / sqrt(2)
    
    if (runif(1) > p_dispute) next  # no dispute this dyad
    
    # Effective war costs and uncertainty given hierarchy membership
    same_hier <- (!is.na(m_i) && !is.na(m_j) && m_i == m_j)
    
    # j's effective cost (increased if i is in a hierarchy and j is not)
    i_in_hier <- !is.na(m_i)
    j_in_hier <- !is.na(m_j)
    
    c_j_eff   <- c_base
    sigma_eff <- sigma_base
    
    if (i_in_hier && !j_in_hier) {
      # i's hierarchy increases j's war cost
      c_j_eff <- c_base * (1 + ch)
    }
    if (i_in_hier) {
      # i's hierarchy reduces i's uncertainty about j
      sigma_eff <- sigma_base * (1 - u)
    }
    # Enforce positivity
    c_j_eff   <- max(c_j_eff, 0.001)
    sigma_eff <- max(sigma_eff, 0.001)
    
    result  <- bargaining_outcome(c_base, c_j_eff, sigma_eff, pi, p_vic)
    total_U <- total_U + result$U_i
  }
  
  total_U
}


# ==============================================================================
# 8. INTERSTATE BARGAINING STAGE (Algorithm 3)
# ==============================================================================
# Issues arise stochastically between each dyad; states bargain;
# war or peace results.

bargaining_stage <- function(states, v, c_base, sigma_base,
                             u, ch, pi, p_vic) {
  n       <- nrow(states)
  n_wars  <- 0
  wars_df <- data.frame(sender = integer(0),
                        receiver = integer(0),
                        cross_hier = logical(0),
                        within_hier = logical(0),
                        outside_hier = logical(0))
  
  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      if (j <= i) next  # undirected, avoid double-counting
      
      r_i <- states$r[i]; s_i <- states$s[i]; m_i <- states$hier_id[i]
      r_j <- states$r[j]; s_j <- states$s[j]; m_j <- states$hier_id[j]
      
      dist_ij   <- interest_distance(r_i, s_i, r_j, s_j, v)
      p_dispute <- dist_ij / sqrt(2)
      
      if (runif(1) > p_dispute) next
      
      # Classify dyad relationship
      same_hier    <- (!is.na(m_i) && !is.na(m_j) && m_i == m_j)
      cross_hier   <- (!is.na(m_i) && !is.na(m_j) && m_i != m_j)
      within_hier  <- same_hier
      outside_hier <- (is.na(m_i) && is.na(m_j))
      
      # i makes demand of j
      i_in_hier <- !is.na(m_i)
      c_j_eff   <- if (i_in_hier && !same_hier) c_base * (1 + ch) else c_base
      sigma_eff <- if (i_in_hier) sigma_base * (1 - u) else sigma_base
      c_j_eff   <- max(c_j_eff, 0.001)
      sigma_eff <- max(sigma_eff, 0.001)
      
      result <- bargaining_outcome(c_base, c_j_eff, sigma_eff, pi, p_vic)
      
      if (result$war) {
        n_wars <- n_wars + 1
        wars_df <- rbind(wars_df, data.frame(
          sender      = i,
          receiver    = j,
          cross_hier  = cross_hier,
          within_hier = within_hier,
          outside_hier = outside_hier
        ))
      }
    }
  }
  
  list(n_wars = n_wars, wars = wars_df, states = states)
}


# ==============================================================================
# 9. MAIN SIMULATION LOOP (Algorithm 1)
# ==============================================================================
run_simulation <- function(params,
                           v          = params$v_baseline,
                           c_base     = params$c_baseline,
                           sigma_base = params$sigma_base,
                           u          = params$u_baseline,
                           ch         = params$ch_baseline,
                           n_turns    = params$n_turns_burnin,
                           states_in  = NULL,
                           progress   = TRUE) { # [MW: added progress option]
  
  if (is.null(states_in)) {
    states <- initialize_states(params$n_states, params$n_hierarchs)
  } else {
    states <- states_in
  }
  
  hierarch_ids <- states$id[states$hierarch]
  
  results <- vector("list", n_turns)
  
  for (turn in seq_len(n_turns)) {
    
    # Stage 1: Hierarchy evaluation
    states <- evaluate_hierarchies(
      states      = states,
      hierarch_ids = hierarch_ids,
      v           = v,
      c_base      = c_base,
      sigma_base  = sigma_base,
      u           = u,
      ch          = ch,
      T_h_base    = params$T_h_base,
      pi          = params$pi,
      p_vic       = params$p_victory,
      n_states    = params$n_states
    )
    
    # Stage 2: Interstate bargaining
    barg <- bargaining_stage(
      states     = states,
      v          = v,
      c_base     = c_base,
      sigma_base = sigma_base,
      u          = u,
      ch         = ch,
      pi         = params$pi,
      p_vic      = params$p_victory
    )
    
    states <- barg$states
    
    # Record outcomes
    n_members <- sum(!is.na(states$hier_id))
    results[[turn]] <- list(
      turn          = turn,
      n_wars        = barg$n_wars,
      n_within      = sum(barg$wars$within_hier),
      n_cross       = sum(barg$wars$cross_hier),
      n_outside     = sum(barg$wars$outside_hier),
      n_members     = n_members,
      pct_members   = n_members / params$n_states,
      wars          = barg$wars, # [MW: added]
      states        = states
    )
    
    # [MW: give me a progress update]
    if(progress) {
      msg <- sprintf("\rProcessing: %d%% (%d/%d)", round(turn/n_turns*100), turn, n_turns)
      
      # Print, flush to console, and carriage return to start of line
      cat(msg)
      flush.console()
    }
  }
  
  list(results = results, final_states = states)
}


# ==============================================================================
# 10. MONTE CARLO WRAPPER & SIMULATED ATE CALCULATION
# ==============================================================================
# Runs `n_iter` full simulations, each with:
#   - burnin_turns turns at baseline parameters
#   - post_turns turns at post-manipulation parameters
# Returns the average difference in outcome (wars per turn) between
# pre- and post-manipulation equilibria (= "simulated ATE")

run_ate_simulation <- function(params,
                               # Pre-manipulation parameters
                               v_pre = params$v_baseline,
                               c_pre = params$c_baseline,
                               sig_pre = params$sigma_base,
                               u_pre = params$u_baseline,
                               ch_pre = params$ch_baseline,
                               # Post-manipulation parameters
                               v_post = params$v_baseline,
                               c_post = params$c_baseline,
                               sig_post = params$sigma_base,
                               u_post = params$u_baseline,
                               ch_post = params$ch_baseline,
                               n_iter = params$n_iterations,
                               progress = TRUE) { # [MW: report progress]
  
  ate_wars        <- numeric(n_iter)
  ate_within      <- numeric(n_iter)
  ate_cross       <- numeric(n_iter)
  ate_members     <- numeric(n_iter)
  
  for (iter in seq_len(n_iter)) {
      
    # [MW: give me a progress update]
    if(progress) {
      msg <- sprintf("\rProcessing: %d%% (%d/%d)", round(iter/n_iter*100), iter, n_iter)
      
      # Print, flush to console, and carriage return to start of line
      cat(msg)
      flush.console()
    }
    
    # Re-initialize states each iteration (per paper: "nonfixed variables
    # resampled from their distributions at each iteration")
    states_init <- initialize_states(params$n_states, params$n_hierarchs)
    
    # Burn-in: run to equilibrium under pre-manipulation parameters
    pre <- run_simulation(params,
                          v = v_pre, c_base = c_pre, sigma_base = sig_pre,
                          u = u_pre, ch = ch_pre,
                          n_turns = params$n_turns_burnin,
                          states_in = states_init,
                          progress = FALSE)
    
    # Extract last-turn equilibrium values (average over last 5 turns)
    last5_pre <- tail(pre$results, 5)
    w_pre     <- mean(sapply(last5_pre, `[[`, "n_wars"))
    wi_pre    <- mean(sapply(last5_pre, `[[`, "n_within"))
    wc_pre    <- mean(sapply(last5_pre, `[[`, "n_cross"))
    m_pre     <- mean(sapply(last5_pre, `[[`, "pct_members"))
    
    # Post-manipulation: continue from pre-equilibrium states
    post <- run_simulation(params,
                           v = v_post, c_base = c_post, sigma_base = sig_post,
                           u = u_post, ch = ch_post,
                           n_turns = params$n_turns_post,
                           states_in = pre$final_states,
                           progress = FALSE)
    
    last5_post <- tail(post$results, 5)
    w_post     <- mean(sapply(last5_post, `[[`, "n_wars"))
    wi_post    <- mean(sapply(last5_post, `[[`, "n_within"))
    wc_post    <- mean(sapply(last5_post, `[[`, "n_cross"))
    m_post     <- mean(sapply(last5_post, `[[`, "pct_members"))
    
    ate_wars[iter]    <- w_post    - w_pre
    ate_within[iter]  <- wi_post   - wi_pre
    ate_cross[iter]   <- wc_post   - wc_pre
    ate_members[iter] <- m_post    - m_pre
  }
  
  list(
    ate_wars    = mean(ate_wars),
    ate_within  = mean(ate_within),
    ate_cross   = mean(ate_cross),
    ate_members = mean(ate_members),
    se_wars     = sd(ate_wars)   / sqrt(n_iter),
    se_within   = sd(ate_within) / sqrt(n_iter),
    se_cross    = sd(ate_cross)  / sqrt(n_iter),
    se_members  = sd(ate_members)/ sqrt(n_iter),
    raw_wars    = ate_wars,
    raw_within  = ate_within,
    raw_cross   = ate_cross,
    raw_members = ate_members
  )
}


# ==============================================================================
# 11. REPLICATE FIGURE 2: Cross-hierarchy wars (mechanisms)
#     Three manipulations:
#       (a) Uncertainty shock: u increases (hierarchies reduce uncertainty more)
#       (b) War-cost shock:    ch increases (hierarchies raise opponents' costs)
#       (c) Screening shock:   v increases (governance interests dominate)
# ==============================================================================

cat("=== Simulating Figure 2: Cross-hierarchy wars ===\n")

cat("  [2a] Uncertainty shock...\n")
ate_uncertainty <- run_ate_simulation(
  params,
  u_pre  = params$u_baseline,
  u_post = params$u_high,
  n_iter = params$n_iterations
)

cat("  [2b] War-cost shock...\n")
ate_warcost <- run_ate_simulation(
  params,
  ch_pre  = params$ch_baseline,
  ch_post = params$ch_high,
  n_iter  = params$n_iterations
)

cat("  [2c] Screening shock (governance conflict)...\n")
ate_screening <- run_ate_simulation(
  params,
  v_pre  = params$v_baseline,
  v_post = params$v_high,
  n_iter = params$n_iterations
)

# Compile results for Figure 2
fig2_data <- data.frame(
  mechanism = factor(c("Uncertainty", "War Cost", "Screening"),
                     levels = c("Screening", "War Cost", "Uncertainty")),
  ate       = c(ate_uncertainty$ate_cross,
                ate_warcost$ate_cross,
                ate_screening$ate_cross),
  se        = c(ate_uncertainty$se_cross,
                ate_warcost$se_cross,
                ate_screening$se_cross)
)
fig2_data$lo <- fig2_data$ate - 1.96 * fig2_data$se
fig2_data$hi <- fig2_data$ate + 1.96 * fig2_data$se


# ==============================================================================
# 12. REPLICATE FIGURE 4: Within-hierarchy wars (same three mechanisms)
# ==============================================================================

cat("\n=== Simulating Figure 4: Within-hierarchy wars ===\n")

fig4_data <- data.frame(
  mechanism = factor(c("Uncertainty", "War Cost", "Screening"),
                     levels = c("Screening", "War Cost", "Uncertainty")),
  ate       = c(ate_uncertainty$ate_within,
                ate_warcost$ate_within,
                ate_screening$ate_within),
  se        = c(ate_uncertainty$se_within,
                ate_warcost$se_within,
                ate_screening$se_within)
)
fig4_data$lo <- fig4_data$ate - 1.96 * fig4_data$se
fig4_data$hi <- fig4_data$ate + 1.96 * fig4_data$se


# ==============================================================================
# 13. REPLICATE FIGURE 5: Hierarchy formation after large wars
#     Two manipulations:
#       (a) Systemic war-cost shock
#       (b) Systemic uncertainty shock
# ==============================================================================

cat("\n=== Simulating Figure 5: Hierarchy formation after large wars ===\n")

cat("  [5a] Systemic war-cost shock...\n")
ate_syscost <- run_ate_simulation(
  params,
  c_pre  = params$c_baseline,
  c_post = params$c_high,
  n_iter = params$n_iterations
)

cat("  [5b] Systemic uncertainty shock...\n")
ate_sysuncert <- run_ate_simulation(
  params,
  sig_pre  = params$sigma_base,
  sig_post = params$sigma_high,
  n_iter   = params$n_iterations
)

fig5_data <- data.frame(
  mechanism = factor(c("Systemic War Costs", "Systemic Uncertainty"),
                     levels = c("Systemic Uncertainty", "Systemic War Costs")),
  ate       = c(ate_syscost$ate_members,
                ate_sysuncert$ate_members),
  se        = c(ate_syscost$se_members,
                ate_sysuncert$se_members)
)
fig5_data$lo <- fig5_data$ate - 1.96 * fig5_data$se
fig5_data$hi <- fig5_data$ate + 1.96 * fig5_data$se


# ==============================================================================
# 14. REPLICATE FIGURE 3: Screening - rates of conflict across/outside/within
#     Run single long simulation under low-governance and high-governance
#     conflict, record rates across / outside / within hierarchies.
# ==============================================================================

cat("\n=== Simulating Figure 3: Screening effect on conflict rates ===\n")

collect_conflict_rates <- function(params, v, c_base, sigma_base, u, ch,
                                   n_turns_total = 100, n_iter = 50) {
  rates <- data.frame(type = character(0), rate = numeric(0))
  
  for (iter in seq_len(n_iter)) {
    states_init <- initialize_states(params$n_states, params$n_hierarchs)
    sim <- run_simulation(params,
                          v = v, c_base = c_base, sigma_base = sigma_base,
                          u = u, ch = ch,
                          n_turns = n_turns_total,
                          states_in = states_init)
    
    # Average over last 20 turns (equilibrium)
    last20 <- tail(sim$results, 20)
    total_w  <- mean(sapply(last20, `[[`, "n_wars"))
    within_w <- mean(sapply(last20, `[[`, "n_within"))
    cross_w  <- mean(sapply(last20, `[[`, "n_cross"))
    outside_w<- mean(sapply(last20, `[[`, "n_outside"))
    denom    <- max(total_w, 1)
    
    rates <- rbind(rates, data.frame(
      iter    = iter,
      Across  = cross_w   / denom,
      Outside = outside_w / denom,
      Within  = within_w  / denom
    ))
  }
  rates
}

rates_no_screening <- collect_conflict_rates(params,
                                             v = params$v_baseline,
                                             c_base = params$c_baseline,
                                             sigma_base = params$sigma_base,
                                             u = params$u_baseline,
                                             ch = params$ch_baseline,
                                             n_iter = 50)

rates_screening    <- collect_conflict_rates(params,
                                             v = params$v_high,
                                             c_base = params$c_baseline,
                                             sigma_base = params$sigma_base,
                                             u = params$u_baseline,
                                             ch = params$ch_baseline,
                                             n_iter = 50)

fig3_no  <- rates_no_screening  %>%
  pivot_longer(-iter, names_to = "type", values_to = "rate") %>%
  group_by(type) %>%
  summarise(mean = mean(rate), se = sd(rate)/sqrt(n()), .groups = "drop")

fig3_yes <- rates_screening %>%
  pivot_longer(-iter, names_to = "type", values_to = "rate") %>%
  group_by(type) %>%
  summarise(mean = mean(rate), se = sd(rate)/sqrt(n()), .groups = "drop")


# ==============================================================================
# 15. POISSON CHECK: Distribution of war onsets
# ==============================================================================

cat("\n=== Checking Poisson distribution of war onsets ===\n")

run_long_sim <- function(params, n_turns = 200) {
  states <- initialize_states(params$n_states, params$n_hierarchs)
  sim    <- run_simulation(params,
                           n_turns   = n_turns,
                           states_in = states)
  sapply(sim$results, `[[`, "n_wars")
}

war_counts <- run_long_sim(params, n_turns = 200)
lambda_est <- mean(war_counts)

poisson_test <- ks.test(war_counts, "ppois", lambda = lambda_est)
cat(sprintf("  Estimated lambda: %.2f\n", lambda_est))
cat(sprintf("  KS test p-value vs Poisson(%.2f): %.4f\n",
            lambda_est, poisson_test$p.value))


# ==============================================================================
# 16. PLOTS
# ==============================================================================

theme_paper <- theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold", size = 10),
        axis.title = element_text(size = 9))

# --- Figure 2 (cross-hierarchy wars) ---
p_fig2 <- ggplot(fig2_data, aes(x = ate, y = mechanism)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.15, linewidth = 0.7) +
  geom_point(size = 3, color = "#D55E00") +
  labs(title = "Figure 2: Simulated ATE on Cross-Hierarchy Wars",
       x = "Simulated ATE (wars per turn)",
       y = NULL) +
  theme_paper

# --- Figure 4 (within-hierarchy wars) ---
p_fig4 <- ggplot(fig4_data, aes(x = ate, y = mechanism)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.15, linewidth = 0.7) +
  geom_point(size = 3, color = "#0072B2") +
  labs(title = "Figure 4: Simulated ATE on Within-Hierarchy Wars",
       x = "Simulated ATE (wars per turn)",
       y = NULL) +
  theme_paper

# --- Figure 5 (hierarchy membership) ---
p_fig5 <- ggplot(fig5_data, aes(x = ate * 100, y = mechanism)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  geom_errorbarh(aes(xmin = lo * 100, xmax = hi * 100),
                 height = 0.15, linewidth = 0.7) +
  geom_point(size = 3, color = "#009E73") +
  labs(title = "Figure 5: Simulated ATE on Hierarchy Membership",
       x = "Simulated ATE (% states in hierarchy)",
       y = NULL) +
  theme_paper

# --- Figure 3 (screening: conflict rates by dyad type) ---
fig3_no$condition  <- "No Governance Conflict (Low v)"
fig3_yes$condition <- "High Governance Conflict (High v)"
fig3_combined      <- bind_rows(fig3_no, fig3_yes)
fig3_combined$type <- factor(fig3_combined$type,
                             levels = c("Across", "Outside", "Within"))

p_fig3 <- ggplot(fig3_combined,
                 aes(x = type, y = mean, ymin = mean - 1.96 * se,
                     ymax = mean + 1.96 * se, color = condition)) +
  geom_pointrange(position = position_dodge(0.4), size = 0.6) +
  scale_color_manual(values = c("#CC79A7", "#56B4E9")) +
  labs(title = "Figure 3: Screening Effect on Conflict Rates",
       x = "Dyad Type",
       y = "Share of All Wars",
       color = NULL) +
  theme_paper +
  theme(legend.position = "bottom")

# --- Poisson check plot ---
war_df     <- data.frame(n_wars = war_counts)
pois_df    <- data.frame(
  x   = 0:max(war_counts),
  prob = dpois(0:max(war_counts), lambda = lambda_est) * length(war_counts)
)

p_poisson <- ggplot(war_df, aes(x = n_wars)) +
  geom_histogram(aes(y = after_stat(count)), binwidth = 1,
                 fill = "steelblue", color = "white", alpha = 0.7) +
  geom_line(data = pois_df, aes(x = x, y = prob),
            color = "red", linewidth = 1) +
  labs(title = sprintf("Poisson Check: War Onsets per Turn (KS p = %.3f)",
                       poisson_test$p.value),
       x = "Wars per Turn",
       y = "Count") +
  theme_paper

# --- Combine and print ---
combined_plot <- (p_fig2 | p_fig4) / (p_fig5 | p_fig3) / p_poisson +
  plot_annotation(
    title = "van Beek et al. (2024) — Hierarchy and War: Simulated Results",
    subtitle = paste0("n_states = ", params$n_states,
                      ", n_iterations = ", params$n_iterations,
                      ", burnin = ", params$n_turns_burnin,
                      " turns"),
    theme = theme(plot.title = element_text(size = 13, face = "bold"))
  )

print(combined_plot)


# ==============================================================================
# 17. SUMMARY TABLE
# ==============================================================================

cat("\n=== Summary of Simulated ATEs ===\n\n")

summary_tbl <- data.frame(
  Manipulation = c(
    "Uncertainty shock → cross-hier wars",
    "War-cost shock → cross-hier wars",
    "Screening shock → cross-hier wars",
    "Uncertainty shock → within-hier wars",
    "War-cost shock → within-hier wars",
    "Screening shock → within-hier wars",
    "Systemic war-cost shock → hierarchy membership",
    "Systemic uncertainty shock → hierarchy membership"
  ),
  Expected_Direction = c("-", "-", "+", "-", "-", "-", "+", "+"),
  Simulated_ATE = round(c(
    ate_uncertainty$ate_cross,
    ate_warcost$ate_cross,
    ate_screening$ate_cross,
    ate_uncertainty$ate_within,
    ate_warcost$ate_within,
    ate_screening$ate_within,
    ate_syscost$ate_members  * 100,
    ate_sysuncert$ate_members* 100
  ), 3),
  SE = round(c(
    ate_uncertainty$se_cross,
    ate_warcost$se_cross,
    ate_screening$se_cross,
    ate_uncertainty$se_within,
    ate_warcost$se_within,
    ate_screening$se_within,
    ate_syscost$se_members  * 100,
    ate_sysuncert$se_members* 100
  ), 3)
)
print(summary_tbl, row.names = FALSE)

cat("\nNote: Membership ATEs are in percentage-point units.\n")
cat("      All other ATEs are in wars-per-turn units.\n")
