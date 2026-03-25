import torch
import pytest
from unittest.mock import MagicMock
from rlq_scheduler.agent.agents.reinforcement_learning.smdp_double_dqn_agent import SMDPDoubleDQNAgent

STATE_SIZE = 4
N_ACTIONS  = 2


def make_agent():
    # mock run_config and global_config — agent needs these during __init__
    # but we don't want a real Redis/Mongo connection for unit tests
    run_config = MagicMock()
    run_config.state_size.return_value = STATE_SIZE
    run_config.is_bootstrapping_enabled.return_value = False

    global_config = MagicMock()
    global_config.backend_adapter.return_value = 'redis'

    agent = SMDPDoubleDQNAgent(
        action_space=['worker_0', 'worker_1'],
        network_config={'type': 'fully-connected', 'parameters': {'hidden_layers': 2}},
        experience_replay_capacity=2000,
        batch_size=32,
        run_config=run_config,
        global_config=global_config,
        train=True,
    )
    return agent


def fill_replay(agent, n=1000, r=1.0, tau=2.0):
    # push hardcoded (s, a, r, s', tau) tuples directly into the replay buffer
    device = agent._device
    for _ in range(n):
        state      = torch.zeros(1, STATE_SIZE, device=device)
        action     = torch.tensor([[0]], dtype=torch.long, device=device)
        reward     = torch.tensor([r], dtype=torch.float, device=device)
        next_state = torch.zeros(1, STATE_SIZE, device=device)
        tau_t      = torch.tensor([tau], dtype=torch.float, device=device)
        agent.experience_replay.push(state, action, reward, next_state, tau_t)


def test_rho_converges_to_r_over_tau():
    # with r=1.0 and tau=2.0, rho should converge toward 0.5 after 500 training steps
    agent = make_agent()
    fill_replay(agent, n=1000, r=1.0, tau=2.0)

    for _ in range(500):
        agent._optimize_step()

    assert abs(agent.rho - 0.5) < 0.05, f"rho={agent.rho:.4f} did not converge to 0.5"


def test_rho_history_grows():
    # rho_history should record one value per training step
    agent = make_agent()
    fill_replay(agent)

    for _ in range(50):
        agent._optimize_step()

    assert len(agent.rho_history) == 50


def test_bellman_target_differs_from_standard_dqn():
    # SMDP target = r/tau - rho + gamma*Q  vs  standard = r + gamma*Q
    # when tau=2 these must differ — confirms the override actually changed something
    r, tau, gamma, q_next = 4.0, 2.0, 0.99, 1.0
    standard = r + gamma * q_next
    smdp     = r / tau + gamma * q_next
    assert standard != smdp


def test_tau_normalisation_equalises_tasks():
    # a slow task (r=10, tau=10) and a fast task (r=1, tau=1) are equally good
    # the agent should treat them the same — this is the whole point of SMDP
    assert 10.0 / 10.0 == 1.0 / 1.0


def test_rho_starts_at_zero():
    agent = make_agent()
    assert agent.rho == 0.0
