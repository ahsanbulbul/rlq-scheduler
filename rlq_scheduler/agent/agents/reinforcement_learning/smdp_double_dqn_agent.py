import torch
import torch.nn.functional as F
from collections import namedtuple

from rlq_scheduler.agent.agents.reinforcement_learning.dqn_agent import DoubleDQNAgent
from rlq_scheduler.common.experience_replay.experience_replay import ExperienceReplay
from rlq_scheduler.common.state.state import state_to_tensor

# place holder
sample_tau = lambda psi, omega: 1.0
mean_tau   = lambda psi, omega: 1.0

# extend standard experience entry to also store holding time (tau)
# base ExperienceEntry only has (state, action, reward, next_state)
SMDPExperienceEntry = namedtuple('SMDPExperienceEntry', 'state action reward next_state tau')


class SMDPDoubleDQNAgent(DoubleDQNAgent):
    """
    Extends DoubleDQNAgent to support Semi-Markov Decision Processes.
    The key difference is in the Bellman update: instead of y = r + gamma * Q(s'),
    we use y = r/tau - rho + gamma * Q(s'), which normalises reward per unit time.
    """

    def __init__(self, *args, rho_lr=0.01, **kwargs):
        kwargs.setdefault('name', 'SMDPDoubleDQN')

        # swap out the replay buffer for one that also stores tau per transition
        capacity = kwargs.pop('experience_replay_capacity', 2000)
        smdp_replay = ExperienceReplay(
            capacity=capacity,
            entry_structure=SMDPExperienceEntry
        )
        super().__init__(*args, experience_replay=smdp_replay, **kwargs)

        # rho is long running avg cost
        self.rho = 0.0
        self.rho_lr = rho_lr
        self.rho_history = []  

    def push_experience_entry(self, state, action, reward, next_state, tau=1.0):
        # extend base class to store holding time, tau
        tensor_state = state_to_tensor(state, device=self._device)
        tensor_next_state = state_to_tensor(next_state, device=self._device)
        self.experience_replay.push(
            tensor_state.view(-1, state.size),
            torch.tensor([[action]], device=self._device, dtype=torch.long),
            torch.tensor([reward], device=self._device, dtype=torch.float),
            tensor_next_state.view(-1, next_state.size),
            torch.tensor([tau], device=self._device, dtype=torch.float),
        )

    def _optimize_step(self):
        # pull a random batch from replay
        samples = self.experience_replay.sample(self._batch_size)
        batch = SMDPExperienceEntry(*zip(*samples))
        batch_states = torch.cat(batch.state)
        batch_actions = torch.cat(batch.action)
        batch_rewards = torch.cat(batch.reward)
        batch_next_states = torch.cat(batch.next_state)
        batch_taus = torch.cat(batch.tau)

        expected_values = self._compute_expected_state_action_values(
            batch_next_states, batch_rewards, batch_taus
        )

        # Q(s, a) for the actions that were actually taken
        state_action_values = self.action_value_network(batch_states).gather(1, batch_actions)

        loss = F.smooth_l1_loss(input=state_action_values, target=expected_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _compute_expected_state_action_values(self, batch_next_states, batch_rewards, batch_taus=None):
        # Double DQN: use action_value_network to pick the best action, target_network to score it
        # this reduces overestimation bias compared to vanilla DQN
        next_state_action_values = self.action_value_network(batch_next_states)
        action_max_indexes = next_state_action_values.max(dim=1)[1]

        next_state_target_values = self.target_network(batch_next_states)
        next_state_target_values = next_state_target_values.gather(
            1, action_max_indexes.unsqueeze(1)
        ).squeeze(1)

        # SMDP Bellman target: y = r/tau - rho + gamma * Q(s', a*)
        # dividing by tau converts reward into reward-per-second so the agent
        # doesn't unfairly prefer slow tasks that happen to have higher raw reward
        normalised_rewards = batch_rewards / batch_taus
        expected = normalised_rewards - self.rho + (next_state_target_values * self._gamma)

        # update rho as a running avg of r/tau across the batch
        rho_sample = normalised_rewards.mean().item()
        self.rho = (1 - self.rho_lr) * self.rho + self.rho_lr * rho_sample
        self.rho_history.append(self.rho)

        return expected.unsqueeze(1)