extern crate rand;
use self::rand::{XorShiftRng, Rng, SeedableRng};

use std;
use super::*;
use search_tree::*;

pub trait TreePolicy<Spec: MCTS>: Sync {
    type ThreadLocalData;

    fn choose_child(&self, moves: &[MoveInfo<Spec>], handle: SearchHandle<Spec>) -> usize;
}

pub struct UCTPolicy {
    exploration_constant: f64,
}

impl UCTPolicy {
    pub fn new(exploration_constant: f64) -> Self {
        assert!(exploration_constant > 0.0,
            "exploration constant is {} (must be positive)",
            exploration_constant);
        UCTPolicy {exploration_constant}
    }

    pub fn exploration_constant(&self) -> f64 {
        self.exploration_constant
    }
}

impl<Spec: MCTS> TreePolicy<Spec> for UCTPolicy
where for<'a> (&'a mut Spec::ThreadLocalData): Into<&'a mut PolicyRng>
{
    type ThreadLocalData = PolicyRng;

    fn choose_child(&self, moves: &[MoveInfo<Spec>], mut handle: SearchHandle<Spec>) -> usize {
        assert!(moves.len() != 0);
        let mut choice: usize = std::usize::MAX;
        let mut num_optimal: u32 = 0;
        let mut best_so_far: f64 = std::f64::NEG_INFINITY;
        let node = handle.node();
        let total_visits = node.visits();
        let rng = &mut handle.thread_local_data().into().rng;
        for (index, mov) in moves.iter().enumerate() {
            let sum_evaluations = mov.sum_evaluations();
            let child_visits = mov.visits();
            // http://mcts.ai/pubs/mcts-survey-master.pdf
            let adjusted_total = (total_visits + 1) as f64;
            let explore_term = if child_visits == 0 {
                std::f64::INFINITY
            } else {
                2.0 * (adjusted_total.ln() / child_visits as f64).sqrt()
            };
            let average_reward = sum_evaluations as f64 / adjusted_total;
            let score =
                  self.exploration_constant * explore_term
                + average_reward;
            if score > best_so_far {
                best_so_far = score;
                choice = index;
                num_optimal = 1;
            } else if score == best_so_far {
                num_optimal += 1;
                if rng.gen_weighted_bool(num_optimal) {
                    choice = index;
                }
            }
        }
        choice
    }
}

pub struct PolicyRng {
    rng: XorShiftRng
}

impl Default for PolicyRng {
    fn default() -> Self {
        let rng = SeedableRng::from_seed([1, 2, 3, 4]);
        Self {rng}
    }
}
