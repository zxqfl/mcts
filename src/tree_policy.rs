extern crate rand;
use self::rand::{XorShiftRng, Rng, SeedableRng};

use std;
use super::*;
use search_tree::*;

pub trait TreePolicy<Spec: MCTS<TreePolicy=Self>>: Sync + Sized {
    type MoveEvaluation: Sync;
    type ThreadLocalData: Default;

    fn choose_child<'a, MoveIter>(&self, moves: MoveIter, handle: SearchHandle<Spec>) -> &'a MoveInfo<Spec>
        where MoveIter: Iterator<Item=&'a MoveInfo<Spec>> + Clone;
}

pub struct UCTPolicy {
    exploration_constant: f64,
}

impl UCTPolicy {
    pub fn new(exploration_constant: f64) -> Self {
        assert!(exploration_constant > 0.0,
            "exploration constant is {} (must be positive)",
            exploration_constant);
        Self {exploration_constant}
    }

    pub fn exploration_constant(&self) -> f64 {
        self.exploration_constant
    }
}

pub struct AlphaGoPolicy {
    exploration_constant: f64,
}

impl AlphaGoPolicy {
    /// Creates a new policy with an exploration constant of 5, as in the original AlphaGo paper.
    pub fn new() -> Self {
        Self::with_exploration_constant(5.0)
    }

    pub fn with_exploration_constant(exploration_constant: f64) -> Self {
        assert!(exploration_constant > 0.0,
            "exploration constant is {} (must be positive)",
            exploration_constant);
        Self {exploration_constant}
    }

    pub fn exploration_constant(&self) -> f64 {
        self.exploration_constant
    }
}

impl<Spec: MCTS<TreePolicy=Self>> TreePolicy<Spec> for UCTPolicy
{
    type ThreadLocalData = PolicyRng;
    type MoveEvaluation = ();

    fn choose_child<'a, MoveIter>(&self, moves: MoveIter, mut handle: SearchHandle<Spec>) -> &'a MoveInfo<Spec>
        where MoveIter: Iterator<Item=&'a MoveInfo<Spec>> + Clone
    {
        let total_visits = moves.clone().map(|x| x.visits()).sum::<u64>();
        let adjusted_total = (total_visits + 1) as f64;
        let ln_adjusted_total = adjusted_total.ln();
        handle.thread_local_data().policy_data.select_by_key(moves, |mov| {
            let sum_rewards = mov.sum_rewards();
            let child_visits = mov.visits();
            // http://mcts.ai/pubs/mcts-survey-master.pdf
            let explore_term = if child_visits == 0 {
                std::f64::INFINITY
            } else {
                2.0 * (ln_adjusted_total / child_visits as f64).sqrt()
            };
            let mean_action_value = sum_rewards as f64 / adjusted_total;
            self.exploration_constant * explore_term + mean_action_value
        }).unwrap()
    }
}

impl<Spec: MCTS<TreePolicy=Self>> TreePolicy<Spec> for AlphaGoPolicy
{
    type ThreadLocalData = PolicyRng;
    type MoveEvaluation = f64;

    fn choose_child<'a, MoveIter>(&self, moves: MoveIter, mut handle: SearchHandle<Spec>) -> &'a MoveInfo<Spec>
        where MoveIter: Iterator<Item=&'a MoveInfo<Spec>> + Clone
    {
        let total_visits = moves.clone().map(|x| x.visits()).sum::<u64>();
        let sqrt_total_visits = (total_visits as f64).sqrt();
        handle.thread_local_data().policy_data.select_by_key(moves, |mov| {
            let sum_rewards = mov.sum_rewards() as f64;
            let child_visits = mov.visits();
            let adj_child_visits = (child_visits + 1) as f64;
            let policy_evaln = *mov.move_evaluation() as f64;
            (sum_rewards + self.exploration_constant * policy_evaln * sqrt_total_visits)
                / adj_child_visits
        }).unwrap()
    }
}

#[derive(Clone)]
pub struct PolicyRng {
    rng: XorShiftRng
}

impl PolicyRng {
    pub fn new() -> Self {
        let rng = SeedableRng::from_seed([1, 2, 3, 4]);
        Self {rng}
    }

    pub fn select_by_key<T, Iter, KeyFn>(&mut self, elts: Iter, mut key_fn: KeyFn) -> Option<T>
        where Iter: Iterator<Item=T>, KeyFn: FnMut(&T) -> f64
    {
        let mut choice = None;
        let mut num_optimal: u32 = 0;
        let mut best_so_far: f64 = std::f64::NEG_INFINITY;
        for elt in elts {
            let score = key_fn(&elt);
            if score > best_so_far {
                choice = Some(elt);
                num_optimal = 1;
                best_so_far = score;
            } else if score == best_so_far {
                num_optimal += 1;
                if self.rng.gen_weighted_bool(num_optimal) {
                    choice = Some(elt);
                }
            }
        }
        choice
    }
}

impl Default for PolicyRng {
    fn default() -> Self {
        Self::new()
    }
}
