
//! This is a library for Monte Carlo tree search. 
//! 
//! It is still under development and the documentation isn't good. However, the following example may be helpful:
//!
//! ```
//! use mcts::*;
//! use mcts::tree_policy::*;
//! 
//! #[derive(Clone, Debug, PartialEq)]
//! struct CountingGame(i64);
//! 
//! #[derive(Clone, Debug, PartialEq)]
//! enum Move {
//!     Add, Sub
//! }
//! 
//! impl GameState for CountingGame {
//!     type Move = Move;
//!     type Player = ();
//! 
//!     fn current_player(&self) -> Self::Player {
//!         ()
//!     }
//!     fn available_moves(&self) -> Vec<Self::Move> {
//!         let x = self.0;
//!         if x == 100 {
//!             vec![]
//!         } else {
//!             vec![Move::Add, Move::Sub]
//!         }
//!     }
//!     fn make_move(&mut self, mov: &Self::Move) {
//!         match *mov {
//!             Move::Add => self.0 += 1,
//!             Move::Sub => self.0 -= 1,
//!         }
//!     }
//! }
//! 
//! struct MyEvaluator {}
//! 
//! impl Evaluator<MyMCTS> for MyEvaluator {
//!     type StateEvaluation = i64;
//! 
//!     fn evaluate_new_state(&self, state: &CountingGame, moves: &[Move],
//!         _: Option<SearchHandle<MyMCTS>>)
//!         -> (Vec<f64>, i64) {
//!         (moves.iter().map(|_| 0.0).collect(), state.0)
//!     }
//!     fn interpret_evaluation_for_player(&self, evaln: &i64, _player: &()) -> i64 {
//!         *evaln
//!     }
//!     fn evaluate_existing_state(&self, _: &CountingGame,  evaln: &i64, _: SearchHandle<MyMCTS>) -> i64 {
//!         *evaln
//!     }
//! }
//! 
//! struct MyMCTS {}
//! 
//! impl MCTS for MyMCTS {
//!     type State = CountingGame;
//!     type Eval = MyEvaluator;
//!     type NodeData = ();
//!     type ThreadLocalData = PolicyRng;
//!     type GlobalData = ();
//!     type TreePolicy = UCTPolicy;
//! }
//! 
//! let game = CountingGame(0);
//! let mut mcts = MCTSManager::new(game, MyMCTS{}, UCTPolicy::new(0.5), MyEvaluator{});
//! mcts.playout_n_parallel(100000, 4);
//! assert_eq!(mcts.principal_variation(5),
//!     vec![Move::Add, Move::Add, Move::Add, Move::Add, Move::Add]);
//! assert_eq!(mcts.principal_variation_states(5),
//!     vec![
//!         CountingGame(0),
//!         CountingGame(1),
//!         CountingGame(2),
//!         CountingGame(3),
//!         CountingGame(4),
//!         CountingGame(5)]);
//! ```

extern crate crossbeam;
extern crate smallvec;

mod search_tree;
pub mod tree_policy;

pub use search_tree::*;
use tree_policy::*;

use std::sync::atomic::{AtomicIsize, AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

pub trait MCTS: Sized + Sync {
    type State: GameState + Sync;
    type Eval: Evaluator<Self>;
    type TreePolicy: TreePolicy<Self>;
    type NodeData: Default + Sync;
    type ThreadLocalData: Sync;
    type GlobalData: Default + Sync;

    fn virtual_loss(&self) -> i64 {
        0
    }
    fn visits_before_expansion(&self) -> u64 {
        1
    }
    fn select_child_after_search<'a>(&self, children: &'a [MoveInfo<Self>]) -> &'a MoveInfo<Self> {
        children.into_iter().max_by_key(|child| child.visits()).unwrap()
    }

    // fn add_state_to_transposition_table<'a>(&'a self, _state: &Self::State, _node: NodeHandle<'a, Self>,
    //     _handle: SearchHandle<Self>) {}

    // fn lookup_transposition_table<'a>(&'a self, _state: &Self::State) -> Option<NodeHandle<'a, Self>> {
    //     None
    // }

    fn on_backpropagation(&self, _evaln: &StateEvaluation<Self>, _handle: SearchHandle<Self>) {}
}

pub type StateEvaluation<Spec> = <<Spec as MCTS>::Eval as Evaluator<Spec>>::StateEvaluation;
pub type Move<Spec> = <<Spec as MCTS>::State as GameState>::Move;
pub type Player<Spec> = <<Spec as MCTS>::State as GameState>::Player;

pub trait GameState: Clone {
    type Move: Sync + Clone;
    type Player: Sync;

    fn current_player(&self) -> Self::Player;
    fn available_moves(&self) -> Vec<Self::Move>;
    fn make_move(&mut self, mov: &Self::Move);
}

pub trait Evaluator<Spec: MCTS>: Sync {
    type StateEvaluation: Sync;

    fn evaluate_new_state(&self,
        state: &Spec::State, moves: &[Move<Spec>],
        handle: Option<SearchHandle<Spec>>)
        -> (Vec<f64>, Self::StateEvaluation);

    fn evaluate_existing_state(&self, state: &Spec::State, existing_evaln: &Self::StateEvaluation,
        handle: SearchHandle<Spec>)
        -> Self::StateEvaluation;

    fn interpret_evaluation_for_player(&self,
        evaluation: &Self::StateEvaluation,
        player: &Player<Spec>) -> i64;
}


pub struct MCTSManager<Spec: MCTS> {
    search_tree: SearchTree<Spec>,
    // thread local data when we have no asynchronous workers
    single_threaded_tld: Option<Spec::ThreadLocalData>,
}

impl<Spec: MCTS> MCTSManager<Spec> where Spec::ThreadLocalData: Default {
    pub fn new(state: Spec::State, manager: Spec, tree_policy: Spec::TreePolicy, eval: Spec::Eval)
            -> Self {
        let search_tree = SearchTree::new(state, manager, tree_policy, eval);
        let single_threaded_tld = None;
        Self {search_tree, single_threaded_tld}
    }

    pub fn playout(&mut self) {
        // Avoid overhead of thread creation
        if self.single_threaded_tld.is_none() {
            self.single_threaded_tld = Some(Default::default());
        }
        self.search_tree.playout(self.single_threaded_tld.as_mut().unwrap());
    }
    pub fn playout_until<Predicate: FnMut() -> bool>(&mut self, mut pred: Predicate) {
        while !pred() {
            self.playout();
        }
    }
    pub fn playout_n(&mut self, n: u64) {
        for _ in 0..n {
            self.playout();
        }
    }
    pub fn playout_parallel_async<'a>(&'a mut self, num_threads: usize) -> AsyncSearch<'a, Spec> {
        assert!(num_threads != 0);
        let stop_signal = Arc::new(AtomicBool::new(false));
        let threads = (0..num_threads).map(|_| {
            unsafe {
                let stop_signal = stop_signal.clone();
                let search_tree = &self.search_tree;
                crossbeam::spawn_unsafe(move || {
                    let mut tld = Spec::ThreadLocalData::default();
                    loop {
                        if stop_signal.load(Ordering::SeqCst) {
                            break;
                        }
                        search_tree.playout(&mut tld);
                    }
                })
            }
        }).collect();
        AsyncSearch {
            manager: self,
            stop_signal,
            threads,
        }
    }
    pub fn playout_parallel_for(&mut self, duration: Duration, num_threads: usize) {
        let search = self.playout_parallel_async(num_threads);
        std::thread::sleep(duration);
        search.halt();
    }
    pub fn playout_n_parallel(&mut self, n: u32, num_threads: usize) {
        if n == 0 {
            return;
        }
        assert!(num_threads != 0);
        let counter = AtomicIsize::new(n as isize);
        crossbeam::scope(|scope| {
            for _ in 0..num_threads {
                scope.spawn(|| {
                    let mut tld = Spec::ThreadLocalData::default();
                    loop {
                        let count = counter.fetch_sub(1, Ordering::SeqCst);
                        if count <= 0 {
                            break;
                        }
                        self.search_tree.playout(&mut tld);
                    }
                });
            }
        });
    }
    pub fn principal_variation(&self, num_moves: usize) -> Vec<Move<Spec>> {
        self.search_tree.principal_variation(num_moves)
    }
    pub fn principal_variation_states(&self, num_moves: usize)
            -> Vec<Spec::State> {
        let moves = self.principal_variation(num_moves);
        let mut states = vec![self.search_tree.root_state().clone()];
        for mov in moves {
            let mut state = states[states.len() - 1].clone();
            state.make_move(&mov);
            states.push(state);
        }
        states
    }
    pub fn tree(&self) -> &SearchTree<Spec> {&self.search_tree}
}

#[must_use]
pub struct AsyncSearch<'a, Spec: 'a + MCTS> {
    manager: &'a mut MCTSManager<Spec>,
    stop_signal: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

impl<'a, Spec: MCTS> AsyncSearch<'a, Spec> {
    pub fn halt(self) -> &'a  MCTSManager<Spec> {
        self.manager
    }
}

impl<'a, Spec: MCTS> Drop for AsyncSearch<'a, Spec> {
    fn drop(&mut self) {
        self.stop_signal.store(true, Ordering::SeqCst);
        for t in self.threads.drain(..) {
            t.join().unwrap();
        }
    }
}
