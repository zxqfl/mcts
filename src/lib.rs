
//! This is a library for Monte Carlo tree search. 
//! 
//! It is still under development and the documentation isn't good. However, the following example may be helpful:
//!
//! ```
//! use mcts::*;
//! use mcts::tree_policy::*;
//! use mcts::transposition_table::*;
//! 
//! // A really simple game. There's one player and one number. In each move the player can
//! // increase or decrease the number. The player's score is the number.
//! // The game ends when the number reaches 100.
//! // 
//! // The best strategy is to increase the number at every step.
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
//!     type MoveList = Vec<Move>;
//! 
//!     fn current_player(&self) -> Self::Player {
//!         ()
//!     }
//!     fn available_moves(&self) -> Vec<Move> {
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
//! impl TranspositionHash for CountingGame {
//!     fn hash(&self) -> u64 {
//!         self.0 as u64
//!     }
//! }
//! 
//! struct MyEvaluator;
//! 
//! impl Evaluator<MyMCTS> for MyEvaluator {
//!     type StateEvaluation = i64;
//! 
//!     fn evaluate_new_state(&self, state: &CountingGame, moves: &Vec<Move>,
//!         _: Option<SearchHandle<MyMCTS>>)
//!         -> (Vec<()>, i64) {
//!         (vec![(); moves.len()], state.0)
//!     }
//!     fn interpret_evaluation_for_player(&self, evaln: &i64, _player: &()) -> i64 {
//!         *evaln
//!     }
//!     fn evaluate_existing_state(&self, _: &CountingGame,  evaln: &i64, _: SearchHandle<MyMCTS>) -> i64 {
//!         *evaln
//!     }
//! }
//! 
//! #[derive(Default)]
//! struct MyMCTS;
//! 
//! impl MCTS for MyMCTS {
//!     type State = CountingGame;
//!     type Eval = MyEvaluator;
//!     type NodeData = ();
//!     type ExtraThreadData = ();
//!     type TreePolicy = UCTPolicy;
//!     type TranspositionTable = LossyQuadraticProbingHashTableForMCTS<Self>;
//! }
//! 
//! let game = CountingGame(0);
//! let mut mcts = MCTSManager::new(game, MyMCTS, MyEvaluator, UCTPolicy::new(0.5));
//! mcts.playout_n_parallel(100000, 4);
//! mcts.tree().debug_moves();
//! assert_eq!(mcts.principal_variation(50),
//!     vec![Move::Add; 50]);
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
mod atomics;
pub mod tree_policy;
pub mod transposition_table;

pub use search_tree::*;
use tree_policy::*;
use transposition_table::*;

use atomics::*;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

pub trait MCTS: Sized + Sync {
    type State: GameState + Sync;
    type Eval: Evaluator<Self>;
    type TreePolicy: TreePolicy<Self>;
    type NodeData: Default + Sync;
    type TranspositionTable: TranspositionTable<Self>;
    type ExtraThreadData;

    fn virtual_loss(&self) -> i64 {
        0
    }
    fn visits_before_expansion(&self) -> u64 {
        1
    }
    fn node_limit(&self) -> usize {
        std::usize::MAX
    }
    fn select_child_after_search<'a>(&self, children: &'a [MoveInfo<Self>]) -> &'a MoveInfo<Self> {
        children.into_iter().max_by_key(|child| child.visits()).unwrap()
    }

    fn on_backpropagation(&self, _evaln: &StateEvaluation<Self>, _handle: SearchHandle<Self>) {}
}

pub struct ThreadData<Spec: MCTS> {
    pub policy_data: TreePolicyThreadData<Spec>,
    pub extra_data: Spec::ExtraThreadData,
}

impl<Spec: MCTS> Default for ThreadData<Spec>
    where TreePolicyThreadData<Spec>: Default, Spec::ExtraThreadData: Default
{
    fn default() -> Self {
        Self {
            policy_data: Default::default(),
            extra_data: Default::default(),
        }
    }
} 

pub type MoveEvaluation<Spec> = <<Spec as MCTS>::TreePolicy as TreePolicy<Spec>>::MoveEvaluation;
pub type StateEvaluation<Spec> = <<Spec as MCTS>::Eval as Evaluator<Spec>>::StateEvaluation;
pub type Move<Spec> = <<Spec as MCTS>::State as GameState>::Move;
pub type MoveList<Spec> = <<Spec as MCTS>::State as GameState>::MoveList;
pub type Player<Spec> = <<Spec as MCTS>::State as GameState>::Player;
pub type TreePolicyThreadData<Spec> = <<Spec as MCTS>::TreePolicy as TreePolicy<Spec>>::ThreadLocalData;

pub trait GameState: Clone {
    type Move: Sync + Clone;
    type Player: Sync;
    type MoveList: std::iter::IntoIterator<Item=Self::Move>;

    fn current_player(&self) -> Self::Player;
    fn available_moves(&self) -> Self::MoveList;
    fn make_move(&mut self, mov: &Self::Move);
}

pub trait Evaluator<Spec: MCTS>: Sync {
    type StateEvaluation: Sync;

    fn evaluate_new_state(&self,
        state: &Spec::State, moves: &MoveList<Spec>,
        handle: Option<SearchHandle<Spec>>)
        -> (Vec<MoveEvaluation<Spec>>, Self::StateEvaluation);

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
    single_threaded_tld: Option<ThreadData<Spec>>,
    print_on_playout_error: bool,
}

impl<Spec: MCTS> MCTSManager<Spec> where ThreadData<Spec>: Default {
    pub fn new(state: Spec::State, manager: Spec, eval: Spec::Eval, tree_policy: Spec::TreePolicy,
            table: Spec::TranspositionTable) -> Self {
        let search_tree = SearchTree::new(state, manager, tree_policy, eval, table);
        let single_threaded_tld = None;
        Self {search_tree, single_threaded_tld, print_on_playout_error: true}
    }

    pub fn print_on_playout_error(&mut self, v: bool) -> &mut Self {
        self.print_on_playout_error = v;
        self
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
    unsafe fn spawn_worker_thread(&self, stop_signal: Arc<AtomicBool>) -> JoinHandle<()> {
        let search_tree = &self.search_tree;
        let print_on_playout_error = self.print_on_playout_error;
        crossbeam::spawn_unsafe(move || {
            let mut tld = Default::default();
            loop {
                if stop_signal.load(Ordering::SeqCst) {
                    break;
                }
                if !search_tree.playout(&mut tld) {
                    if print_on_playout_error {
                        eprintln!("Node limit of {} reached. Halting search.",
                            search_tree.spec().node_limit());
                    }
                    break;
                }
            }
        })
    }
    pub fn playout_parallel_async<'a>(&'a mut self, num_threads: usize) -> AsyncSearch<'a, Spec> {
        assert!(num_threads != 0);
        let stop_signal = Arc::new(AtomicBool::new(false));
        let threads = (0..num_threads).map(|_| {
            let stop_signal = stop_signal.clone();
            unsafe {
                self.spawn_worker_thread(stop_signal)
            }
        }).collect();
        AsyncSearch {
            manager: self,
            stop_signal,
            threads,
        }
    }
    pub fn into_playout_parallel_async(self, num_threads: usize) -> AsyncSearchOwned<Spec> {
        assert!(num_threads != 0);
        let self_box = Box::new(self);
        let stop_signal = Arc::new(AtomicBool::new(false));
        let threads = (0..num_threads).map(|_| {
            let stop_signal = stop_signal.clone();
            unsafe {
                self_box.spawn_worker_thread(stop_signal)
            }
        }).collect();
        AsyncSearchOwned {
            manager: Some(self_box),
            stop_signal,
            threads
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
        let search_tree = &self.search_tree;
        crossbeam::scope(|scope| {
            for _ in 0..num_threads {
                scope.spawn(|| {
                    let mut tld = Default::default();
                    loop {
                        let count = counter.fetch_sub(1, Ordering::SeqCst);
                        if count <= 0 {
                            break;
                        }
                        search_tree.playout(&mut tld);
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
    pub fn best_move(&self) -> Option<Move<Spec>> {
        self.principal_variation(1).get(0).map(|x| x.clone())
    }
    pub fn perf_test<F>(&mut self, num_threads: usize, mut f: F) where F: FnMut(usize) {
        let search = self.playout_parallel_async(num_threads);
        for _ in 0..10 {
            let n1 = search.manager.search_tree.node_count();
            std::thread::sleep(Duration::from_secs(1));
            let n2 = search.manager.search_tree.node_count();
            let diff = if n2 > n1 {
                n2 - n1
            } else {
                0
            };
            f(diff);
        }
    }
    pub fn perf_test_to_stderr(&mut self, num_threads: usize) {
        self.perf_test(num_threads, |x| eprintln!("{} nodes/sec", thousands_separate(x)));
    }
    pub fn reset(self) -> Self {
        Self {
            search_tree: self.search_tree.reset(),
            print_on_playout_error: self.print_on_playout_error,
            single_threaded_tld: None,
        }
    }
}

// https://stackoverflow.com/questions/26998485/rust-print-format-number-with-thousand-separator
fn thousands_separate(x: usize) -> String {
    let s = format!("{}", x);
    let bytes: Vec<_> = s.bytes().rev().collect();
    let chunks: Vec<_> = bytes.chunks(3).map(|chunk| String::from_utf8(chunk.to_vec()).unwrap()).collect();
    let result: Vec<_> = chunks.join(",").bytes().rev().collect();
    String::from_utf8(result).unwrap()
}

#[must_use]
pub struct AsyncSearch<'a, Spec: 'a + MCTS> {
    manager: &'a mut MCTSManager<Spec>,
    stop_signal: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

impl<'a, Spec: MCTS> AsyncSearch<'a, Spec> {
    pub fn halt(self) {}
    pub fn num_threads(&self) -> usize {
        self.threads.len()
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

#[must_use]
pub struct AsyncSearchOwned<Spec: MCTS> {
    manager: Option<Box<MCTSManager<Spec>>>,
    stop_signal: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

impl<Spec: MCTS> AsyncSearchOwned<Spec> {
    fn stop_threads(&mut self) {
        self.stop_signal.store(true, Ordering::SeqCst);
        for t in self.threads.drain(..) {
            t.join().unwrap();
        }
    }
    pub fn halt(mut self) -> MCTSManager<Spec> {
        self.stop_threads();
        *self.manager.take().unwrap()
    }
    pub fn num_threads(&self) -> usize {
        self.threads.len()
    }
}

impl<Spec: MCTS> Drop for AsyncSearchOwned<Spec> {
    fn drop(&mut self) {
        self.stop_threads();
    }
}

impl<Spec: MCTS> From<MCTSManager<Spec>> for AsyncSearchOwned<Spec> {
    /// An `MCTSManager` is an `AsyncSearchOwned` with zero threads searching.
    fn from(m: MCTSManager<Spec>) -> Self {
        Self {
            manager: Some(Box::new(m)),
            stop_signal: Arc::new(AtomicBool::new(false)),
            threads: Vec::new(),
        }
    }
}
