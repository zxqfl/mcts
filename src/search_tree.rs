#[macro_use]
use super::{MCTS, GameState, Evaluator};
use std::sync::atomic::{AtomicIsize, AtomicUsize, AtomicPtr, Ordering};
use std;
use std::ptr::{null, null_mut};
use smallvec::SmallVec;

use tree_policy::TreePolicy;

pub struct SearchTree<Spec: MCTS> {
    root_node: SearchNode<Spec>,
    root_state: Spec::State,
    tree_policy: Spec::TreePolicy,
    eval: Spec::Eval,
    manager: Spec,
    global_data: Spec::GlobalData,
}

pub struct MoveInfo<Spec: MCTS> {
    mov: <<Spec as MCTS>::State as GameState>::Move,
    move_evaluation: f64,
    child: AtomicPtr<SearchNode<Spec>>,
}

pub struct SearchNode<Spec: MCTS> {
    moves: Vec<MoveInfo<Spec>>,
    visits: AtomicUsize,
    sum_evaluations: AtomicIsize,
    data: Spec::NodeData,
    evaln: Option<<<Spec as MCTS>::Eval as Evaluator<Spec>>::StateEvaluation>,
}

impl<Spec: MCTS> SearchNode<Spec> {
    fn new(moves: Vec<MoveInfo<Spec>>,
            evaln: Option<<<Spec as MCTS>::Eval as Evaluator<Spec>>::StateEvaluation>) -> Self {
        Self {
            moves,
            visits: AtomicUsize::new(0),
            sum_evaluations: AtomicIsize::new(0),
            data: Default::default(),
            evaln
        }
    }
}

impl<Spec: MCTS> MoveInfo<Spec> {
    fn new(mov: <<Spec as MCTS>::State as GameState>::Move, move_evaluation: f64) -> Self {
        MoveInfo {
            mov,
            move_evaluation,
            child: AtomicPtr::default(),
        }
    }

    pub fn get_move(&self) -> &<<Spec as MCTS>::State as GameState>::Move {
        &self.mov
    }

    pub fn evaluation(&self) -> f64 {
        self.move_evaluation
    }

    pub fn visits(&self) -> u32 {
        let child = self.child.load(Ordering::Relaxed) as *const SearchNode<Spec>;
        if child == null() {
            0
        } else {
            unsafe {
                (*child).visits.load(Ordering::Relaxed) as u32
            }
        }
    }

    pub fn sum_evaluations(&self) -> i64 {
        let child = self.child.load(Ordering::Relaxed) as *const SearchNode<Spec>;
        if child == null() {
            0
        } else {
            unsafe {
                (*child).sum_evaluations.load(Ordering::Relaxed) as i64
            }
        }
    }
}

fn drop_if_present<T>(ptr: &AtomicPtr<T>) {
    let ptr = ptr.load(Ordering::SeqCst);
    if ptr != null_mut() {
        unsafe {
            std::ptr::drop_in_place(ptr);
        }
    }
}

impl<Spec: MCTS> Drop for MoveInfo<Spec> {
    fn drop(&mut self) {
        drop_if_present(&self.child);
    }
}

impl<Spec: MCTS> SearchTree<Spec> {
    pub fn playout(&self, tld: &mut Spec::ThreadLocalData) {
        const LARGE_DEPTH: usize = 64;
        let mut state = self.root_state.clone();
        let mut path: SmallVec<[*const SearchNode<Spec>; LARGE_DEPTH]> = SmallVec::new();
        path.push(&self.root_node);
        while state.result().is_none() {
            if path.len() == LARGE_DEPTH {
                for i in 0..(LARGE_DEPTH - 1) {
                    assert!(path[i] != path[LARGE_DEPTH - 1],
                        "Cycle detected: disable transposition table \
                        or make the game state acyclic");
                }
            }
            let node = unsafe {
                &*path[path.len() - 1]
            };
            let choice = self.tree_policy.choose_child(&node.moves, self.make_handle(node, tld));
            assert!(choice < node.moves.len(),
                "The index {} chosen by the tree policy is out of the allowed range [0, {})",
                choice, node.moves.len());
            let choice = &node.moves[choice];
            let mut child;
            loop {
                child = choice.child.load(Ordering::Acquire) as *const SearchNode<Spec>;
                if child == null() {
                    let new_child = self.expand_child(&mut state, &choice.mov, self.make_handle(node, tld));
                    let result = choice.child.compare_and_swap(null_mut(), new_child, Ordering::Release);
                    if result == null_mut() {
                        // compare and swap was successful
                        break;
                    } else {
                        // someone else expanded this child already
                        unsafe {
                            std::ptr::drop_in_place(new_child);
                        }
                    }
                }
            }
        }
        self.finish_playout(&state, &path);
    }

    fn finish_playout(&self, state: &Spec::State, path: &[*const SearchNode<Spec>]) {

    }

    fn expand_child(&self, state: &mut Spec::State,
            mov: &<<Spec as MCTS>::State as GameState>::Move, handle: SearchHandle<Spec>)
            -> *mut SearchNode<Spec> {
        state.make_move(mov);
        let moves = state.available_moves();
        let (move_eval, state_eval) = self.eval.evaluate_moves(&state, &moves, handle);
        let moves = moves.into_iter()
            .zip(move_eval.into_iter())
            .map(|(m, e)| MoveInfo::new(m, e))
            .collect();
        let node = SearchNode::new(moves, state_eval);
        Box::into_raw(Box::new(node))
    }

    fn make_handle<'a>(&'a self, node: &'a SearchNode<Spec>, tld: &'a mut Spec::ThreadLocalData)
            -> SearchHandle<'a, Spec> {
        let global_data = &self.global_data;
        SearchHandle {node, tld, global_data}
    }
}

#[derive(Clone, Copy)]
pub struct NodeHandle<'a, Spec: 'a + MCTS> {
    node: &'a SearchNode<Spec>,
}

#[derive(Clone, Copy)]
pub struct OpaqueNode<Spec: MCTS> {
    ptr: *const SearchNode<Spec>,
}

impl<'a, Spec: MCTS> NodeHandle<'a, Spec> {
    pub fn data(&self) -> &'a Spec::NodeData {
        &self.node.data
    }

    pub fn visits(&self) -> u32 {
        self.node.visits.load(Ordering::Relaxed) as u32
    }
}

pub struct SearchHandle<'a, Spec: 'a + MCTS> {
    node: &'a SearchNode<Spec>,
    tld: &'a mut Spec::ThreadLocalData,
    global_data: &'a Spec::GlobalData,
}

impl<'a, Spec: MCTS> SearchHandle<'a, Spec> {
    pub fn node(&self) -> NodeHandle<'a, Spec> {
        NodeHandle {node: self.node}
    }
    pub fn thread_local_data(&mut self) -> &mut Spec::ThreadLocalData {
        self.tld
    }
    pub fn global_data(&self) -> &'a Spec::GlobalData {
        self.global_data
    }
}
