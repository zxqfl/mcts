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
    evaln: <<Spec as MCTS>::Eval as Evaluator<Spec>>::StateEvaluation,
    player: <<Spec as MCTS>::State as GameState>::Player,
}

impl<Spec: MCTS> SearchNode<Spec> {
    fn new(moves: Vec<MoveInfo<Spec>>,
            evaln: <<Spec as MCTS>::Eval as Evaluator<Spec>>::StateEvaluation,
            player: <<Spec as MCTS>::State as GameState>::Player) -> Self {
        Self {
            moves,
            visits: AtomicUsize::new(0),
            sum_evaluations: AtomicIsize::new(0),
            data: Default::default(),
            evaln,
            player,
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

    pub fn visits(&self) -> u64 {
        let child = self.child.load(Ordering::Relaxed) as *const SearchNode<Spec>;
        if child == null() {
            0
        } else {
            unsafe {
                (*child).visits.load(Ordering::Relaxed) as u64
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
        let mut did_we_create = false;
        loop {
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
            if node.moves.len() == 0 {
                break;
            }
            node.sum_evaluations.fetch_sub(self.manager.virtual_loss() as isize, Ordering::Relaxed);
            let visits = node.visits.fetch_add(1, Ordering::Release) + 1;
            if visits as u64 <= self.manager.visits_before_expansion() {
                break;
            }
            let choice = self.tree_policy.choose_child(&node.moves, self.make_handle(node, tld));
            assert!(choice < node.moves.len(),
                "The index {} chosen by the tree policy is out of the allowed range [0, {})",
                choice, node.moves.len());
            let choice = &node.moves[choice];
            let mut child;
            loop {
                child = choice.child.load(Ordering::Acquire) as *const SearchNode<Spec>;
                did_we_create = false;
                if child == null() {
                    let new_child = self.expand_child(&mut state, &choice.mov, self.make_handle(node, tld));
                    let result = choice.child.compare_and_swap(null_mut(), new_child, Ordering::Release);
                    if result == null_mut() {
                        // compare and swap was successful
                        did_we_create = true;
                        child = new_child;
                        break;
                    } else {
                        // someone else expanded this child before we did
                        unsafe {
                            std::ptr::drop_in_place(new_child);
                        }
                    }
                } else {
                    break;
                }
            }
            assert!(child != null());
            path.push(child);
        }
        self.finish_playout(did_we_create, &state, &path, tld);
    }

    fn finish_playout(&self, did_we_create: bool, state: &Spec::State,
            path: &[*const SearchNode<Spec>], tld: &mut Spec::ThreadLocalData) {
        let last = unsafe {
            &*path[path.len() - 1]
        };
        let new_evaln = if did_we_create {
            None
        } else {
            Some(self.eval.evaluate_existing_state(state, &last.evaln, self.make_handle(last, tld)))
        };
        let evaln = new_evaln.as_ref().unwrap_or(&last.evaln);
        for &ptr in path.iter().rev() {
            let node = unsafe {
                &*ptr
            };
            let delta =
                  self.eval.interpret_evaluation_for_player(evaln, &node.player)
                + self.manager.virtual_loss();
            node.sum_evaluations.fetch_add(delta as isize, Ordering::Relaxed);
            self.manager.on_backpropagation(&evaln, self.make_handle(node, tld));
        }
    }

    fn expand_child(&self, state: &mut Spec::State,
            mov: &<<Spec as MCTS>::State as GameState>::Move, handle: SearchHandle<Spec>)
            -> *mut SearchNode<Spec> {
        state.make_move(mov);
        let moves = state.available_moves();
        let (move_eval, state_eval) = self.eval.evaluate_new_state(&state, &moves, handle);
        let moves = moves.into_iter()
            .zip(move_eval.into_iter())
            .map(|(m, e)| MoveInfo::new(m, e))
            .collect();
        let player = state.current_player();
        let node = SearchNode::new(moves, state_eval, player);
        Box::into_raw(Box::new(node))
    }

    fn make_handle<'a>(&'a self, node: &'a SearchNode<Spec>, tld: &'a mut Spec::ThreadLocalData)
            -> SearchHandle<'a, Spec> {
        let global_data = &self.global_data;
        SearchHandle {node, tld, global_data}
    }

    pub fn principal_variation(&mut self) -> Vec<<<Spec as MCTS>::State as GameState>::Move> {
        let mut result = Vec::new();
        let mut crnt = &self.root_node;
        while crnt.moves.len() != 0 {
            let choice = self.manager.select_child_after_search(&crnt.moves);
            result.push(choice.mov.clone());
            let child = choice.child.load(Ordering::SeqCst) as *const SearchNode<Spec>;
            if child == null() {
                break;
            } else {
                unsafe {
                    crnt = &*child;
                }
            }
        }
        result
    }
}

#[derive(Clone, Copy)]
pub struct NodeHandle<'a, Spec: 'a + MCTS> {
    node: &'a SearchNode<Spec>,
}

impl<'a, Spec: MCTS> NodeHandle<'a, Spec> {
    pub fn data(&self) -> &'a Spec::NodeData {
        &self.node.data
    }

    pub fn visits(&self) -> u64 {
        self.node.visits.load(Ordering::Relaxed) as u64
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
